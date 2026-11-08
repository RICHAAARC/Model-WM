"""Local engineering checks only: no real weights, generation or scientific result."""
import json
import math
import tempfile
import unittest
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qualification.protocol import build_manifest
from qualification.records import Ledger, write_json
from qualification.metrics import mix_rgb, quality, native_gate, interaction
from colab_bootstrap import receipt, run_colab, Supervisor

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_manifest_and_physical_denominators(self):
        m = build_manifest()
        self.assertEqual(m, json.loads((ROOT/"qualification/manifest.json").read_text()))
        self.assertEqual(len(m["trajectories"]), 44)
        self.assertEqual(len({t["trajectory_id"] for t in m["trajectories"]}), 44)
        self.assertEqual(len({w["source_group"] for w in m["works"]}), 8)
        self.assertEqual(sum(t["bits"] is None for t in m["trajectories"]), 8)
        self.assertEqual(sum(t["role"].startswith("registered") for t in m["trajectories"]), 16)
        self.assertEqual(sum(t["role"].startswith("foreign") for t in m["trajectories"]), 16)
        for w in m["works"]:
            rows = [t for t in m["trajectories"] if t["work_id"] == w["work_id"]]
            self.assertEqual({t["noise_seed"] for t in rows}, {w["noise_seed"]})
            self.assertEqual({t["prompt"] for t in rows}, {w["prompt"]})
            for t in rows:
                if t["role"].startswith(("registered", "foreign")):
                    source = t["role"].rsplit("_", 1)[0]
                    self.assertEqual(t["bits"], m[source][t["user"]][w["position"]*48:(w["position"]+1)*48])
        for b in range(2):
            works = [w for w in m["works"] if w["bundle"] == b]
            self.assertEqual({tuple(w["registered_pair"]) for w in works}, {tuple(works[0]["registered_pair"])})
            self.assertEqual({tuple(w["foreign_pair"]) for w in works}, {tuple(works[0]["foreign_pair"])})

    def test_probe_flips_only_predeclared_positions(self):
        probes = {t["role"]: t["bits"] for t in build_manifest()["trajectories"] if t["role"].startswith("probe")}
        for name, changed in [("probe_j", [0]), ("probe_h", [17]), ("probe_jh", [0, 17])]:
            self.assertEqual([i for i, (a, b) in enumerate(zip(probes["probe_k"], probes[name])) if a != b], changed)

    def test_started_and_failed_slots_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Ledger(d, build_manifest())
            ledger.start(0)
            self.assertEqual(json.loads(ledger.path.read_text())[0]["status"], "STARTED")
            with self.assertRaises(RuntimeError):
                ledger.start(0)
            ledger.end(0, "FAILED", error="injected control failure")
            with self.assertRaises(RuntimeError):
                ledger.start(0)
            with self.assertRaises(RuntimeError):
                Ledger(d, build_manifest())
            self.assertEqual(sum(x["status"] == "NOT_STARTED" for x in ledger.rows), 43)

    def test_existing_output_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)
            root = parent/build_manifest()["run_id"]
            root.mkdir()
            expected = dict(status="PRESERVED", output=str(root), qualification_pass=False)
            write_json(root/"summary.json", expected)
            before = (root/"summary.json").read_bytes()
            with patch("colab_bootstrap.DRIVE_PARENT", parent), patch("colab_bootstrap.subprocess.run", side_effect=AssertionError("no device or run call expected")):
                self.assertEqual(run_colab(), expected)
            self.assertEqual(before, (root/"summary.json").read_bytes())


class MetricTests(unittest.TestCase):
    def test_pixel_average_uses_rint_ties_and_avoids_uint8_overflow(self):
        a = np.array([0, 1, 254, 255], dtype=np.uint8)
        b = np.array([1, 2, 255, 255], dtype=np.uint8)
        np.testing.assert_array_equal(mix_rgb(a, b), [0, 2, 254, 255])

    def test_quality_scale_and_identical_image(self):
        a = np.full((16, 16, 3), 127, dtype=np.uint8)
        identical = quality(a, a)
        self.assertEqual(identical["ssim"], 1.)
        self.assertTrue(identical["psnr_infinite"])
        changed = quality(a, a+1)
        self.assertEqual(changed["mse"], 1.)
        self.assertAlmostEqual(changed["psnr_db"], 20*math.log10(255))

    def test_missing_native_copy_never_reduces_denominator(self):
        m = build_manifest()
        rows = [dict(trajectory_id=f"{s}{i}", source=s, correct_bits=48, psnr_db=30.,
                     psnr_infinite=False, ssim=.9) for s in ["registered", "foreign"] for i in range(16)]
        complete = native_gate(rows, m["gates"])
        self.assertTrue(all(x["passed"] for x in complete.values()))
        missing = native_gate(rows[1:], m["gates"])
        self.assertFalse(missing["registered"]["passed"])
        self.assertEqual(missing["registered"]["denominator"], 16)
        self.assertEqual(missing["registered"]["bit_accuracy"], 15/16)
        # The per-copy 95% condition means all 16 at this denominator.
        rows[0]["psnr_db"] = 19.
        self.assertFalse(native_gate(rows, m["gates"])["registered"]["passed"])

    def test_zero_probe_denominator_stays_unresolved(self):
        names = ["probe_k", "probe_j", "probe_h", "probe_jh"]
        row = interaction({n: np.zeros((4, 4, 3)) for n in names}, {n: np.zeros((48, 2)) for n in names})
        self.assertEqual(row["status"], "UNRESOLVED")
        self.assertIsNone(row["relative_interaction"])

    def test_budget_margin_is_a_stop(self):
        supervisor = object.__new__(Supervisor)
        supervisor.manifest = build_manifest()
        baseline = dict(wall_seconds=1, gpu_occupancy_seconds=1, cpu_seconds=1, peak_ram_bytes=1,
                        peak_device_used_bytes=1, download_charge_bytes=1, result_bytes=1)
        self.assertIsNone(supervisor.reason(baseline))
        self.assertIn("gpu_occupancy_seconds", supervisor.reason(dict(baseline, gpu_occupancy_seconds=1780)))


class NotebookTests(unittest.TestCase):
    def test_colab_entry_is_thin_and_unexecuted(self):
        import nbformat
        nb = nbformat.read(ROOT/"notebooks/aqualora_qualification_colab.ipynb", as_version=4)
        nbformat.validate(nb)
        codes = [c for c in nb.cells if c.cell_type == "code"]
        self.assertEqual(codes[0].source, "from google.colab import drive\ndrive.mount('/content/drive')")
        for i, cell in enumerate(codes):
            compile(cell.source, f"cell{i}", "exec")
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
        self.assertEqual(sum("run_first_run(started_unix=" in c.source for c in codes), 1)
        self.assertEqual(nb.metadata["accelerator"], "GPU")

    def test_existing_checkout_and_cached_entry_receive_the_fix(self):
        # Exercise the exact source cell with a local Git origin and a harmless
        # bootstrap module. No Drive mount, model loader, or run cell is invoked.
        nb = json.loads((ROOT/"notebooks/aqualora_qualification_colab.ipynb").read_text())
        source = next("".join(c["source"]) for c in nb["cells"]
                      if c["cell_type"] == "code" and "SOURCE_BRANCH" in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as d:
            origin, checkout = Path(d)/"origin", Path(d)/"checkout"
            def git(*args):
                return subprocess.check_output(["git", *map(str, args)], stderr=subprocess.STDOUT, text=True)
            git("init", "-b", "main", origin)
            git("-C", origin, "config", "user.name", "Interface regression")
            git("-C", origin, "config", "user.email", "regression@example.invalid")
            module = origin/"colab_bootstrap.py"
            module.write_text("def run_first_run():\n    return 'initial'\n")
            git("-C", origin, "add", "colab_bootstrap.py")
            git("-C", origin, "commit", "-m", "old entry")
            # file:// makes the first clone genuinely shallow, as in Colab.
            code = source.replace("https://github.com/RICHAAARC/Model-WM.git", origin.as_uri())
            code = code.replace("/content/model-wm-code", str(checkout))
            old_path = sys.path.copy()
            cached = sys.modules.pop("colab_bootstrap", None)
            try:
                scope = {}
                exec(compile(code, "notebook_source_cell", "exec"), scope)
                self.assertEqual(scope["run_first_run"](), "initial")
                self.assertEqual(git("-C", checkout, "rev-parse", "--is-shallow-repository").strip(), "true")
                # Different source length also invalidates Python's timestamp pyc.
                module.write_text("def run_first_run():\n    return 'corrected entry'\n")
                git("-C", origin, "add", "colab_bootstrap.py")
                git("-C", origin, "commit", "-m", "corrected entry")
                exec(compile(code, "notebook_source_cell", "exec"), scope)
                self.assertEqual(scope["run_first_run"](), "corrected entry")
                self.assertEqual(git("-C", checkout, "rev-parse", "HEAD"), git("-C", origin, "rev-parse", "HEAD"))
            finally:
                sys.path[:] = old_path
                sys.modules.pop("colab_bootstrap", None)
                if cached is not None:
                    sys.modules["colab_bootstrap"] = cached


class SupervisorTests(unittest.TestCase):
    def supervisor(self, parent):
        root, scratch = Path(parent)/"drive", Path(parent)/"runtime"
        root.mkdir()
        scratch.mkdir()
        return Supervisor(root, scratch, build_manifest(), 0)

    def test_budget_stop_terminates_child_and_saves_reason(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.supervisor(d)
            with patch.object(s, "usage", return_value={"test_only": True}), patch.object(s, "reason", side_effect=[None, "injected budget stop"]):
                with self.assertRaisesRegex(RuntimeError, "injected budget stop"):
                    s.command("test_sleep", [sys.executable, "-c", "import time; time.sleep(30)"])
            self.assertIsNotNone(s.logs[0]["returncode"])
            self.assertTrue((s.root/"STOP_REQUESTED.json").is_file())
            self.assertTrue((s.root/"setup_stages.json").is_file())

    def test_failed_child_is_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.supervisor(d)
            with patch.object(s, "usage", return_value={}), patch.object(s, "reason", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "exited 7"):
                    s.command("test_fail", [sys.executable, "-c", "print('failure retained', flush=True); raise SystemExit(7)"])
            self.assertEqual(len(s.logs), 1)
            self.assertEqual(s.logs[0]["returncode"], 7)
            self.assertIn("failure retained", (s.root/"test_fail.log").read_text())

    def test_nonfinite_evidence_cannot_be_written_as_json(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                write_json(Path(d)/"invalid.json", {"score": float("nan")})


if __name__ == "__main__":
    unittest.main()
