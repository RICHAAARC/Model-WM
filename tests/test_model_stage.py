"""Synthetic CPU controls for the sole model-stage handoff; no real weights/GPU."""
import io
import json
import subprocess
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import colab_bootstrap as boot
from qualification import model_stage as stage, environment_check, prepare
from qualification.python_runtime import ASSET_API
from qualification.protocol import build_manifest
from qualification.records import Ledger, write_json
from qualification.runtime_cache import charged_bytes, paid_links
from tests import test_environment_recovery


class ModelStageTests(unittest.TestCase):
    def fixture(self, folder):
        parent, attempt = test_environment_recovery.RecoveryTests().fixture(folder, ledger=False)
        original = json.loads((attempt/"summary.json").read_text())
        ready = attempt/boot.RECOVERY_NAME
        ready.mkdir()
        write_json(ready/"summary.json", dict(status="ENVIRONMENT_READY", error=None, output=str(ready),
            attempt_output=str(attempt), resources_final=True, environment_only=True,
            trajectories_started=0, trajectories_completed=0, trajectory_denominator=44,
            resources=stage.MINIMUM_READY.copy()))
        write_json(ready/"manifest.json", build_manifest())
        write_json(ready/"carry_forward.json", original)
        write_json(ready/"setup_stages.json", [dict(stage=s, returncode=0) for s in stage.ENV_STAGES])
        (ready/"environment_versions.log").write_text((boot.ROOT/"requirements-colab.txt").read_text())
        (ready/"python_download.jsonl").write_text(json.dumps(dict(status="COMPLETED", version="Python 3.10.14", asset_api=ASSET_API))+"\n")
        return parent, attempt

    def call(self, folder, parent):
        with patch.object(boot, "DRIVE_PARENT", parent), patch.object(boot, "WORK_PARENT", Path(folder)/"runtime"), \
             patch.object(stage.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "synthetic GPU\n")), \
             patch.object(stage.ModelSupervisor, "usage", lambda s, child=None: dict(s.carry)):
            return stage.continue_attempt()

    def test_latest_receipt_carried_exactly_original_history_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, attempt = self.fixture(folder)
            before = {str(p.relative_to(attempt)): p.read_bytes() for p in attempt.rglob("*") if p.is_file()}
            with patch.object(stage.ModelSupervisor, "models") as model:
                result = self.call(folder, parent)
                model.assert_called_once()
            root = attempt/stage.STAGE_NAME
            self.assertEqual(json.loads((root/"carry_forward.json").read_text())["resources"], stage.MINIMUM_READY)
            self.assertEqual(result["resources"], stage.MINIMUM_READY)
            self.assertEqual(result["trajectories_started"], 0)
            self.assertEqual(result["trajectory_denominator"], 44)
            self.assertEqual(before, {str(p.relative_to(attempt)): p.read_bytes() for p in attempt.rglob("*") if p.is_file() and root not in p.parents})
            with patch.object(stage.subprocess, "run", side_effect=AssertionError("read-only, no GPU probe")), \
                 patch.object(boot, "DRIVE_PARENT", parent):
                self.assertEqual(stage.continue_attempt(), result)

    def test_failed_trajectories_preserved_and_no_second_model_invocation(self):
        def injected(s, *args):
            ledger = Ledger(s.root, s.manifest)
            ledger.start(0)
            ledger.end(0, "COMPLETED")
            ledger.start(1)
            ledger.end(1, "FAILED", error="injected model error")
            raise RuntimeError("injected model error")
        with tempfile.TemporaryDirectory() as folder:
            parent, attempt = self.fixture(folder)
            with patch.object(stage.ModelSupervisor, "models", injected):
                result = self.call(folder, parent)
            self.assertEqual(result["status"], "STOPPED_INCOMPLETE")
            self.assertEqual((result["trajectories_started"], result["trajectories_completed"]), (2, 1))
            ledger = attempt/stage.STAGE_NAME/"trajectory_status.json"
            before = ledger.read_bytes()
            with patch.object(stage.ModelSupervisor, "models", side_effect=AssertionError("no retry")):
                reopened = self.call(folder, parent)
            self.assertEqual(reopened, result)
            self.assertEqual(ledger.read_bytes(), before)

    def test_hard_interruption_reconciles_started_ledger_without_resetting_it(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, attempt = self.fixture(folder)
            root = attempt/stage.STAGE_NAME
            root.mkdir()
            write_json(root/"summary.json", dict(status="MODEL_STAGE_RESERVED", output=str(root),
                trajectories_started=0, trajectories_completed=0, resources=stage.MINIMUM_READY, resources_final=False))
            ledger = Ledger(root, build_manifest())
            ledger.start(0)
            before = ledger.path.read_bytes()
            with patch.object(stage.ModelSupervisor, "models", side_effect=AssertionError("no restart")):
                result = self.call(folder, parent)
            self.assertEqual(result["status"], "EXISTING_MODEL_STAGE_READ_ONLY")
            self.assertEqual(result["trajectories_started"], 1)
            self.assertFalse(result["resources_final"])
            self.assertEqual(before, ledger.path.read_bytes())

    def test_missing_gpu_does_not_reserve_model_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, attempt = self.fixture(folder)
            with patch.object(boot, "DRIVE_PARENT", parent), patch.object(stage.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "")):
                with self.assertRaisesRegex(RuntimeError, "one GPU"):
                    stage.continue_attempt()
            self.assertFalse((attempt/stage.STAGE_NAME).exists())

    def test_source_or_resource_regression_stops_before_reservation(self):
        for condition in ("old_cost", "nonfinal", "seed", "stages", "versions", "extra_history", "ledger"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as folder:
                parent, attempt = self.fixture(folder)
                ready = attempt/boot.RECOVERY_NAME
                if condition in ("old_cost", "nonfinal"):
                    summary = json.loads((ready/"summary.json").read_text())
                    if condition == "old_cost":
                        summary["resources"] = boot.MINIMUM_CARRY
                    else:
                        summary["resources_final"] = False
                    write_json(ready/"summary.json", summary)
                elif condition == "seed":
                    manifest = build_manifest()
                    manifest["trajectories"][0]["noise_seed"] += 1
                    write_json(ready/"manifest.json", manifest)
                elif condition == "stages":
                    write_json(ready/"setup_stages.json", [])
                elif condition == "versions":
                    (ready/"environment_versions.log").write_text("torch==2.1.0\n")
                elif condition == "extra_history":
                    (attempt/"environment-recovery-02").mkdir()
                else:
                    Ledger(attempt, build_manifest())
                with self.assertRaises(RuntimeError):
                    stage.inputs(attempt, build_manifest())
                self.assertFalse((attempt/stage.STAGE_NAME).exists())

    def test_exhausted_cumulative_budget_stops_before_download_subprocess(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, attempt = self.fixture(folder)
            path = attempt/boot.RECOVERY_NAME/"summary.json"
            summary = json.loads(path.read_text())
            summary["resources"]["gpu_occupancy_seconds"] = 1780
            write_json(path, summary)
            with patch.object(boot.subprocess, "Popen", side_effect=AssertionError("no download process")), \
                 patch.object(stage.subprocess, "check_output", return_value="synthetic commit"):
                result = self.call(folder, parent)
            self.assertEqual(result["status"], "STOPPED_INCOMPLETE")
            self.assertIn("gpu_occupancy_seconds", result["error"])
            self.assertFalse((attempt/stage.STAGE_NAME/"trajectory_status.json").exists())

    def test_reuses_ready_environment_and_skips_all_installation(self):
        with tempfile.TemporaryDirectory() as folder:
            root, scratch, old = Path(folder)/"result", Path(folder)/"scratch", Path(folder)/"old"
            root.mkdir()
            python = old/"venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("synthetic")
            s = stage.ModelSupervisor(root, scratch, build_manifest(), time.time(), carry=stage.MINIMUM_READY, monitor_gpu=False)
            with patch.object(s, "command") as cmd, patch.object(s, "rebuild_environment", side_effect=AssertionError("no reinstall")):
                s.models(old, dict(resources=boot.MINIMUM_CARRY), "numpy==1.24.3\n")
            self.assertEqual([c.args[0] for c in cmd.call_args_list], ["reuse_environment_check", "reuse_dependency_check",
                "download_preflight", "download_models", "qualification"])
            self.assertEqual(cmd.call_args_list[-1].args[1][0], str(python))
            self.assertEqual(json.loads((root/"environment_plan.json").read_text())["estimated_new_software_bytes"], 0)


class CacheAndBudgetTests(unittest.TestCase):
    def wheel(self, path, name="numpy", version="1.24.3"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as stream:
            stream.writestr(f"{name}-{version}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")

    def test_paid_cache_reused_without_double_charge_new_payload_is_charged(self):
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder)/"old", Path(folder)/"downloads"
            destination.mkdir()
            self.wheel(source/"numpy.whl")
            before = (source/"numpy.whl").read_bytes()
            prepaid = paid_links(source, destination, {"numpy": "1.24.3"})
            self.assertEqual(charged_bytes(destination, prepaid), 0)
            (destination/"new_payload").write_bytes(b"12345")
            self.assertEqual(charged_bytes(destination, prepaid), 5)
            self.assertEqual(before, (source/"numpy.whl").read_bytes())
            (destination/"numpy.whl").unlink()
            (destination/"numpy.whl").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "accounting unresolved"):
                charged_bytes(destination, prepaid)

    def test_unknown_cache_version_stops_instead_of_using_or_redownloading_it(self):
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder)/"old", Path(folder)/"downloads"
            destination.mkdir()
            self.wheel(source/"numpy.whl", version="2.0.0")
            with self.assertRaisesRegex(RuntimeError, "Unknown cached wheel"):
                paid_links(source, destination, {"numpy": "1.24.3"})

    def test_lost_cache_projects_software_plus_models_before_any_payload_get(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root/"environment_plan.json", dict(estimated_new_software_bytes=70))
            response = io.BytesIO()
            response.headers = {"Content-Length": "50"}
            planned = [("https://official.invalid/frozen/weights", root/"downloads/assets/file")]
            with patch.object(prepare.urllib.request, "urlopen", return_value=response) as head:
                with self.assertRaisesRegex(RuntimeError, "remaining original budget"):
                    prepare.preflight(root, root/"downloads", planned, 100, 0, {})
            self.assertEqual(head.call_args.args[0].get_method(), "HEAD")
            self.assertEqual(json.loads((root/"download_preflight.json").read_text())["projected_segment_bytes"], 120)
            self.assertFalse(planned[0][1].exists())

    def test_unknown_asset_cost_stops_and_stream_download_honors_remaining_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root/"environment_plan.json", dict(estimated_new_software_bytes=0))
            response = io.BytesIO()
            response.headers = {}
            planned = [("https://official.invalid/frozen/weights", root/"downloads/assets/file")]
            with patch.object(prepare.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "cost unresolved"):
                    prepare.preflight(root, root/"downloads", planned, 10, 0, {})
            response = io.BytesIO(b"123456")
            response.headers = {"Content-Length": "6"}
            with patch.object(prepare.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "remaining cumulative budget"):
                    prepare.fetch(planned[0][0], planned[0][1], root, root/"downloads", 5, 0)
            self.assertFalse(planned[0][1].exists())

    def test_environment_check_uses_metadata_and_rejects_wrong_package_version(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder)/"lock.txt"
            lock.write_text("torch==2.0.1+cu118\n")
            with patch.object(environment_check.platform, "python_version", return_value="3.10.14"), \
                 patch.object(environment_check.importlib.metadata, "version", return_value="2.0.1+cu118"), \
                 patch("builtins.print"):
                environment_check.check(lock)
            with patch.object(environment_check.platform, "python_version", return_value="3.10.14"), \
                 patch.object(environment_check.importlib.metadata, "version", return_value="2.0.1+cpu"):
                with self.assertRaisesRegex(RuntimeError, "version differs"):
                    environment_check.check(lock)


if __name__ == "__main__":
    unittest.main()
