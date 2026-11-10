"""CPU engineering regressions. Synthetic data/state only, never qualification evidence."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .core import Ledger, effective_limits, native_gates
from .entry import inherit, read_only
from .protocol import build_manifest, freeze


FLOOR = dict(wall_seconds=2500.71440577507, gpu_occupancy_seconds=2500.71440577507,
             cpu_seconds=1962.099122, download_charge_bytes=10791365453,
             peak_ram_bytes=7001604096, peak_device_used_bytes=6474956800., result_bytes=41151509)


class Bookkeeping(unittest.TestCase):
    def test_frozen_draw_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"manifest.json"
            first = freeze(p, FLOOR)
            before = p.read_bytes()
            self.assertEqual(first, freeze(p, {}))
            self.assertEqual(before, p.read_bytes())
            self.assertEqual(len(first["operations"]), 240)
            self.assertEqual(len(set(w["prompt"] for w in first["works"])), 16)
            self.assertEqual({kind:sum(o["kind"]==kind for o in first["operations"]) for kind in ("generate", "write", "extract")},
                             {"generate":16, "write":64, "extract":160})

    def test_failed_operation_is_consumed(self):
        with tempfile.TemporaryDirectory() as d:
            m = build_manifest(FLOOR)
            ledger = Ledger(d,m)
            key = m["operations"][0]["id"]
            ledger.start(key); ledger.end(key,status="FAILED")
            with self.assertRaises(RuntimeError): ledger.start(key)
            with self.assertRaises(RuntimeError): Ledger(d,m,precreated=True)
            self.assertEqual(sum(r["status"]=="NOT_STARTED" for r in ledger.rows),239)

    def test_historical_cost_and_remaining_limit(self):
        summary = dict(trajectories_completed=44,trajectories_started=44,cumulative_resources=FLOOR.copy())
        self.assertEqual(inherit(summary,FLOOR),FLOOR)
        summary["cumulative_resources"]["cpu_seconds"] -= .01
        with self.assertRaises(RuntimeError): inherit(summary,FLOOR)
        carry = dict(FLOOR,gpu_occupancy_seconds=14300)
        self.assertEqual(effective_limits(build_manifest(FLOOR),carry)["gpu_seconds"],100)

    def test_missing_rows_do_not_shrink_denominator(self):
        rows = [dict(id=str(i),source="registered",correct_bits=32,psnr_db=30,ssim=.9) for i in range(31)]
        g = native_gates(rows,build_manifest(FLOOR)["gates"])["registered"]
        self.assertEqual(g["bit_denominator"],1024)
        self.assertFalse(g["passed"])
        self.assertEqual(g["bit_accuracy"],31/32)

    def test_reopening_has_no_resume(self):
        with tempfile.TemporaryDirectory() as d:
            result = read_only(Path(d))
            self.assertTrue(result["reopened_read_only"])
            self.assertEqual(result["status"],"UNFINISHED_OR_INTERRUPTED_RESOURCE_TOTAL_UNRESOLVED")
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_unknown_cache_does_not_download_over_it(self):
        from .setup import fetch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); f=root/"unknown.pth"; f.write_bytes(b"old")
            with patch("urllib.request.urlopen",side_effect=AssertionError("network forbidden")):
                with self.assertRaises(RuntimeError): fetch("https://example.invalid/model",f,root)
            self.assertEqual(f.read_bytes(),b"old")

    def test_gpu_clock_stops_cpu_setup_before_spawning(self):
        from .entry import Supervisor
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/"cache";cache.mkdir()
            monitor=Supervisor(root,build_manifest(FLOOR),FLOOR,0,0,cache)
            stats=dict(gpu_occupancy_seconds=881,wall_seconds=881,cpu_seconds=1,
                peak_ram_bytes=0,peak_device_used_bytes=0,result_bytes=0)
            with patch.object(monitor,"usage",return_value=stats),patch("subprocess.Popen") as spawn:
                with self.assertRaises(RuntimeError): monitor.command("not_allowed",["must-not-run"])
                spawn.assert_not_called()

    def test_device_and_cpu_resource_margin(self):
        from .entry import Supervisor
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            monitor=Supervisor(root,build_manifest(FLOOR),FLOOR,0,0,root)
            stats=dict(gpu_occupancy_seconds=1,wall_seconds=1,cpu_seconds=581,
                peak_ram_bytes=0,peak_device_used_bytes=0,result_bytes=0)
            self.assertIn("cpu_seconds",monitor.reason(stats))
            stats["cpu_seconds"]=0;stats["peak_device_used_bytes"]=24*1024**3
            self.assertIn("peak_device_used_bytes",monitor.reason(stats))


class FrozenMedia(unittest.TestCase):
    def test_real_png_jpeg_mix_and_fixed_counts(self):
        import numpy as np
        from .engine import run
        class Fake:
            def __init__(self): self.calls=dict(generate=0,write=0,extract=0)
            def synchronize(self): pass
            def generate(self,work):
                self.calls["generate"]+=1
                return np.full((512,512,3),128,dtype=np.uint8)
            def write(self,rgb,bits):
                self.calls["write"]+=1
                return rgb.copy()
            def extract(self,rgb):
                self.calls["extract"]+=1
                return np.ones((33,256,256),dtype=np.float32), np.ones((33,4,4),dtype=np.float32), dict(status="VALID",mask_pixels=65536,bits=[1]*32)
        with tempfile.TemporaryDirectory() as d:
            fake=Fake()
            # Exercise real media/quality/readouts, omit 1.3 GiB redundant synthetic arrays.
            with patch("wam_qualification.engine.save_array") as saver:
                result=run(d,build_manifest(FLOOR),fake)
            self.assertEqual(fake.calls,dict(generate=16,write=64,extract=160))
            self.assertEqual(saver.call_count,320)
            self.assertEqual(len((Path(d)/"readouts.jsonl").read_text().splitlines()),160)
            self.assertEqual(len(json.loads((Path(d)/"paired_diagnostics.json").read_text())),24)
            self.assertFalse(result["qualification_pass"])
            self.assertIsNone(result["scientific_pass"])

    def test_worker_failure_retains_started_and_not_started(self):
        from .engine import run
        class Broken:
            def generate(self,work): raise ValueError("synthetic generation error")
            def synchronize(self): pass
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): run(d,build_manifest(FLOOR),Broken())
            rows=json.loads((Path(d)/"operation_status.json").read_text())
            self.assertEqual(rows[0]["status"],"FAILED")
            self.assertEqual(sum(r["status"]=="NOT_STARTED" for r in rows),239)

    def test_blind_scores_depend_only_on_common_evidence_and_candidate_words(self):
        import numpy as np
        from .engine import blind_scores
        p=np.zeros((33,4,4),dtype=np.float32);p[1:]=2
        book=[[0]*128,[1]*128]
        self.assertEqual(blind_scores(p,book,0),[-2,2])
        p[0]=999
        self.assertEqual(blind_scores(p,book,0),[-2,2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
