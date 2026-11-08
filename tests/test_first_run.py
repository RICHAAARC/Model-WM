"""Infrastructure regression checks; no model weights or trajectories."""
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qualification.durable_download import fetch
from qualification.first_run import read_started, history_resources, FirstRunSupervisor
from qualification.protocol import build_manifest
from qualification.records import Ledger, write_json


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}

    def getcode(self):
        return self.status


class DurableCacheTests(unittest.TestCase):
    def test_partial_transfer_resumes_and_completed_file_survives_new_caller(self):
        with tempfile.TemporaryDirectory() as d:
            p, log = Path(d)/"drive/model", Path(d)/"events.jsonl"
            p.parent.mkdir()
            p.with_name("model.part").write_bytes(b"first-")
            calls = []
            def request(req, **kw):
                calls.append(req.get_header("Range"))
                return Response(b"second", 206, {"Content-Length": "6", "Content-Range": "bytes 6-11/12"})
            fetch("https://example.invalid/frozen", p, log, opener=request)
            self.assertEqual(p.read_bytes(), b"first-second")
            self.assertEqual(calls, ["bytes=6-"])
            # New Python caller/new VM sees the durable cache and needs no network.
            fetch("https://example.invalid/frozen", p, log,
                  opener=lambda *a, **k: self.fail("completed durable cache must be reused"))

    def test_ignored_range_does_not_concatenate_a_second_full_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"model"
            p.with_name("model.part").write_bytes(b"old")
            fetch("https://example.invalid/frozen", p, Path(d)/"events", opener=lambda *a, **k: Response(b"complete"))
            self.assertEqual(p.read_bytes(), b"complete")

    def test_transient_download_failure_retries_without_any_trajectory(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def request(*a, **k):
                calls.append(1)
                if len(calls) == 1:
                    raise TimeoutError("test transport timeout")
                return Response(b"complete")
            with patch("qualification.durable_download.time.sleep"):
                fetch("https://example.invalid/frozen", Path(d)/"file", Path(d)/"events", opener=request)
            rows = [json.loads(x) for x in (Path(d)/"events").read_text().splitlines()]
            self.assertEqual([r["status"] for r in rows], ["TRANSPORT_FAILED", "COMPLETED"])
            self.assertFalse(list(Path(d).rglob("trajectory_status.json")))


class FirstRunTests(unittest.TestCase):
    def test_unstarted_environment_failure_does_not_consume_a_generation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"old_failed_load"
            root.mkdir()
            Ledger(root, build_manifest())
            self.assertIsNone(read_started(d, build_manifest()))

    def test_started_slot_is_retained_and_blocks_regeneration(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"old"
            root.mkdir()
            ledger = Ledger(root, build_manifest())
            ledger.start(0)
            ledger.end(0, "FAILED", error="synthetic control")
            saved, rows = read_started(d, build_manifest())
            self.assertEqual(saved, root)
            self.assertEqual(rows[0]["status"], "FAILED")

    def test_cumulative_history_not_summed_twice_or_cleared(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for name, count in [("initial", 10), ("recovery", 30), ("model-stage", 31)]:
                (p/name).mkdir()
                write_json(p/name/"summary.json", dict(resources=dict(download_charge_bytes=count)))
            self.assertEqual(history_resources(p, p/"first-run")["download_charge_bytes"], 31)

    def test_download_cap_no_longer_blocks_and_model_time_still_stops(self):
        s = object.__new__(FirstRunSupervisor)
        s.manifest = build_manifest()
        row = dict(wall_seconds=100, model_phase_seconds=0, cpu_seconds=100,
                   peak_ram_bytes=1, peak_device_used_bytes=1, result_bytes=1,
                   download_charge_bytes=11*1024**3)
        self.assertIsNone(s.reason(row))
        self.assertIn("model_phase_seconds", s.reason(dict(row, model_phase_seconds=1780)))

    def test_existing_scratch_is_reusable_after_setup_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/"root").mkdir()
            (p/"cache").mkdir()
            for _ in range(2):
                FirstRunSupervisor(p/"root", p/"runtime", build_manifest(), 0,
                                   cache=p/"cache", monitor_gpu=False)


if __name__ == "__main__":
    unittest.main()
