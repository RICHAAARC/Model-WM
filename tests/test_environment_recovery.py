"""CPU-only recovery controls. Synthetic history and HTTP, no models or GPU."""
import io
import json
import tarfile
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import colab_bootstrap as boot
from qualification import python_runtime
from qualification.protocol import build_manifest
from qualification.records import Ledger, write_json, total_bytes


class RecoveryTests(unittest.TestCase):
    def fixture(self, folder, ledger=True):
        parent = Path(folder)/"drive"
        root = parent/build_manifest()["run_id"]
        root.mkdir(parents=True)
        write_json(root/"manifest.json", build_manifest())
        write_json(root/"summary.json", dict(status="STOPPED_INCOMPLETE", output=str(root),
            error="RuntimeError: python_environment exited 2; see python_environment.log",
            trajectories_started=0, trajectories_completed=0, trajectory_denominator=44,
            resources=boot.MINIMUM_CARRY.copy()))
        write_json(root/"setup_stages.json", [dict(stage="install_uv", returncode=0),
                                            dict(stage="python_environment", returncode=2)])
        for name in ("python_environment.log", "install_uv.log", "runtime_entry.json", "execution_failure.json", "resources.json", "executed_source.tar.gz"):
            (root/name).write_bytes(b"synthetic original evidence\n")
        if ledger:
            Ledger(root, build_manifest())
        return parent, root

    def call(self, folder, parent, **kwargs):
        with patch.object(boot, "DRIVE_PARENT", parent), patch.object(boot, "WORK_PARENT", Path(folder)/"scratch"):
            return boot.recover_environment(**kwargs)

    def test_success_preserves_all_files_counts_and_is_read_only_on_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, root = self.fixture(folder)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            # Only the actual dependency operation is mocked; entry, accounting,
            # reservation, original-file inventory and reporting run for real.
            with patch.object(boot.Supervisor, "environment_only") as setup, \
                 patch.object(boot.Supervisor, "execute", side_effect=AssertionError("no model path")):
                result = self.call(folder, parent)
                self.assertEqual(result["status"], "ENVIRONMENT_READY")
                self.assertEqual(self.call(folder, parent), result)
                setup.assert_called_once()
            self.assertEqual(result["trajectories_started"], 0)
            self.assertEqual(result["trajectory_denominator"], 44)
            self.assertFalse(result["qualification_pass"])
            self.assertTrue(result["resources_final"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()})
            self.assertEqual(json.loads((root/boot.RECOVERY_NAME/"preserved_files.json").read_text()).keys(), before.keys())

    def test_failure_and_interrupt_consume_only_recovery_and_keep_original_failure(self):
        for failure in (RuntimeError("injected HTTP failure"), KeyboardInterrupt("injected interruption")):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as folder:
                parent, root = self.fixture(folder, ledger=False)
                original = (root/"summary.json").read_bytes()
                with patch.object(boot.Supervisor, "environment_only", side_effect=failure) as setup:
                    result = self.call(folder, parent)
                    self.assertEqual(result["status"], "STOPPED_INCOMPLETE")
                    self.assertEqual(self.call(folder, parent), result)
                    setup.assert_called_once()
                self.assertIn(type(failure).__name__, result["error"])
                self.assertEqual((root/"summary.json").read_bytes(), original)
                self.assertFalse((root/"trajectory_status.json").exists())
                self.assertTrue((root/boot.RECOVERY_NAME/"execution_failure.json").exists())

    def test_resource_costs_accumulate_and_peaks_use_max_without_gpu_query(self):
        with tempfile.TemporaryDirectory() as folder:
            root, scratch = Path(folder)/"result", Path(folder)/"scratch"
            root.mkdir()
            carry = dict(boot.MINIMUM_CARRY, peak_ram_bytes=2**40, peak_device_used_bytes=12345)
            s = boot.Supervisor(root, scratch, build_manifest(), 100, carry=carry,
                                account_root=root, monitor_gpu=False)
            s.cpu_base = 10
            (scratch/"downloads/new-payload").write_bytes(b"a"*321)
            with patch.object(boot.time, "time", return_value=107), patch.object(boot, "cpu_usage", return_value=13), \
                 patch.object(boot.subprocess, "run", side_effect=AssertionError("no GPU/device query")):
                stats = s.usage()
            self.assertEqual(stats["wall_seconds"], carry["wall_seconds"]+7)
            self.assertEqual(stats["gpu_occupancy_seconds"], carry["gpu_occupancy_seconds"]+7)
            self.assertEqual(stats["cpu_seconds"], carry["cpu_seconds"]+3)
            self.assertEqual(stats["peak_ram_bytes"], carry["peak_ram_bytes"])
            self.assertEqual(stats["peak_device_used_bytes"], carry["peak_device_used_bytes"])
            self.assertEqual(stats["download_charge_bytes"], carry["download_charge_bytes"]+321)
            self.assertEqual(stats["result_bytes"], carry["result_bytes"])
            (root/"large-receipt").write_bytes(b"a"*140000)
            self.assertEqual(s.usage()["result_bytes"], total_bytes(root))

    def test_budget_exhausted_before_subprocess_cannot_launch_setup(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, root = self.fixture(folder)
            summary = json.loads((root/"summary.json").read_text())
            summary["resources"]["gpu_occupancy_seconds"] = 1780
            write_json(root/"summary.json", summary)
            with patch.object(boot.subprocess, "Popen", side_effect=AssertionError("budget must stop before launch")), \
                 patch.object(boot.subprocess, "check_output", return_value="synthetic commit"):
                result = self.call(folder, parent)
            self.assertEqual(result["status"], "STOPPED_INCOMPLETE")
            self.assertIn("gpu_occupancy_seconds", result["error"])
            self.assertTrue((root/boot.RECOVERY_NAME/"STOP_REQUESTED.json").exists())

    def test_unknown_or_regressed_costs_stop_before_recovery_reservation(self):
        for key in boot.MINIMUM_CARRY:
            for value in (None, float("nan"), -1):
                with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as folder:
                    parent, root = self.fixture(folder)
                    summary = json.loads((root/"summary.json").read_text())
                    summary["resources"][key] = value
                    (root/"summary.json").write_text(json.dumps(summary))
                    with self.assertRaisesRegex(RuntimeError, "resource receipt"):
                        self.call(folder, parent)
                    self.assertFalse((root/boot.RECOVERY_NAME).exists())

    def test_changed_manifest_or_consumed_trajectory_cannot_be_reset(self):
        for condition in ("manifest", "STARTED", "FAILED", "COMPLETED", "missing_log"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as folder:
                parent, root = self.fixture(folder)
                if condition == "manifest":
                    m = build_manifest()
                    m["budget"]["trajectories"] = 45
                    write_json(root/"manifest.json", m)
                elif condition == "missing_log":
                    (root/"python_environment.log").unlink()
                else:
                    path = root/"trajectory_status.json"
                    rows = json.loads(path.read_text())
                    rows[0]["status"] = condition
                    write_json(path, rows)
                before = {p.name: p.read_bytes() for p in root.iterdir()}
                with self.assertRaises(RuntimeError):
                    self.call(folder, parent)
                self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_reserved_but_unfinalized_recovery_stays_consumed(self):
        with tempfile.TemporaryDirectory() as folder:
            parent, root = self.fixture(folder)
            (root/boot.RECOVERY_NAME).mkdir()
            with patch.object(boot.Supervisor, "environment_only", side_effect=AssertionError("no retry")):
                result = self.call(folder, parent)
            self.assertEqual(result["status"], "EXISTING_ATTEMPT_READ_ONLY")

    def test_environment_plan_has_no_model_commands_and_explicit_python_no_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"result"
            root.mkdir()
            s = boot.Supervisor(root, Path(folder)/"scratch", build_manifest(), time.time(),
                                carry=boot.MINIMUM_CARRY, monitor_gpu=False)
            with patch.object(s, "command") as cmd:
                s.environment_only()
            calls = {c.args[0]: c.args[1] for c in cmd.call_args_list}
            self.assertEqual(list(calls), ["download_uv", "install_uv", "download_python", "python_environment",
                                          "download_dependencies", "install_dependencies", "dependency_check", "environment_versions"])
            self.assertIn("--no-python-downloads", calls["python_environment"])
            self.assertTrue(calls["python_environment"][3].endswith("python/python/bin/python3.10"))
            self.assertEqual(s.environment["UV_HTTP_RETRIES"], "0")
            self.assertEqual(s.environment["PIP_RETRIES"], "0")
            self.assertEqual(s.environment["CUDA_VISIBLE_DEVICES"], "")
            self.assertIn("downloads/uv-cache", s.environment["UV_CACHE_DIR"])


class PythonDownloadTests(unittest.TestCase):
    def response(self, data, length=None):
        response = io.BytesIO(data)
        response.headers = {"Content-Length": str(len(data) if length is None else length)}
        return response

    def archive(self, name="python/bin/python3.10", link=None):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
            item = tarfile.TarInfo(name)
            if link:
                item.type, item.linkname = tarfile.SYMTYPE, link
                bundle.addfile(item)
            else:
                item.size = 4
                bundle.addfile(item, io.BytesIO(b"test"))
        return stream.getvalue()

    def test_official_api_once_and_python_version_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            scratch, root = Path(folder)/"scratch", Path(folder)
            with patch.object(python_runtime.urllib.request, "urlopen", return_value=self.response(self.archive())) as fetch, \
                 patch.object(python_runtime.subprocess, "check_output", return_value="Python 3.10.14\n"):
                result = python_runtime.acquire(scratch, root, 1024**2)
            fetch.assert_called_once()
            request = fetch.call_args.args[0]
            self.assertTrue(request.full_url.startswith(python_runtime.ASSET_API+"?"))
            self.assertEqual(request.get_header("Accept"), "application/octet-stream")
            self.assertTrue(result.exists())
            records = [json.loads(x) for x in (root/"python_download.jsonl").read_text().splitlines()]
            self.assertEqual([r["status"] for r in records], ["STARTED", "COMPLETED"])

    def test_http_failure_no_retry_and_failure_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(python_runtime.urllib.request, "urlopen", side_effect=urllib.error.HTTPError(
                python_runtime.ASSET_API, 504, "injected", {}, None)) as fetch:
                with self.assertRaises(urllib.error.HTTPError):
                    python_runtime.acquire(Path(folder)/"scratch", folder, 1024**2)
            fetch.assert_called_once()
            self.assertIn('"status": "FAILED"', (Path(folder)/"python_download.jsonl").read_text())

    def test_partial_and_over_budget_downloads_are_retained_without_extraction(self):
        for length, allowance in ((100, 1000), (4, 2)):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as folder:
                scratch = Path(folder)/"scratch"
                with patch.object(python_runtime.urllib.request, "urlopen", return_value=self.response(b"test", length)):
                    with self.assertRaises(RuntimeError):
                        python_runtime.acquire(scratch, folder, allowance)
                self.assertTrue((scratch/"downloads"/python_runtime.ASSET_NAME).exists())
                self.assertFalse((scratch/"python").exists())
                with self.assertRaisesRegex(RuntimeError, "no retry"):
                    python_runtime.acquire(scratch, folder, allowance)

    def test_archive_traversal_and_external_symlink_rejected(self):
        for name, link in (("../escape", None), ("python/link", "../../escape")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                with patch.object(python_runtime.urllib.request, "urlopen", return_value=self.response(self.archive(name, link))):
                    with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                        python_runtime.acquire(Path(folder)/"scratch", folder, 1024**2)
                self.assertFalse((Path(folder)/"escape").exists())

    def test_wrong_python_version_stops_with_retained_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(python_runtime.urllib.request, "urlopen", return_value=self.response(self.archive())), \
                 patch.object(python_runtime.subprocess, "check_output", return_value="Python 3.10.15\n"):
                with self.assertRaisesRegex(RuntimeError, "Unexpected interpreter"):
                    python_runtime.acquire(Path(folder)/"scratch", folder, 1024**2)
            self.assertIn('"status": "FAILED"', (Path(folder)/"python_download.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
