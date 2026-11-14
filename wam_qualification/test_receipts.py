"""CPU-only local IPC, sealed publication and old-schema recovery regressions."""
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import separated_entry as entry
from .core import Ledger, save
from .entry import PACKAGE, size
from .phases import Supervisor, combine, own_cpu
from .receipts import read_receipt
from .test_phases import B, M, closed_fixture
from .history import close_original


def previous_failures(parent):
    original=closed_fixture(parent)
    history=close_original(original,M,PACKAGE)
    root=parent/B["execution_id"];root.mkdir()
    save(root/"manifest.json",M);save(root/"execution_budget.json",B);Ledger(root,M)
    resources={}
    for number,wall,cpu,error in [(1,30.793078422546387,.8634970000000003,"FileNotFoundError"),
                                  (2,38.398735761642456,1.4544189999999997,"JSONDecodeError")]:
        inv=root/"invocations"/f"{number:04d}";inv.mkdir(parents=True)
        stats=dict(wall_seconds=wall,gpu_occupancy_seconds=wall,cpu_seconds=cpu,
            peak_ram_bytes=153587712,peak_device_used_bytes=0,download_charge_bytes=0,result_bytes=746396)
        resources=combine(resources,stats)
        save(inv/"summary.json",dict(resources=stats,resources_final=True,error=error))
        save(inv/"execution_failure.json",dict(error=error))
        save(inv/"asset_preflight.usage.json",dict(cpu_seconds=.517439,resources_final=True,
            all_descendants_reaped=True,interrupted=True,exit_code=-15))
    phase={k:resources[k] for k in ("wall_seconds","gpu_occupancy_seconds","cpu_seconds")}
    save(root/"summary.json",dict(resources=resources,phases=dict(preparation=phase),
        inherited_resources=history["carry"],resources_final=True,environment_resume_allowed=True,
        status="ENVIRONMENT_STOPPED"))
    return root


class LocalReceipts(unittest.TestCase):
    def test_atomic_local_replacements_under_concurrent_reads(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"usage.json";save(path,dict(cpu_seconds=0,body="x"*16384))
            errors=[];samples=[];done=threading.Event()
            def writer():
                try:
                    for i in range(1,401):save(path,dict(cpu_seconds=i,body="x"*16384))
                except BaseException as exc:errors.append(exc)
                finally:done.set()
            t=threading.Thread(target=writer);t.start()
            try:
                while not done.is_set():
                    receipt=read_receipt(path);samples.append(receipt["cpu_seconds"])
                    self.assertEqual(len(receipt["body"]),16384)
            finally:t.join(timeout=10)
            self.assertFalse(errors);self.assertGreater(len(samples),10)
            self.assertEqual(samples,sorted(samples));self.assertEqual(read_receipt(path)["cpu_seconds"],400)

    def test_corruption_is_not_silently_zeroed_and_has_path_and_cause(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"usage.json";path.write_text("")
            with self.assertRaisesRegex(RuntimeError,"usage.json") as caught:read_receipt(path)
            self.assertIsInstance(caught.exception.__cause__,json.JSONDecodeError)

    def test_polling_never_reads_or_writes_drive_and_worker_reads_local(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);drive=parent/"drive";drive.mkdir()
            work=parent/"content";work.mkdir();inv=work/"invocations/0003";inv.mkdir(parents=True)
            cache=parent/"cache";cache.mkdir();Ledger(work,M)
            s=Supervisor(work,inv,M,M["inherited_resources"],dict(resources={},phases={}),B,time.time(),own_cpu(),cache,0,archive_root=drive)
            original_read=Path.read_text
            original_save=save
            def read(path,*args,**kwargs):
                self.assertFalse(path.is_relative_to(drive),"Drive used as live IPC")
                return original_read(path,*args,**kwargs)
            def write(path,value):
                self.assertFalse(Path(path).is_relative_to(drive),"Live write reached Drive")
                return original_save(path,value)
            with patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                # A real child continuously reads the exact worker mailbox while
                # the parent rewrites it; the metered owner writes a second mailbox.
                code="import json,time,sys; from pathlib import Path; p=Path(sys.argv[1]); end=time.monotonic()+.6\nwhile time.monotonic()<end:\n r=json.loads(p.read_text()); assert r['resources']['cpu_seconds']>=0\n"
                with patch.object(s,"publish_stage"),patch.object(Path,"read_text",read),patch("wam_qualification.phases.save",write):
                    s.command("local_reader",[sys.executable,"-c",code,work/"resources.json"])
                self.assertEqual(list(drive.iterdir()),[])
                s.publish_stage("local_reader")
                sealed=read_receipt(drive/"invocations/0003/local_reader.usage.json")
                self.assertTrue(sealed["resources_final"] and sealed["all_descendants_reaped"])
                self.assertFalse((drive/"resources.json").exists())
                final=s.finish("synthetic stop")
            self.assertEqual(final["resources"]["result_bytes"],size(drive))
            self.assertEqual(final["resources"]["cpu_seconds"],json.loads((drive/"resources.json").read_text())["resources"]["cpu_seconds"])

    def test_two_failures_resume_without_budget_reset_and_exact_final_volume(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=previous_failures(parent)
            old={str(p.relative_to(root)):p.read_bytes() for p in (root/"invocations").rglob("*") if p.is_file()}
            old_root=(root/"summary.json").read_bytes()
            old_ledger=(root/"operation_status.json").read_bytes()
            before=json.loads((root/"summary.json").read_text())
            def command(s,stage,args):
                self.assertTrue(s.root.is_relative_to(parent/"content"))
                self.assertEqual(s.budget,B)
                self.assertAlmostEqual(s.prior["resources"]["wall_seconds"],69.19181418418884)
                self.assertAlmostEqual(s.prior["phases"]["preparation"]["cpu_seconds"],2.317916)
                if stage=="asset_preflight":
                    return Supervisor.command(s,"asset_preflight_fixture",[sys.executable,"-c","raise ValueError('fixture failure before models')"])
            real_command=Supervisor.command
            def dispatch(s,stage,args):
                if stage=="asset_preflight_fixture":return real_command(s,stage,args)
                return command(s,stage,args)
            with patch.object(entry,"PARENT",parent),patch.object(entry,"RUNTIME_ROOT",parent/"content"),patch.object(Supervisor,"command",dispatch),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                final=entry.run_colab()
            inv=root/"invocations/0003"
            self.assertEqual(len(list((root/"invocations").iterdir())),3)
            self.assertTrue(final["resources_final"] and final["environment_resume_allowed"])
            self.assertGreater(final["phases"]["preparation"]["wall_seconds"],69.19181418418884)
            self.assertGreater(final["resources"]["cpu_seconds"],2.317916)
            self.assertNotIn("model",final["phases"])
            self.assertEqual(final["execution_budget"],B)
            self.assertFalse((root/"model_phase_started.json").exists())
            self.assertEqual(final["resources"]["result_bytes"],size(root))
            self.assertEqual(final["cumulative_resources"]["result_bytes"],final["inherited_resources"]["result_bytes"]+size(root))
            self.assertEqual(json.loads((inv/"summary.json").read_text())["resources"]["result_bytes"],size(inv))
            self.assertEqual(before,json.loads((inv/"prior_summary.json").read_text()))
            self.assertEqual(old_root,(inv/"prior_summary.json").read_bytes())
            self.assertEqual(old_ledger,(inv/"prior_operation_status.json").read_bytes())
            self.assertIn("Traceback (most recent call last)",json.loads((inv/"execution_failure.json").read_text())["traceback"])
            self.assertIn("ValueError: fixture failure",(inv/"asset_preflight_fixture.log").read_text())
            for p,content in old.items():self.assertEqual((root/p).read_bytes(),content)

    def test_regressed_costs_stop_before_new_invocation(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=previous_failures(parent)
            summary=json.loads((root/"summary.json").read_text());summary["resources"]["cpu_seconds"]=0;save(root/"summary.json",summary)
            with patch.object(entry,"PARENT",parent),self.assertRaisesRegex(RuntimeError,"Regressed prior resource ledger"):
                entry.run_colab()
            self.assertEqual(len(list((root/"invocations").iterdir())),2)

    def test_phase_debit_cannot_reset_behind_correct_total(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=previous_failures(parent)
            summary=json.loads((root/"summary.json").read_text());summary["phases"]["preparation"]["wall_seconds"]=0;save(root/"summary.json",summary)
            with patch.object(entry,"PARENT",parent),self.assertRaisesRegex(RuntimeError,"Regressed preparation budget debit"):
                entry.run_colab()
            self.assertEqual(len(list((root/"invocations").iterdir())),2)

    def test_concurrent_entry_is_read_only_before_reading_dynamic_files(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=previous_failures(parent)
            (root/"active.lock").write_text("reserved")
            (root/"summary.json").write_text("")
            with patch.object(entry,"PARENT",parent),patch.object(Supervisor,"command",side_effect=AssertionError("No concurrent work")):
                result=entry.run_colab()
            self.assertEqual(result["status"],"ACTIVE_OR_UNSEALED_INVOCATION")
            self.assertEqual(len(list((root/"invocations").iterdir())),2)

    def test_finalization_failure_has_stack_and_cannot_resume(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=previous_failures(parent)
            with patch.object(entry,"PARENT",parent),patch.object(entry,"RUNTIME_ROOT",parent/"content"),patch.object(Supervisor,"command",side_effect=RuntimeError("setup fixture")),patch.object(Supervisor,"finish",side_effect=OSError("publication fixture")):
                final=entry.run_colab();again=entry.run_colab()
            self.assertFalse(final["resources_final"])
            self.assertIn("OSError: publication fixture",final["traceback"])
            self.assertAlmostEqual(final["prior_resources"]["wall_seconds"],69.19181418418884)
            self.assertEqual(again["status"],"ACTIVE_OR_UNSEALED_INVOCATION")
            self.assertEqual(len(list((root/"invocations").iterdir())),3)

    def test_durable_model_reservation_blocks_after_local_runtime_loss(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=previous_failures(parent)
            def command(s,stage,args):
                if stage=="environment":save(s.invocation/"python.json",dict(python=sys.executable))
                if stage=="official_assets":save(s.invocation/"model_paths.json",{})
                if stage=="qualification":
                    self.assertTrue((root/"model_phase_started.json").exists())
                    raise RuntimeError("fixture model load failure, no weights")
            with patch.object(entry,"PARENT",parent),patch.object(entry,"RUNTIME_ROOT",parent/"content"),patch.object(Supervisor,"command",command),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                first=entry.run_colab()
                shutil.rmtree(parent/"content")  # Disposable fixture only.
                second=entry.run_colab()
            self.assertFalse(first["environment_resume_allowed"])
            self.assertTrue(second["reopened_read_only"])
            self.assertEqual(len(list((root/"invocations").iterdir())),3)


if __name__=="__main__":unittest.main(verbosity=2)
