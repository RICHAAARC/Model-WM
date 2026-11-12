"""Real CPU cancellation plus synthetic setup/resume/budget regressions."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from .core import save,Ledger
from .entry import PACKAGE,Supervisor as Legacy
from .phases import Supervisor,own_cpu
from .test_recovery2 import latest_fixture,M
from .history import close_original

B=json.loads((PACKAGE/"execution_budget.json").read_text())
LAST=dict(wall_seconds=271.46111035346985,gpu_occupancy_seconds=271.46111035346985,
    cpu_seconds=18.619529,peak_ram_bytes=260591616,peak_device_used_bytes=0,
    download_charge_bytes=0,result_bytes=758514)


def closed_fixture(parent):
    original=latest_fixture(parent)
    previous=json.loads((original/"technical-recovery-01/summary.json").read_text())
    root=original/"technical-recovery-02";root.mkdir()
    save(root/"manifest.json",M)
    save(root/"operation_status.json",[dict(op,status="NOT_STARTED") for op in M["operations"]])
    inherited=previous["cumulative_resources"]
    cumulative={k:max(inherited[k],v) if k.startswith("peak_") else inherited[k]+v for k,v in LAST.items()}
    save(root/"summary.json",dict(status="STOPPED_RESOURCE_MARGIN",resources_final=True,resources=LAST,
        inherited_resources=inherited,cumulative_resources=cumulative,operations=previous["operations"],
        inherited_trajectories_started=44,inherited_trajectories_completed=44))
    sample=dict(LAST,cpu_seconds=78.361534)
    save(root/"resources.json",dict(stage="environment",resources=sample,resources_final=False))
    return original


class Accounting(unittest.TestCase):
    def test_supervisor_cancellation_retains_sealed_cpu_in_final_total(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inv=root/"inv";inv.mkdir();cache=root/"cache";cache.mkdir();Ledger(root,M)
            budget=copy.deepcopy(B);budget["preparation"]["wall_seconds"]=20.8
            child="import time; end=time.monotonic()+30\nwhile time.monotonic()<end: pass"
            parent=f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
            s=Supervisor(root,inv,M,M["inherited_resources"],dict(resources={},phases={}),budget,time.time(),own_cpu(),cache,0)
            with patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                with self.assertRaisesRegex(RuntimeError,"Stopping margin"):
                    s.command("synthetic",[sys.executable,"-c",parent])
                final=s.finish("synthetic budget interruption")
            observed=json.loads((inv/"synthetic.usage.json").read_text())
            self.assertTrue(observed["resources_final"] and observed["owner_exit_cpu_included"])
            self.assertGreaterEqual(final["resources"]["cpu_seconds"],observed["cpu_seconds"])
            self.assertGreaterEqual(final["cumulative_resources"]["cpu_seconds"],M["inherited_resources"]["cpu_seconds"]+observed["cpu_seconds"])
            self.assertTrue(final["resources_final"])
            self.assertFalse(final["environment_resume_allowed"])

    def test_real_grandchild_cancellation_seals_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);receipt=d/"usage.json"
            child="import time; end=time.monotonic()+30\nwhile time.monotonic()<end: pass"
            parent=f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
            with (d/"log").open("w") as log:
                p=subprocess.Popen([sys.executable,"-m","wam_qualification.metered",str(receipt),sys.executable,"-c",parent],stdout=log,stderr=log)
                try:
                    deadline=time.monotonic()+10
                    while time.monotonic()<deadline:
                        if receipt.exists():
                            before=json.loads(receipt.read_text())
                            if before["cpu_seconds"]>.3:break
                        time.sleep(.05)
                    else:self.fail("Meter did not observe live descendant CPU")
                    p.terminate();p.wait(timeout=10)
                    final=json.loads(receipt.read_text())
                    self.assertEqual(p.returncode,130)
                    self.assertTrue(final["resources_final"] and final["all_descendants_reaped"])
                    self.assertGreaterEqual(final["cpu_seconds"],before["cpu_seconds"])
                    self.assertGreater(final["cpu_seconds"],.3)
                finally:
                    if p.poll() is None:p.terminate();p.wait(timeout=12)

    def test_legacy_cpu_observation_cannot_regress(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);s=Legacy(root,M,M["inherited_resources"],time.time(),0,root)
            with patch("wam_qualification.entry.cpu",side_effect=[78.36,18.62]),patch("wam_qualification.entry.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                self.assertEqual(s.usage()["cpu_seconds"],78.36)
                self.assertEqual(s.usage()["cpu_seconds"],78.36)

    def test_history_correction_is_additive_without_rewriting_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=closed_fixture(Path(d));before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()}
            h=close_original(root,M,PACKAGE)
            self.assertAlmostEqual(h["carry"]["cpu_seconds"],2404.914067)
            self.assertAlmostEqual(h["corrections"]["cpu_seconds"]["added_minimum"],59.742005)
            self.assertFalse(h["historical_cpu_exact"])
            self.assertAlmostEqual(h["budget_carry"]["cpu_seconds"],2562.099122)
            self.assertEqual(before,{str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_preparation_does_not_use_model_allowance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inv=root/"inv";inv.mkdir();cache=root/"cache";cache.mkdir()
            with patch("wam_qualification.phases.time.time",return_value=600.),patch("wam_qualification.phases.own_cpu",return_value=100.),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                s=Supervisor(root,inv,M,M["inherited_resources"],dict(resources={},phases={}),B,100.,0,cache,0)
                self.assertIsNone(s.reason(s.usage()))
                self.assertEqual(s.last_phase["wall_seconds"],500)
                s.enter_model()
                self.assertEqual(s.last_phase["wall_seconds"],0)
                self.assertEqual(s.last_phase["cpu_seconds"],0)
                self.assertEqual(s.preparation_fixed["wall_seconds"],500)
                with patch("wam_qualification.phases.time.time",return_value=1481.):
                    self.assertIn("model gpu_occupancy_seconds",s.reason(s.usage()))

    def test_preparation_budget_is_cumulative_across_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inv=root/"inv";inv.mkdir();cache=root/"cache";cache.mkdir()
            prior=dict(resources={},phases=dict(preparation=dict(wall_seconds=1000,cpu_seconds=100,gpu_occupancy_seconds=1000)))
            with patch("wam_qualification.phases.time.time",return_value=161.),patch("wam_qualification.phases.own_cpu",return_value=0.),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                s=Supervisor(root,inv,M,M["inherited_resources"],prior,B,100.,0,cache,0)
                self.assertIn("preparation wall_seconds",s.reason(s.usage()))

    def test_global_cpu_gate_uses_conservative_history_debit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inv=root/"inv";inv.mkdir();cache=root/"cache";cache.mkdir()
            with patch("wam_qualification.phases.time.time",return_value=100.),patch("wam_qualification.phases.own_cpu",return_value=0.),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                s=Supervisor(root,inv,M,M["inherited_resources"],dict(resources={},phases={}),B,100.,0,cache,0)
                s.budget_carry=dict(M["inherited_resources"],cpu_seconds=M["budget"]["global_cpu_seconds"]-19)
                self.assertIn("global_cpu_seconds",s.reason(s.usage()))


class Recovery(unittest.TestCase):
    def test_old_ready_does_not_replace_missing_runtime(self):
        from .prepare import environment
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/"out";out.mkdir();cache=root/"cache";(cache/"wheels").mkdir(parents=True)
            save(out/"python.json",dict(python="/disappeared/runtime/python",reused=True))
            with patch("wam_qualification.prepare.RUNTIME_ROOT",root/"fresh-runtime"),patch("wam_qualification.prepare.fetch",return_value=root/"paid-python.tar") as fetch,patch("wam_qualification.prepare.unpack"),patch("wam_qualification.prepare.call") as call:
                environment(out,root/"scratch",root/"aqua",cache)
            self.assertEqual(fetch.call_count,1)
            commands=[[str(x) for x in c.args[0]] for c in call.call_args_list]
            self.assertTrue(any("venv" in c for c in commands))
            self.assertFalse(json.loads((out/"python.json").read_text())["reused"])

    def test_unsealed_runtime_loss_blocks_automatic_resume(self):
        from . import separated_entry as e
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);closed_fixture(parent)
            root=parent/B["execution_id"];root.mkdir()
            save(root/"manifest.json",M);save(root/"execution_budget.json",B)
            save(root/"summary.json",dict(status="PREPARING",resources_final=False))
            with patch.object(e,"PARENT",parent),patch.object(Supervisor,"command",side_effect=AssertionError("No resume with unknown costs")):
                result=e.run_colab()
            self.assertTrue(result["reopened_read_only"])
            self.assertEqual(result["status"],"UNFINISHED_OR_INTERRUPTED_RESOURCE_TOTAL_UNRESOLVED")

    def test_partial_environment_installs_missing_pins_then_rechecks(self):
        from .prepare import environment
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/"out";out.mkdir();runtime=root/"runtime"
            py=runtime/"wam-pinned-v1/venv/bin/python";py.parent.mkdir(parents=True);py.write_text("fixture")
            cache=root/"cache";(cache/"wheels").mkdir(parents=True)
            with patch("wam_qualification.prepare.RUNTIME_ROOT",runtime),patch("wam_qualification.prepare.subprocess.run",return_value=SimpleNamespace(returncode=1)),patch("wam_qualification.prepare.call") as call:
                environment(out,root/"scratch",root/"aqua",cache)
            commands=[[str(x) for x in c.args[0]] for c in call.call_args_list]
            self.assertTrue(any("install" in c and "--no-index" in c for c in commands))
            self.assertFalse(any("--force-reinstall" in c or "download" in c or "venv" in c for c in commands))
            self.assertTrue(json.loads((out/"python.json").read_text())["validated_in_current_runtime"])

    def test_current_compatible_environment_skips_installation(self):
        from .prepare import environment
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime=root/"runtime";py=runtime/"wam-pinned-v1/venv/bin/python";py.parent.mkdir(parents=True);py.write_text("fixture")
            with patch("wam_qualification.prepare.RUNTIME_ROOT",runtime),patch("wam_qualification.prepare.subprocess.run",return_value=SimpleNamespace(returncode=0)) as probe,patch("wam_qualification.prepare.call") as call:
                environment(root,root/"scratch",root/"aqua",root/"cache")
            self.assertEqual(probe.call_count,1);call.assert_not_called()
            self.assertTrue(json.loads((root/"python.json").read_text())["reused"])

    def test_existing_runtime_cache_is_reused_without_drive_copy(self):
        from .storage import restore,wheels
        from .transport import ORIGINAL
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);local=root/"old-content";local.mkdir()
            (local/"model").write_bytes(b"fixture");save(local/"model.complete.json",dict(url=ORIGINAL))
            (local/"wheels").mkdir()
            save(root/"local_storage.json",dict(mappings=[dict(new=str(root/"new"),paid=str(root/"paid"),drive=str(root/"drive"),local=[str(local)])]))
            with patch("shutil.copyfile",side_effect=AssertionError("No copy")),patch("shutil.copytree",side_effect=AssertionError("No copy")):
                self.assertEqual(restore(root,root/"new/model",ORIGINAL),local/"model")
                self.assertEqual(wheels(root,root/"new/wheels"),local/"wheels")

    def test_environment_resume_then_exactly_one_model_phase(self):
        from . import separated_entry as e
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);old=closed_fixture(parent);before={str(p.relative_to(old)):p.read_bytes() for p in old.rglob("*") if p.is_file()}
            count=[0]
            def command(s,stage,args):
                if stage=="environment":
                    count[0]+=1
                    if count[0]==1:raise RuntimeError("synthetic interrupted installation")
                    save(s.invocation/"python.json",dict(python=sys.executable))
                if stage=="official_assets":save(s.invocation/"model_paths.json",dict(source="fixture",checkpoint="fixture",base="fixture"))
                if stage=="qualification":save(s.root/"numeric_result.json",dict(status="SYNTHETIC_ONLY",qualification_pass=False))
            with patch.object(e,"PARENT",parent),patch.object(e,"RUNTIME_ROOT",parent/"content"),patch.object(Supervisor,"command",command),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                first=e.run_colab();self.assertTrue(first["environment_resume_allowed"])
                second=e.run_colab();self.assertFalse(second["environment_resume_allowed"])
                third=e.run_colab();self.assertTrue(third["reopened_read_only"])
            self.assertEqual(count[0],2)
            self.assertEqual(len(list((parent/B["execution_id"]/"invocations").iterdir())),2)
            self.assertGreaterEqual(second["phases"]["preparation"]["wall_seconds"],first["phases"]["preparation"]["wall_seconds"])
            self.assertGreaterEqual(second["cumulative_resources"]["cpu_seconds"],first["cumulative_resources"]["cpu_seconds"])
            self.assertEqual(before,{str(p.relative_to(old)):p.read_bytes() for p in old.rglob("*") if p.is_file()})


if __name__=="__main__":unittest.main(verbosity=2)
