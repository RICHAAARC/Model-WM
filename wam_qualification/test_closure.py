"""Closure admission, retained history, small metadata first, and cost gates."""
import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import separated_entry as entry, closure, setup, transport
from .core import save, Ledger
from .phases import Supervisor, own_cpu, combine
from .test_continuation import three_failures
from .test_phases import M, B
from .entry import PACKAGE, size

A=json.loads((PACKAGE/"preparation_amendment.json").read_text())


def four_failures(parent):
    root=three_failures(parent)
    s=json.loads((root/"summary.json").read_text())
    inv=root/"invocations/0004";inv.mkdir()
    stats=dict(wall_seconds=458.1681249141693,gpu_occupancy_seconds=458.1681249141693,
        cpu_seconds=171.779985,peak_ram_bytes=748982272,peak_device_used_bytes=3145728,
        download_charge_bytes=0,result_bytes=843890)
    s["resources"]=combine(s["resources"],stats)
    s["phases"]["preparation"]=dict(wall_seconds=1022.1840512752533,
        gpu_occupancy_seconds=1022.1840512752533,cpu_seconds=352.111483)
    s["budget_resources"]=dict(s["resources"])
    s["phase_budget_debits"]=copy.deepcopy(s["phases"])
    for k in ("wall_seconds","gpu_occupancy_seconds","cpu_seconds"):
        s["budget_resources"][k]+=80
        s["phase_budget_debits"]["preparation"][k]+=60
        s["phase_budget_debits"]["model"][k]+=20
    s.update(status="ENVIRONMENT_STOPPED",technical_continuation_consumed=True,
        error="RuntimeError: Stopping margin reached: preparation wall_seconds",execution_budget=B)
    save(root/"summary.json",s)
    save(root/"technical_continuation_01.started.json",dict(invocation="0004"))
    save(inv/"summary.json",dict(resources=stats,phases=s["phases"],phase_budget_debits=s["phase_budget_debits"],resources_final=True))
    for stage in ("asset_preflight","environment","official_assets"):
        save(inv/(stage+".usage.json"),dict(resources_final=True,all_descendants_reaped=True,
            exit_code=-15 if stage=="official_assets" else 0,interrupted=stage=="official_assets"))
    return root


class Closure(unittest.TestCase):
    def test_additive_budget_preserves_model_global_original(self):
        before=copy.deepcopy(B);effective=closure.effective_budget(B,A)
        self.assertEqual(B,before)
        self.assertEqual(effective["model"],B["model"])
        self.assertEqual(effective["preparation"],dict(wall_seconds=2640,cpu_seconds=1260))
        broken=copy.deepcopy(A);broken["effective_preparation"]["wall_seconds"]+=1
        with self.assertRaises(RuntimeError):closure.effective_budget(B,broken)

    def test_four_failures_preserved_once_with_no_debit_reset(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=four_failures(parent)
            before={str(p.relative_to(root)):p.read_bytes() for p in (root/"invocations").rglob("*") if p.is_file()}
            original_budget=(root/"execution_budget.json").read_bytes()
            old_markers={n:(root/n).read_bytes() for n in ("model_phase_started.json","technical_continuation_01.started.json")}
            calls=[]
            def command(s,stage,args):
                calls.append(stage)
                self.assertTrue((root/closure.MARKER).exists())
                self.assertGreaterEqual(s.prior["phase_budget_debits"]["preparation"]["wall_seconds"],1082.184)
                self.assertEqual(s.budget["model"],B["model"])
                if stage=="environment":save(s.invocation/"python.json",dict(python=sys.executable,reused=False))
                if stage=="official_assets":save(s.invocation/"model_paths.json",{})
                if stage=="qualification":
                    self.assertTrue((root/closure.MODEL_MARKER).exists())
                    raise RuntimeError("CPU fixture stops before real model execution")
            with patch.object(entry,"PARENT",parent),patch.object(entry,"RUNTIME_ROOT",parent/"content"),patch.object(Supervisor,"command",command),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                first=entry.run_colab(engineering_closure=True)
                second=entry.run_colab(engineering_closure=True)
            self.assertEqual(calls,["asset_preflight","environment","official_assets","qualification"])
            self.assertTrue(second["reopened_read_only"])
            self.assertTrue(first["engineering_closure_consumed"])
            self.assertEqual(len(list((root/"invocations").iterdir())),5)
            self.assertFalse(first["environment_resume_allowed"] or first["model_phase_may_retry"])
            self.assertEqual(original_budget,(root/"execution_budget.json").read_bytes())
            for p,data in before.items():self.assertEqual((root/p).read_bytes(),data)
            for p,data in old_markers.items():self.assertEqual((root/p).read_bytes(),data)
            self.assertGreaterEqual(first["resources"]["wall_seconds"],1030.537269)
            self.assertGreaterEqual(first["budget_resources"]["wall_seconds"],1130.537269)
            self.assertGreaterEqual(first["phase_budget_debits"]["model"]["wall_seconds"],48.353218)
            self.assertEqual(first["resources"]["result_bytes"],size(root))
            self.assertEqual(sum(v["wall_seconds"] for v in first["phase_budget_debits"].values()),first["budget_resources"]["wall_seconds"])
            self.assertTrue(all(v["started"]==v["completed"]==0 for v in first["operations"].values()))

    def test_preparation_failure_consumes_closure_without_model_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=four_failures(parent)
            with patch.object(entry,"PARENT",parent),patch.object(entry,"RUNTIME_ROOT",parent/"content"),patch.object(Supervisor,"command",side_effect=RuntimeError("fixture preflight failure")) as command,patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                first=entry.run_colab(engineering_closure=True);second=entry.run_colab(engineering_closure=True)
            self.assertEqual(command.call_count,1)
            self.assertFalse((root/closure.MODEL_MARKER).exists())
            self.assertTrue(second["reopened_read_only"])
            self.assertGreaterEqual(first["phase_budget_debits"]["preparation"]["wall_seconds"],1102.184)

    def test_reject_unsealed_started_changed_history(self):
        for change in ("unsealed","row","model","phases","budget"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=four_failures(Path(d));s=json.loads((root/"summary.json").read_text())
                if change=="unsealed":s["resources_final"]=False
                if change=="row":
                    rows=json.loads((root/"operation_status.json").read_text());rows[0]["status"]="FAILED";save(root/"operation_status.json",rows)
                if change=="model":save(root/"technical_continuation_01.model_started.json",{})
                if change=="phases":s["phase_budget_debits"]["model"]["wall_seconds"]=0
                if change=="budget":s["execution_budget"]={}
                with self.assertRaises(RuntimeError):closure.admit(root,M,B,s)

    def test_cold_admission_uses_debits_not_cache_hits_or_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inv=root/"inv";inv.mkdir();cache=root/"cache";cache.mkdir()
            budget=closure.effective_budget(B,A)
            prior=dict(resources={},phases={"preparation":dict(wall_seconds=100,cpu_seconds=10,gpu_occupancy_seconds=100)},
                phase_budget_debits={"preparation":dict(wall_seconds=1600,cpu_seconds=100,gpu_occupancy_seconds=1600)})
            with patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                s=Supervisor(root,inv,M,M["inherited_resources"],prior,budget,time.time(),own_cpu(),cache,0)
                with self.assertRaisesRegex(RuntimeError,"Insufficient remaining"):
                    closure.admit_cost(s,"cold",dict(interpreter_exists=True,cache_hit=True))
            self.assertFalse(json.loads((inv/"cold_cost_admission.json").read_text())["admitted"])

    def test_cleanup_margin_leaves_room_for_ordinary_and_terminal_sealing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inv=root/"inv";inv.mkdir();cache=root/"cache";cache.mkdir();Ledger(root,M)
            budget=closure.effective_budget(B,A);clock=[100.]
            with patch("wam_qualification.phases.time.time",side_effect=lambda:clock[0]),patch("wam_qualification.phases.time.monotonic",side_effect=lambda:clock[0]),patch("wam_qualification.phases.own_cpu",return_value=0.),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                s=Supervisor(root,inv,M,M["inherited_resources"],dict(resources={},phases={}),budget,100.,0,cache,0)
                clock[0]+=2580.5
                self.assertIn("preparation",s.reason(s.usage()))
                clock[0]+=12 # bounded owner cancellation
                original=s.write_settlement
                def slow(*args):
                    original(*args);clock[0]+=5
                with patch.object(s,"write_settlement",slow):result=s.finish("fixture stop")
            self.assertLess(result["phase_budget_debits"]["preparation"]["wall_seconds"],2640)

    def test_bad_metadata_rejected_before_any_weight_or_head_request(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);save(root/"manifest.json",M)
            calls=[]
            def fetch(url,target,out):
                calls.append(url);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text("{}" if target.suffix==".json" else "fixture")
                return target
            with patch.object(setup,"fetch",fetch),patch.object(transport,"head",side_effect=AssertionError("No weight HEAD before metadata")):
                with self.assertRaisesRegex(RuntimeError,"component index"):
                    transport.preflight(root,root/"old",root/"new")
            self.assertFalse(any(u.endswith((".pth",".safetensors")) for u in calls))


if __name__=="__main__":unittest.main(verbosity=2)
