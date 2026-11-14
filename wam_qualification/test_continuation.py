"""One technical continuation, original debits, and terminal sealing boundaries."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import separated_entry as entry
from .continuation import MARKER, MODEL_MARKER
from .core import save, Ledger
from .entry import size
from .phases import Supervisor, own_cpu, combine
from .test_receipts import previous_failures
from .test_phases import M, B


def three_failures(parent):
    root=previous_failures(parent)
    prior=json.loads((root/"summary.json").read_text())
    inv=root/"invocations/0003";inv.mkdir()
    stats=dict(wall_seconds=503.1773302555084,gpu_occupancy_seconds=503.1773302555084,
        cpu_seconds=186.058173,peak_ram_bytes=1544597504,peak_device_used_bytes=633339904.,
        download_charge_bytes=0,result_bytes=831622)
    phases=dict(preparation=dict(wall_seconds=564.015926361084,gpu_occupancy_seconds=564.015926361084,cpu_seconds=180.331498),
        model=dict(wall_seconds=8.353218078613281,gpu_occupancy_seconds=8.353218078613281,cpu_seconds=8.044590999999997))
    save(inv/"summary.json",dict(resources=stats,phases=phases,resources_final=True,status="MODEL_STOPPED"))
    for stage in ("asset_preflight","environment","official_assets","qualification"):
        save(inv/(stage+".usage.json"),dict(resources_final=True,all_descendants_reaped=True,
            exit_code=1 if stage=="qualification" else 0,interrupted=False))
    (inv/"qualification.log").write_text("OSError: base does not appear to have a file named preprocessor_config.json")
    save(root/"model_phase_started.json",dict(invocation="0003"))
    save(root/"worker_started.json",dict(started_unix=123.))
    prior.update(resources=combine(prior["resources"],stats),phases=phases,status="MODEL_STOPPED",environment_resume_allowed=False)
    save(root/"summary.json",prior)
    return root


class Continuation(unittest.TestCase):
    def test_existing_conservative_debits_are_not_reduced(self):
        from .continuation import debit_history
        prior=dict(resources=dict(wall_seconds=10.,gpu_occupancy_seconds=10.,cpu_seconds=5.),
            budget_resources=dict(wall_seconds=30.,gpu_occupancy_seconds=30.,cpu_seconds=25.),
            phases=dict(preparation=dict(wall_seconds=10.,gpu_occupancy_seconds=10.,cpu_seconds=5.)),
            phase_budget_debits=dict(preparation=dict(wall_seconds=30.,gpu_occupancy_seconds=30.,cpu_seconds=25.)))
        debit_history(prior,dict(tail_uncertainty_reserves=dict(preparation=20.)))
        self.assertGreaterEqual(prior["budget_resources"]["cpu_seconds"],45.)
        self.assertGreaterEqual(prior["phase_budget_debits"]["preparation"]["wall_seconds"],50.)
        self.assertEqual(prior["resources"]["wall_seconds"],10.)

    def test_three_failures_one_continuation_carries_both_phase_debits(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=three_failures(parent)
            before={str(p.relative_to(root)):p.read_bytes() for p in (root/"invocations").rglob("*") if p.is_file()}
            old_marker=(root/"model_phase_started.json").read_bytes()
            calls=[]
            def command(s,stage,args):
                calls.append(stage)
                self.assertTrue((root/MARKER).exists())
                self.assertEqual(s.budget,B)
                self.assertAlmostEqual(s.prior["resources"]["wall_seconds"],572.3691444396973)
                self.assertAlmostEqual(s.prior["phase_budget_debits"]["preparation"]["wall_seconds"],604.015926361084)
                self.assertAlmostEqual(s.prior["phase_budget_debits"]["model"]["wall_seconds"],28.353218078613281)
                if stage=="environment":save(s.invocation/"python.json",dict(python=sys.executable))
                if stage=="official_assets":save(s.invocation/"model_paths.json",{})
                if stage=="qualification":
                    self.assertTrue((root/MODEL_MARKER).exists())
                    debit=json.loads((s.root/"resources.json").read_text())["resources"]
                    self.assertGreaterEqual(debit["gpu_occupancy_seconds"],28.353218078613281)
                    save(s.root/"numeric_result.json",dict(status="SYNTHETIC_ONLY",qualification_pass=False))
            with patch.object(entry,"PARENT",parent),patch.object(entry,"RUNTIME_ROOT",parent/"content"),patch.object(Supervisor,"command",command),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                final=entry.run_colab(technical_continuation=True)
                second=entry.run_colab(technical_continuation=True)
            self.assertEqual(calls,["asset_preflight","environment","official_assets","qualification"])
            self.assertTrue(second["reopened_read_only"])
            self.assertEqual(len(list((root/"invocations").iterdir())),4)
            self.assertFalse(final["environment_resume_allowed"] or final["model_phase_may_retry"])
            self.assertEqual(old_marker,(root/"model_phase_started.json").read_bytes())
            for p,b in before.items():self.assertEqual((root/p).read_bytes(),b)
            self.assertEqual(final["resources"]["result_bytes"],size(root))
            self.assertAlmostEqual(final["budget_resources"]["wall_seconds"]-final["resources"]["wall_seconds"],80.)
            self.assertAlmostEqual(sum(v["wall_seconds"] for v in final["phase_budget_debits"].values()),final["budget_resources"]["wall_seconds"])
            self.assertTrue(final["historical_cpu_exact"] is False)
            self.assertEqual(final["execution_budget"],B)

    def test_preparation_failure_also_consumes_the_one_continuation(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=three_failures(parent)
            with patch.object(entry,"PARENT",parent),patch.object(entry,"RUNTIME_ROOT",parent/"content"),patch.object(Supervisor,"command",side_effect=RuntimeError("fixture setup failure")) as command,patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                first=entry.run_colab(technical_continuation=True)
                second=entry.run_colab(technical_continuation=True)
            self.assertEqual(command.call_count,1)
            self.assertFalse(first["environment_resume_allowed"])
            self.assertTrue(second["reopened_read_only"])
            self.assertTrue((root/MARKER).exists())
            self.assertFalse((root/MODEL_MARKER).exists())

    def test_admission_rejects_started_rows_other_error_unsealed_and_exhausted(self):
        for change in ("row","error","unsealed","exhausted","phase_transfer"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                parent=Path(d);root=three_failures(parent)
                summary=json.loads((root/"summary.json").read_text())
                if change=="row":
                    rows=json.loads((root/"operation_status.json").read_text());rows[0]["status"]="FAILED";save(root/"operation_status.json",rows)
                elif change=="error":(root/"invocations/0003/qualification.log").write_text("CUDA out of memory")
                elif change=="unsealed":save(root/"invocations/0003/qualification.usage.json",dict(resources_final=False))
                elif change=="phase_transfer":summary["phases"]["model"]["wall_seconds"]=0;save(root/"summary.json",summary)
                else:
                    summary["phases"]["preparation"]["wall_seconds"]=1050
                    save(root/"summary.json",summary)
                    inv=json.loads((root/"invocations/0003/summary.json").read_text());inv["phases"]=summary["phases"];save(root/"invocations/0003/summary.json",inv)
                with patch.object(entry,"PARENT",parent),patch.object(Supervisor,"command",side_effect=AssertionError("No work before admission")),self.assertRaises(RuntimeError):
                    entry.run_colab(technical_continuation=True)
                self.assertFalse((root/MARKER).exists())
                self.assertEqual(len(list((root/"invocations").iterdir())),3)

    def test_without_new_authorization_flag_original_model_reservation_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=three_failures(parent)
            with patch.object(entry,"PARENT",parent):result=entry.run_colab()
            self.assertTrue(result["reopened_read_only"])
            self.assertFalse((root/MARKER).exists())

    def test_terminal_failure_retains_lock_costs_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);root=three_failures(parent)
            with patch.object(entry,"PARENT",parent),patch.object(entry,"RUNTIME_ROOT",parent/"content"),patch.object(Supervisor,"command",side_effect=RuntimeError("fixture setup")),patch.object(Supervisor,"verify_terminal",side_effect=RuntimeError("terminal exceeded")),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                result=entry.run_colab(technical_continuation=True)
                reopened=entry.run_colab(technical_continuation=True)
            self.assertEqual(result["status"],"ACCOUNTING_UNRESOLVED")
            self.assertAlmostEqual(result["prior_resources"]["wall_seconds"],572.3691444396973)
            self.assertGreaterEqual(result["prior_budget_resources"]["wall_seconds"],632.3691444396973)
            self.assertGreaterEqual(result["resources"]["wall_seconds"],572.3691444396973)
            self.assertGreaterEqual(result["budget_resources"]["wall_seconds"],652.3691444396973)
            self.assertTrue((root/"active.lock").exists())
            self.assertTrue(reopened["reopened_read_only"])


class Settlement(unittest.TestCase):
    def test_delayed_ordinary_sealing_is_measured_and_terminal_is_predebited(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inv=root/"inv";inv.mkdir();cache=root/"cache";cache.mkdir();Ledger(root,M)
            clock=[100.];cores=[10.]
            with patch("wam_qualification.phases.time.time",side_effect=lambda:clock[0]),patch("wam_qualification.phases.time.monotonic",side_effect=lambda:clock[0]),patch("wam_qualification.phases.own_cpu",side_effect=lambda:cores[0]),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                s=Supervisor(root,inv,M,M["inherited_resources"],dict(resources={},phases={}),B,100.,10.,cache,0)
                original=s.write_settlement
                measured=[]
                def slow(result,stats,invocation):
                    original(result,stats,invocation)
                    measured.append(result["resources_final"])
                    clock[0]+=5;cores[0]+=.2
                with patch.object(s,"write_settlement",slow):result=s.finish("fixture stop")
            self.assertEqual(measured,[False,True])
            self.assertGreaterEqual(result["resources"]["wall_seconds"],5)
            self.assertGreaterEqual(result["resources"]["cpu_seconds"],.1999)
            self.assertEqual(result["budget_resources"]["wall_seconds"]-result["resources"]["wall_seconds"],20)
            self.assertEqual(s.terminal_observed["wall_seconds"],5)
            self.assertEqual(result["resources"]["result_bytes"],size(root))

    def test_terminal_delay_above_reserve_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inv=root/"inv";inv.mkdir();cache=root/"cache";cache.mkdir();Ledger(root,M)
            clock=[100.]
            with patch("wam_qualification.phases.time.time",side_effect=lambda:clock[0]),patch("wam_qualification.phases.time.monotonic",side_effect=lambda:clock[0]),patch("wam_qualification.phases.own_cpu",return_value=0),patch("wam_qualification.phases.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                s=Supervisor(root,inv,M,M["inherited_resources"],dict(resources={},phases={}),B,100.,0,cache,0)
                original=s.write_settlement
                def slow(result,stats,invocation):
                    original(result,stats,invocation)
                    if result["resources_final"]:clock[0]+=21
                with patch.object(s,"write_settlement",slow),self.assertRaisesRegex(RuntimeError,"Terminal seal exceeded"):
                    s.finish("fixture stop")
                self.assertEqual(s.terminal_observed["wall_seconds"],21)


if __name__=="__main__":unittest.main(verbosity=2)
