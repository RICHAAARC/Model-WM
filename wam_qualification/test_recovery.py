"""CPU-only technical recovery regressions: no weights, downloads or GPU."""
import copy
import io
import json
import ssl
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from .core import save, Ledger
from .entry import PACKAGE, RUN_ID, OLD_NAME, Supervisor, run_colab, size
from .recovery import admit, SEGMENT
from .transport import ORIGINAL, OFFICIAL_MIRROR, head


M = json.loads((PACKAGE/"manifest.json").read_text())
USED = dict(wall_seconds=255.0184359550476, gpu_occupancy_seconds=255.0184359550476,
    cpu_seconds=199.421128, peak_ram_bytes=691040256, peak_device_used_bytes=3145728.,
    download_charge_bytes=3223378974, result_bytes=764526)


def fixture(parent):
    original = parent/RUN_ID
    original.mkdir()
    save(original/"manifest.json",M); Ledger(original,M)
    inherited=M["inherited_resources"]
    cumulative={k:max(inherited[k],v) if k.startswith("peak_") else inherited[k]+v for k,v in USED.items()}
    save(original/"summary.json",dict(status="STOPPED_INCOMPLETE", resources_final=True,
        error="RuntimeError: official_assets exited 1; see official_assets.log",resources=USED,
        inherited_resources=inherited,cumulative_resources=cumulative,
        inherited_trajectories_started=44,inherited_trajectories_completed=44,
        operations={k:dict(denominator=v,started=0,completed=0) for k,v in M["physical_limits"].items()}))
    save(original/"python.json",dict(python="/synthetic/python",reused=False))
    (original/"executed_source").mkdir()
    for n in ("requirements-colab.txt","requirements-cpu.txt"):
        (original/"executed_source"/n).write_bytes((PACKAGE/n).read_bytes())
    (original/"downloads.jsonl").write_text(json.dumps(dict(url=ORIGINAL,status="FAILED",received_bytes=0,error="certificate has expired"))+"\n")
    old=parent/OLD_NAME/"rerun-02";old.mkdir(parents=True)
    save(old/"latest.json",dict(trajectories_started=44,trajectories_completed=44,cumulative_resources=inherited))
    cache=parent/"cache"/RUN_ID;cache.mkdir(parents=True);(cache/"paid.bin").write_bytes(b"already paid")
    return original


class Recovery(unittest.TestCase):
    def test_admission_subtracts_original_cost_from_both_envelopes(self):
        with tempfile.TemporaryDirectory() as d:
            a=admit(fixture(Path(d)),M,PACKAGE)
            self.assertAlmostEqual(a["effective_limits"]["gpu_seconds"],644.9815640449524)
            self.assertAlmostEqual(a["effective_limits"]["cpu_seconds"],400.578872)
            self.assertAlmostEqual(a["carry"]["gpu_occupancy_seconds"],2755.732841730118)

    def test_any_started_operation_or_worker_blocks_recovery(self):
        for change in ("row","worker","summary"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root=fixture(Path(d))
                if change=="row":
                    rows=json.loads((root/"operation_status.json").read_text());rows[0]["status"]="FAILED";save(root/"operation_status.json",rows)
                elif change=="worker": (root/"worker_started.json").write_text("{}")
                else:
                    x=json.loads((root/"summary.json").read_text());x["operations"]["generate"]["started"]=1;save(root/"summary.json",x)
                with self.assertRaises(RuntimeError): admit(root,M,PACKAGE)

    def test_unsealed_regressed_exhausted_and_changed_manifest_block(self):
        for change in ("unsealed","regressed","nan","exhausted","manifest"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=fixture(Path(d));x=json.loads((root/"summary.json").read_text())
                if change=="unsealed": x["resources_final"]=False
                elif change=="regressed": x["cumulative_resources"]["cpu_seconds"]-=1
                elif change=="nan": x["resources"]["cpu_seconds"]=float("nan")
                elif change=="exhausted":
                    x["resources"]["cpu_seconds"]=600;x["cumulative_resources"]["cpu_seconds"]=x["inherited_resources"]["cpu_seconds"]+600
                else:
                    bad=copy.deepcopy(M);bad["master_seed"]+=1;save(root/"manifest.json",bad)
                (root/"summary.json").write_text(json.dumps(x))
                with self.assertRaises(RuntimeError): admit(root,M,PACKAGE)

    def test_cache_growth_charges_only_new_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/"cache";cache.mkdir();(cache/"paid").write_bytes(b"paid")
            sup=Supervisor(root,M,M["inherited_resources"],time.time(),0,cache,cache_baseline=size(cache))
            (cache/"new").write_bytes(b"newcost")
            with patch("wam_qualification.entry.cpu",return_value=3),patch("wam_qualification.entry.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="0")):
                self.assertEqual(sup.usage()["download_charge_bytes"],7)

    def test_real_entry_order_failure_finalization_and_reopen(self):
        for fail_preflight in (False, True):
            with self.subTest(fail=fail_preflight),tempfile.TemporaryDirectory() as d:
                parent=Path(d);root=fixture(parent);before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file()}
                calls=[]
                def command(s, stage, args):
                    calls.append(stage)
                    if stage=="asset_preflight" and fail_preflight: raise RuntimeError("TLS preflight failed")
                    if stage=="environment": save(s.root/"python.json",dict(python="/synthetic/python"))
                def usage(s, process=None):
                    return dict(wall_seconds=2,gpu_occupancy_seconds=2,cpu_seconds=1,
                        peak_ram_bytes=1,peak_device_used_bytes=0,download_charge_bytes=7,result_bytes=100)
                with patch("wam_qualification.entry.PARENT",parent),patch("wam_qualification.entry.RUNTIME_ROOT",parent/"runtime"),patch.object(Supervisor,"command",command),patch.object(Supervisor,"usage",usage),patch("wam_qualification.entry.subprocess.run",return_value=SimpleNamespace(returncode=0,stdout="synthetic device")):
                    r=run_colab(recovery=True)
                    again=run_colab(recovery=True)
                self.assertEqual(calls,["asset_preflight"] if fail_preflight else ["asset_preflight","environment","official_assets","qualification"])
                self.assertTrue(again["reopened_read_only"])
                self.assertAlmostEqual(r["cumulative_resources"]["gpu_occupancy_seconds"],2757.732841730118)
                self.assertEqual(r["cumulative_resources"]["download_charge_bytes"],14014744434)
                self.assertTrue(r["resources_final"])
                self.assertEqual(r["operations"]["generate"]["started"],0)
                for name,data in before.items(): self.assertEqual((root/name).read_bytes(),data)

    def test_paid_wheels_install_offline_without_download(self):
        from .setup import environment
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/"out";out.mkdir();scratch=root/"scratch";cache=root/"cache";cache.mkdir();(cache/"wheels").mkdir()
            save(out/"recovery_admission.json",{})
            oldpy=root/"runtime/aqualora-qualification-20261108-attempt01/persistent-runtime/python/python/bin/python3.10"
            oldpy.parent.mkdir(parents=True);oldpy.write_text("synthetic")
            with patch("wam_qualification.setup.RUNTIME_ROOT",root/"runtime"),patch("wam_qualification.setup.call") as call:
                environment(out,scratch,root/"old",cache)
            commands=[[str(x) for x in c.args[0]] for c in call.call_args_list]
            self.assertFalse(any("download" in c for c in commands))
            self.assertTrue(any("--no-index" in c for c in commands))


class Transport(unittest.TestCase):
    def test_head_uses_verified_official_mirror_without_reading_body(self):
        response=SimpleNamespace(status=200,url=OFFICIAL_MIRROR,headers={"Content-Length":"123"})
        class Context:
            def __enter__(self): return response
            def __exit__(self,*args): pass
        with patch("urllib.request.urlopen",return_value=Context()) as opening:
            row=head(ORIGINAL)
        self.assertEqual(opening.call_args.args[0].full_url,OFFICIAL_MIRROR)
        self.assertEqual(opening.call_args.args[0].method,"HEAD")
        self.assertNotIn("context",opening.call_args.kwargs)
        self.assertFalse(row["body_downloaded"])

    def test_tls_failure_has_one_call_and_no_checkpoint(self):
        from .setup import fetch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch("urllib.request.urlopen",side_effect=ssl.SSLCertVerificationError("expired")) as opening:
                with self.assertRaises(ssl.SSLCertVerificationError): fetch(ORIGINAL,root/"model.pth",root)
            self.assertEqual(opening.call_count,1)
            self.assertFalse((root/"model.pth").exists())
            self.assertEqual(json.loads((root/"downloads.jsonl").read_text())["received_bytes"],0)

    def test_mirror_download_keeps_source_and_paid_reuse(self):
        from .setup import fetch
        response=io.BytesIO(b"synthetic bytes");response.headers={"Content-Length":"15"}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch("urllib.request.urlopen",return_value=response) as opening:
                fetch(ORIGINAL,root/"model",root);fetch(ORIGINAL,root/"model",root)
            self.assertEqual(opening.call_count,1)
            receipt=json.loads((root/"model.complete.json").read_text())
            self.assertEqual(receipt["url"],ORIGINAL);self.assertEqual(receipt["transport_url"],OFFICIAL_MIRROR)


if __name__ == "__main__": unittest.main(verbosity=2)
