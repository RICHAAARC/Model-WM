"""Known failure continuation and local-only payload storage regressions."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .test_recovery import fixture, M, PACKAGE
from .core import Ledger, save
from .recovery2 import admit, PREVIOUS
from .storage import locate, wheels
from .setup import fetch
from .transport import ORIGINAL

SECOND = dict(wall_seconds=353.79052996635437,gpu_occupancy_seconds=353.79052996635437,
    cpu_seconds=165.03228299999998,peak_ram_bytes=697356288,peak_device_used_bytes=3145728.,
    download_charge_bytes=377826174,result_bytes=751918)
TRACE = "Traceback (most recent call last):\n    from inference_utils import load_model_from_checkpoint\n    import matplotlib.pyplot as plt\nValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend; supported values are ['agg']\n"


def latest_fixture(parent):
    original=fixture(parent)
    first=json.loads((original/"summary.json").read_text())
    previous=original/PREVIOUS;previous.mkdir()
    save(previous/"manifest.json",M);Ledger(previous,M)
    inherited=first["cumulative_resources"]
    summary=copy.deepcopy(first)
    summary.update(error="RuntimeError: qualification exited 1; see qualification.log",resources=SECOND,
        inherited_resources=inherited,cumulative_resources={k:max(inherited[k],v) if k.startswith("peak_") else inherited[k]+v for k,v in SECOND.items()})
    save(previous/"summary.json",summary)
    save(previous/"worker_started.json",dict(started_unix=1794450933.717325))
    save(previous/"python.json",dict(python="/synthetic/python"))
    save(previous/"model_paths.json",dict(source="/synthetic/source"))
    (previous/"qualification.log").write_text(TRACE)
    (previous/"executed_source").mkdir()
    for n in ("requirements-colab.txt","requirements-cpu.txt"):
        (previous/"executed_source"/n).write_bytes((PACKAGE/n).read_bytes())
    return original


class SecondRecovery(unittest.TestCase):
    def test_full_chain_exact_remainder(self):
        with tempfile.TemporaryDirectory() as d:
            a=admit(latest_fixture(Path(d)),M,PACKAGE)
            self.assertAlmostEqual(a["effective_limits"]["gpu_seconds"],291.191034078598)
            self.assertAlmostEqual(a["effective_limits"]["wall_seconds"],1191.191034078598)
            self.assertAlmostEqual(a["effective_limits"]["cpu_seconds"],235.546589)
            self.assertAlmostEqual(a["carry"]["wall_seconds"],3109.523371696472)
            self.assertEqual(a["carry"]["download_charge_bytes"],14392570601)
            self.assertEqual(a["original_resources"]["download_charge_bytes"],3601205148)

    def test_refuse_ambiguous_or_consumed_history(self):
        for kind in ("row","missing_worker","unsealed","trace","chain","nan","budget","lock","manifest"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=latest_fixture(Path(d));prev=root/PREVIOUS
                s=json.loads((prev/"summary.json").read_text())
                if kind=="row":
                    rows=json.loads((prev/"operation_status.json").read_text());rows[0]["status"]="STARTED";save(prev/"operation_status.json",rows)
                elif kind=="missing_worker": (prev/"worker_started.json").unlink()
                elif kind=="unsealed": s["resources_final"]=False
                elif kind=="trace": (prev/"qualification.log").write_text("RuntimeError: out of memory\n")
                elif kind=="chain": s["inherited_resources"]["cpu_seconds"]-=1
                elif kind=="nan": s["resources"]["cpu_seconds"]=float("nan")
                elif kind=="budget":
                    s["resources"]["cpu_seconds"]=400
                    s["cumulative_resources"]["cpu_seconds"]=s["inherited_resources"]["cpu_seconds"]+400
                elif kind=="lock": (prev/"executed_source/requirements-colab.txt").write_text("changed")
                else:
                    m=copy.deepcopy(M);m["master_seed"]+=1;save(prev/"manifest.json",m)
                (prev/"summary.json").write_text(json.dumps(s))
                with self.assertRaises((RuntimeError,FileNotFoundError)): admit(root,M,PACKAGE)

    def test_drive_payload_copied_local_without_network_or_new_charge(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);drive=root/"drive";drive.mkdir();new=root/"content/new";new.mkdir(parents=True);paid=root/"content/paid"
            save(root/"local_storage.json",dict(mappings=[dict(new=str(new),drive=str(drive),paid=str(paid))]))
            (drive/"model").write_bytes(b"synthetic weights")
            save(drive/"model.complete.json",dict(url=ORIGINAL,bytes=17))
            before={p.name:p.read_bytes() for p in drive.iterdir()}
            self.assertEqual(locate(root,new/"model"),drive/"model")
            with patch("urllib.request.urlopen",side_effect=AssertionError("No download for paid cache")):
                got=fetch(ORIGINAL,new/"model",root)
                self.assertEqual(fetch(ORIGINAL,new/"model",root),got)
            self.assertEqual(got,paid/"model")
            self.assertEqual(got.read_bytes(),b"synthetic weights")
            self.assertEqual(list(new.iterdir()),[])
            self.assertEqual(before,{p.name:p.read_bytes() for p in drive.iterdir()})
            self.assertEqual(json.loads((root/"cache_restore.jsonl").read_text())["download_charge_bytes"],0)

    def test_new_download_stays_local_and_partial_restore_stops(self):
        import io
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);new=root/"content/new";new.mkdir(parents=True);drive=root/"drive";drive.mkdir();paid=root/"content/paid";paid.mkdir()
            save(root/"local_storage.json",dict(mappings=[dict(new=str(new),drive=str(drive),paid=str(paid))]))
            response=io.BytesIO(b"new payload");response.headers={"Content-Length":"11"}
            with patch("urllib.request.urlopen",return_value=response):
                self.assertEqual(fetch(ORIGINAL,new/"model",root),new/"model")
            self.assertEqual(list(drive.iterdir()),[])
            (paid/"other.part").write_bytes(b"partial")
            with patch("urllib.request.urlopen",side_effect=AssertionError("No retry")):
                with self.assertRaises(RuntimeError): fetch(ORIGINAL,new/"other",root)

    def test_paid_wheels_are_restored_locally(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);new=root/"content/new";new.mkdir(parents=True);drive=root/"drive";(drive/"wheels").mkdir(parents=True);paid=root/"content/paid"
            save(root/"local_storage.json",dict(mappings=[dict(new=str(new),drive=str(drive),paid=str(paid))]))
            (drive/"wheels/fixture.whl").write_bytes(b"fixture")
            self.assertEqual(wheels(root,new/"wheels"),paid/"wheels")
            self.assertTrue((paid/"wheels/fixture.whl").is_file())


if __name__=="__main__":unittest.main(verbosity=2)
