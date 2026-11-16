import ast
import base64
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from . import rerun, protocol, entry

class RerunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root/'drive';self.parent.mkdir()
        self.package = self.root/'package';self.package.mkdir()
        for n in ['manifest.json', 'protocol.py', 'source_usage_history.json']:
            shutil.copyfile(protocol.PACKAGE/n, self.package/n)
        self.m = json.loads((self.package/'manifest.json').read_text())
        self.old = self.parent/rerun.OLD;self.old.mkdir()
        self.history = {k:dict(self.m['inherited'][k]) for k in ['cumulative_resources','cumulative_budget_debits']}
        for k in self.history:self.history[k]['wall_seconds'] += 350.992163005
        self.history.update(status='MODEL_STOPPED', resources_final=True, operations={'generate':{'started':1,'completed':1}})
        (self.old/'summary.json').write_text(json.dumps(self.history))
        (self.old/'model.log').write_text('old failure')
        self.other = self.parent/'unrelated';self.other.mkdir();(self.other/'keep').write_text('keep')

    def prepare(self):return rerun.prepare_rerun(self.parent,self.package,40*1024**3)

    def test_cleanup_and_frozen_semantics(self):
        self.prepare();self.assertFalse(self.old.exists());self.assertTrue((self.other/'keep').exists())
        carry=json.loads((self.parent/rerun.CARRY).read_text())
        self.assertEqual(carry['cumulative_resources'],self.history['cumulative_resources'])
        self.assertNotIn('error',carry)
        new=json.loads((self.package/'manifest.json').read_text())
        for k in ['works','generation','physical_limits','master_seed','key','rho','amplitude']:
            self.assertEqual(new[k],self.m[k])
        for k in self.m['budgets']:
            if k!='device_bytes':self.assertEqual(new['budgets'][k],self.m['budgets'][k])
        self.assertEqual(new['budgets']['device_bytes'],40*1024**3)

    def test_interruption_before_delete(self):
        with patch.object(rerun,'bounded_remove',side_effect=InterruptedError):
            with self.assertRaises(InterruptedError):self.prepare()
        self.assertTrue(self.old.exists());self.assertTrue((self.parent/rerun.CARRY).exists())
        self.prepare();self.assertFalse(self.old.exists())

    def test_interruption_after_delete(self):
        self.prepare()
        self.prepare() # no old receipt needed; same cumulative totals, no new sample
        carry=json.loads((self.parent/rerun.CARRY).read_text())
        self.assertEqual(carry['cumulative_resources'],self.history['cumulative_resources'])

    def test_bad_history_retains_old(self):
        self.history['resources_final']=False
        (self.old/'summary.json').write_text(json.dumps(self.history))
        with self.assertRaises(RuntimeError):self.prepare()
        self.assertTrue((self.old/'model.log').exists())

    def test_cleanup_path_guards(self):
        with self.assertRaises(RuntimeError):rerun.bounded_remove(self.parent,'../unrelated')
        self.assertTrue(self.other.exists())
        (self.old/'link').symlink_to(self.other,target_is_directory=True)
        with self.assertRaises(RuntimeError):self.prepare()
        self.assertTrue((self.other/'keep').exists())

    def test_small_gpu_no_cleanup(self):
        with patch.object(rerun.subprocess,'run') as p:
            p.return_value.stdout='14910\n'
            with self.assertRaises(RuntimeError):rerun.run_colab(parent=self.parent)
        self.assertTrue(self.old.exists());self.assertFalse((self.parent/rerun.NEW).exists())

    def test_closed_repeat_read_only(self):
        out=self.parent/rerun.NEW;out.mkdir();(out/'summary.json').write_text('{"status":"DONE"}')
        with patch.object(rerun,'gpu_capacity',side_effect=AssertionError),patch.object(rerun,'bounded_remove',side_effect=AssertionError):
            self.assertEqual(rerun.run_colab(parent=self.parent),{'status':'DONE'})
        self.assertTrue(self.old.exists())

    def test_unsealed_repeat_reserves_budget_no_execution(self):
        self.prepare();(self.parent/rerun.NEW).mkdir()
        with patch.object(protocol,'PACKAGE',self.package),patch.object(protocol,'RUN_ID',rerun.OLD),patch.object(entry,'RUN_ID',rerun.OLD),patch.object(rerun,'gpu_capacity',side_effect=AssertionError):
            result=rerun.run_colab(parent=self.parent,runtime=self.root/'runtime')
        self.assertEqual(result['status'],'STOPPED_UNSEALED_READ_ONLY')
        self.assertGreaterEqual(result['cumulative_budget_debits']['wall_seconds'],
            self.history['cumulative_budget_debits']['wall_seconds']+23400)
        self.assertFalse((self.root/'runtime').exists())

    def test_new_execution_dispatch(self):
        with patch.object(protocol,'PACKAGE',self.package),patch.object(protocol,'RUN_ID',rerun.OLD),patch.object(entry,'RUN_ID',rerun.OLD),patch.object(rerun,'gpu_capacity',return_value=40*1024**3),patch.object(entry,'run_colab',return_value={'status':'CPU_STUB'}) as run:
            self.assertEqual(rerun.run_colab(parent=self.parent),{'status':'CPU_STUB'})
            self.assertEqual(protocol.read()['run_id'],rerun.NEW)
            run.assert_called_once()

    def test_notebook_package_and_real_bootstrap(self):
        import nbformat
        path=protocol.PACKAGE.parent/'notebooks/latent_control_rerun_colab.ipynb'
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        self.assertEqual(nb.cells[0].source,"from google.colab import drive\ndrive.mount('/content/drive')\n")
        code=nb.cells[2].source
        for c in nb.cells:
            if c.cell_type=='code':compile(c.source,'notebook','exec')
        payload=next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_payload' for t in n.targets))
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
            for n in z.namelist():self.assertEqual(z.read(n),(protocol.PACKAGE.parent/n).read_bytes(),n)
        out=self.parent/rerun.NEW;out.mkdir();(out/'summary.json').write_text('{"status":"CPU_READ_ONLY"}')
        code=code.replace('dir="/content"',f'dir={str(self.root)!r}').replace(
            'run_colab(started=_started, cpu_start=_cpu_start)',
            f'run_colab(started=_started,cpu_start=_cpu_start,parent={str(self.parent)!r},runtime={str(self.root/"runtime")!r})')
        import subprocess,sys
        script=self.root/'bootstrap.py';script.write_text(code)
        result=subprocess.run([sys.executable,str(script)],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(result.stdout)['status'],'CPU_READ_ONLY')
        self.assertFalse((self.root/'runtime').exists())

if __name__=='__main__':unittest.main()
