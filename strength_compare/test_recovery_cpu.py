"""Formal owner error paths; consumes only synthetic CPU calls."""
import json,os,signal,sys,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from . import entry
from .test_cpu import install_dev

class Recovery(unittest.TestCase):
    def test_model_failure_seals_and_reopening_is_readonly(self):
        original=entry.Monitor.command
        def command(m,stage,cmd,env):
            if stage=='preparation':
                script="from latent_control.io import save;from pathlib import Path;import sys;save(Path(sys.argv[1])/'python.json',{'python':sys.executable})"
            else:
                script="from latent_control.io import append;from pathlib import Path;import sys;append(Path(sys.argv[1])/'operations.jsonl',dict(kind='generate',state='STARTED',id='CAL-000'));raise RuntimeError('Injected CPU model failure')"
            return original(m,stage,[sys.executable,'-c',script,*cmd[3:]],env)
        with tempfile.TemporaryDirectory() as tmp,patch.object(entry,'gpu_ready',return_value=True),patch.object(entry.Monitor,'command',command):
            p=Path(tmp);install_dev(p/'drive');r=entry.run_colab(p/'drive',p/'content')
            self.assertEqual(r['status'],'FORMAL_STOPPED_INCOMPLETE')
            self.assertEqual(r['operations']['generate'],dict(started=1,completed=0,denominator=184))
            self.assertGreater(r['resources']['wall_seconds'],0)
            self.assertIn('Traceback',r['traceback'])
            out=Path(r['output']);self.assertIn('Injected CPU model failure',(out/'model.log').read_text())
            self.assertEqual(r['resources']['result_bytes'],entry.tree_bytes(out))
            before={x.name:x.read_bytes() for x in out.iterdir() if x.is_file()}
            with patch.object(entry.Monitor,'command',side_effect=AssertionError('No retry')):
                self.assertEqual(entry.run_colab(p/'drive',p/'content'),r)
            self.assertEqual(before,{x.name:x.read_bytes() for x in out.iterdir() if x.is_file()})

    def test_interrupt_reaps_child_and_retains_cpu_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);local=p/'local';local.mkdir()
            m=entry.Monitor(time.monotonic(),entry.owner_cpu(),local,p,p/'cache')
            timer=threading.Timer(.5,lambda:os.kill(os.getpid(),signal.SIGINT));timer.start()
            try:
                with self.assertRaises(KeyboardInterrupt):m.command('model',[sys.executable,'-c','import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);time.sleep(60)'],dict(os.environ))
            finally:timer.cancel()
            usage=entry.read(local/'model.usage.json')
            self.assertTrue(usage['resources_final']);self.assertTrue(usage['all_descendants_reaped'])
            self.assertGreater(m.stages['model']['wall_seconds'],0)
            self.assertGreaterEqual(m.stages['model']['cpu_seconds'],0)

    def test_live_partial_ledger_and_atomic_progress(self):
        from latent_control.io import save
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'operations.jsonl').write_bytes(b'{"kind":"read","state":"STARTED"}\n{"kind":"read"')
            r=entry.operation_counts(p,live=True)
            self.assertEqual(r['read']['started'],1);self.assertEqual(r['read']['completed'],0)
            save(p/'progress.json',dict(n=0));errors=[]
            def writer():
                for i in range(100):save(p/'progress.json',dict(n=i))
            t=threading.Thread(target=writer);t.start()
            while t.is_alive():
                try:self.assertIsInstance(entry.read(p/'progress.json')['n'],int)
                except Exception as e:errors.append(repr(e))
            t.join();self.assertEqual(errors,[])

if __name__=='__main__':unittest.main()
