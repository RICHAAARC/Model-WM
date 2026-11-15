"""Execute embedded host bootstrap with a read-only receipt, never start work."""
import ast
import base64
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

def check(notebook):
    import nbformat
    notebook=Path(notebook);nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    assert nb.cells[0].source=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    code=nb.cells[2].source;tree=ast.parse(code)
    payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_payload' for t in n.targets))
    repository=notebook.parents[1]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
        for name in z.namelist():
            assert z.read(name)==(repository/name).read_bytes(),name
        assert not any('__pycache__' in n or '/test_' in n or '/check_' in n for n in z.namelist())
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);parent=root/'drive/Model-WM';run=parent/'wam-writer-abc-20261115-attempt01';run.mkdir(parents=True)
        (run/'summary.json').write_text(json.dumps({'status':'CPU_READ_ONLY_FIXTURE','retry':False}))
        code=code.replace('dir="/content"',f'dir={str(root)!r}')
        code=code.replace('run_colab(started=_entry_started, cpu_start=_entry_cpu)',
            f'run_colab(started=_entry_started, cpu_start=_entry_cpu, parent={str(parent)!r}, runtime={str(root / "runtime")!r})')
        # Real host extraction/import/read-only path; do not replace run_colab.
        script=root/'bootstrap.py';script.write_text(code)
        p=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,check=True)
        assert json.loads(p.stdout)['status']=='CPU_READ_ONLY_FIXTURE'
        assert not (root/'runtime').exists()
    return dict(status='NOTEBOOK_EMBEDDED_BOOTSTRAP_CPU_PASSED',cells=len(nb.cells),
        notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
        real_weights_loaded=False,gpu_used=False,scientific_pass=None,
        checks=['nbformat schema','first-cell mount only','exact embedded source equality',
                'real host bootstrap extraction/import','existing receipt read-only without subprocess or GPU'])

if __name__=='__main__':
    result=check(sys.argv[1]);Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(result['status'])
