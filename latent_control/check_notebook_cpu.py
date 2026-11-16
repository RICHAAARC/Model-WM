"""Execute the real frozen host bootstrap against an existing read-only receipt."""
import ast
import base64
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

def check(path):
    import nbformat
    path=Path(path);nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    assert nb.cells[0].source=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    code=nb.cells[2].source
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,'notebook','exec')
    payload=next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='_payload' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
        for name in z.namelist():assert z.read(name)==(path.parents[1]/name).read_bytes(),name
        assert 'latent_control/frozen.npz' in z.namelist()
        assert not any(n.endswith(('.safetensors','.pth','.pt')) or '/review/' in n for n in z.namelist())
    with tempfile.TemporaryDirectory() as d:
        p=Path(d);parent=p/'drive';out=parent/'latent-control-20261116-attempt01';out.mkdir(parents=True)
        (out/'summary.json').write_text(json.dumps(dict(status='CPU_READ_ONLY_FIXTURE',retry=False)))
        script=code.replace('dir="/content"',f'dir={str(p)!r}').replace('run_colab(started=_started, cpu_start=_cpu_start)',
            f'run_colab(started=_started, cpu_start=_cpu_start, parent={str(parent)!r}, runtime={str(p/"runtime")!r})')
        target=p/'bootstrap.py';target.write_text(script)
        completed=subprocess.run([sys.executable,str(target)],check=True,capture_output=True,text=True)
        assert json.loads(completed.stdout)['status']=='CPU_READ_ONLY_FIXTURE'
        assert not (p/'runtime').exists()
    result=dict(status='NOTEBOOK_CPU_BOOTSTRAP_PASSED',cells=len(nb.cells),
        checks=['schema','first-cell Drive mount only','exact frozen embedded sources/manifest/arrays',
                'real host extraction and imports','repeat-click read-only without installation/model subprocess'],
        weights_loaded=False,gpu_used=False,scientific_pass=None)
    (path.parents[1]/'latent_control/review/notebook_cpu.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])

if __name__=='__main__':check(sys.argv[1])
