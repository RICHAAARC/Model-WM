"""Self-contained entry: source embedded, no moving Git branch at runtime."""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
stream=io.BytesIO()
with zipfile.ZipFile(stream,"w",zipfile.ZIP_DEFLATED) as z:
    for p in sorted((ROOT/"wam_qualification").iterdir()):
        if p.is_file() and not p.name.startswith(("test_","check_")) and p.suffix in (".py",".json",".txt",".md"):
            info=zipfile.ZipInfo("wam_qualification/"+p.name, date_time=(2026,11,9,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
payload=base64.b64encode(stream.getvalue()).decode()
code='''import time, resource
_entry_started = time.time()
_r, _c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
_entry_cpu = _r.ru_utime + _r.ru_stime + _c.ru_utime + _c.ru_stime
import base64, io, json, pathlib, sys, tempfile, zipfile
_package = pathlib.Path(tempfile.mkdtemp(prefix="wam-frozen-entry-", dir="/content"))
_payload = PAYLOAD
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_payload))) as _z:
    for _name in _z.namelist():
        _dst = (_package / _name).resolve()
        if _package not in _dst.parents:
            raise RuntimeError("Unsafe embedded package")
    _z.extractall(_package)
# A repeated cell must use this frozen package, not an earlier imported version.
for _name in list(sys.modules):
    if _name == "wam_qualification" or _name.startswith("wam_qualification."):
        del sys.modules[_name]
sys.path.insert(0, str(_package))
from wam_qualification.entry import run_colab
result = run_colab(started_unix=_entry_started, started_cpu=_entry_cpu, recovery=True)
print(json.dumps(result, ensure_ascii=False, indent=2))
'''.replace('PAYLOAD',repr(payload))
def cell(kind,source):
    out=dict(cell_type=kind,metadata={},source=source.splitlines(True))
    if kind=="code": out.update(execution_count=None,outputs=[])
    return out
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name="WAM qualification — technical recovery 02"),
    kernelspec=dict(name="python3",display_name="Python 3"),accelerator="GPU"),cells=[
    cell("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),
    cell("markdown","# WAM qualification — technical recovery 02\n16 works · 64 writes · 160 reads. Same frozen attempt; no training or CAL/TEST.\n\nThis entry fixes the inherited Colab Matplotlib backend error, continues only the sealed zero-operation recovery-01 failure, and retains both failure segments. CPU preflight uses synthetic checkpoints through the actual WAM and SD loaders; it does not certify real checkpoint or CUDA execution.\n\nReuse the current GPU runtime when available. Select one GPU and Run all once; no code upload is required. Existing paid weights are copied into `/content`, and missing weights download only into `/content`. Drive receives results, receipts and failure logs, with no new model payloads. Previously downloaded Drive caches are read only and are not deleted. A runtime reset loses new local assets; reopening this segment remains read-only and does not retry.\n\nOutput: `MyDrive/Model-WM/wam-qualification-20261109-attempt01/technical-recovery-02/`. The complete cumulative receipt is inherited: GPU/wall 3109.523371696472 s, CPU 2326.552533 core-s, downloads 14392570601 bytes. Remaining original WAM caps are at most 291.191034 s conservative GPU-clock, 1191.191034 s wall and 235.546589 CPU core-s, before the unchanged stopping margins. Setup, copying, checks, new downloads, loading and IO consume this remainder. Frozen codebook, seed, parameters, 16/64/160 limits and the historical 44 trajectories are unchanged.\n\nGPU execution is user initiated. Source ambiguity, numerical failure, or insufficient budget stops the attempt; completion is not guaranteed by CPU validation. Inspect summary and all sixteen panels if completed.\n"),
    cell("code",code)])
for i,c in enumerate(notebook['cells']): c['id']=f'wam-{i:02d}'
destination=ROOT/"notebooks/wam_qualification_colab.ipynb"
destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
print(destination)
