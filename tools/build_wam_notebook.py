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
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name="WAM qualification — same-attempt recovery"),
    kernelspec=dict(name="python3",display_name="Python 3"),accelerator="GPU"),cells=[
    cell("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),
    cell("markdown",'# WAM qualification — technical recovery\n16 works · 64 writes · 160 reads. Same frozen attempt; no training or CAL/TEST.\n\n**2026-11-12 repair reviewed:** Gemini completed the transport/recovery code audit with no confirmed blocking defect.20 CPU regressions,8 official synthetic-interface checks and14 resource HEAD checks passed. These do not certify real weights, CUDA behavior, quality or completion within the remaining budget.\n\nThe original Meta download certificate expired. This entry uses the publisher’s fixed Hugging Face checkpoint route with TLS verification, and checks missing official assets before installation. Keep the existing Drive history and caches. Select one GPU and Run all once; no code upload is required. Reuse the current Colab runtime when available.\n\nOriginal failure files stay unchanged. Output: `MyDrive/Model-WM/wam-qualification-20261109-attempt01/technical-recovery-01/`. Admission requires the sealed zero-operation certificate failure. It carries cumulative GPU/wall2755.73284s and CPU2161.52025s, leaving at most644.98156s conservative GPU-clock /1544.98156s wall /400.578872 CPU core-seconds under the original caps. Setup, checks, downloads, loading and IO consume that remainder. Reopening only reads the saved recovery; there is no further retry.\n\nActual model execution is user initiated. Budget/source/quality failure remains a valid stop outcome; inspect the full summary and all sixteen panels if completed.\n'),
    cell("code",code)])
for i,c in enumerate(notebook['cells']): c['id']=f'wam-{i:02d}'
destination=ROOT/"notebooks/wam_qualification_colab.ipynb"
destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
print(destination)
