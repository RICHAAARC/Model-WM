"""Deterministic, upload-free notebook with exactly the sealed DEV code path."""
import base64
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'notebooks/lawa_matched_loss_dev_colab.ipynb'


def build():
    files=set(json.loads((ROOT/'matched_loss/receipt.json').read_text())['sha256'])
    for pkg,names in {
        'matched_loss':'__init__.py receipt.json assets.py worker.py entry.py carrier.py',
        'probe_selection':'__init__.py rules.py owner.py worker_paths.py carrier.py',
        'lawa_ab':'__init__.py entry.py assets.py model.py',
        'latent_control':'__init__.py io.py',
        'wam_method':'__init__.py entry.py protocol.py io.py',
        'wam_qualification':'__init__.py metered.py core.py transport.py setup.py prepare.py storage.py environment.py requirements-colab.txt',
    }.items():
        files.update(pkg+'/'+name for name in names.split())
    data=io.BytesIO()
    with ZipFile(data,'w') as z:
        for name in sorted(files):
            info=ZipInfo(name,(2026,11,24,0,0,0))
            info.compress_type=ZIP_DEFLATED
            z.writestr(info,(ROOT/name).read_bytes())
    payload=base64.b64encode(data.getvalue()).decode()
    desc="""# LaWa 等结构损失 DEV
选择 GPU 后全部运行。8 件作品，8 次生成、240 次写入、0 次水印读出。
准备 1080 秒、模型 900 秒、结算 60 秒，承接历史费用。权重只存 /content。
重开只读回执。DEV_MATCHING_GATE_MET 表示匹配可行，后续阶段另行判断。
"""
    runner="""#@title 运行冻结 DEV 或读取既有回执
import base64, io, json, pathlib, resource, sys, tempfile, time, zipfile
started = time.monotonic()
usage = resource.getrusage(resource.RUSAGE_SELF)
root = pathlib.Path(tempfile.mkdtemp(dir='/content', prefix='matched-loss-dev-'))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
    z.extractall(root)
for key in list(sys.modules):
    if key.split('.')[0] in ('matched_loss','shortcode_channel','lawa_ab','probe_selection','wam_method','wam_qualification','latent_control'):
        del sys.modules[key]
sys.path.insert(0,str(root))
from matched_loss.entry import run_colab
result = run_colab(started=started,cpu_start=usage.ru_utime+usage.ru_stime)
print(json.dumps(result,ensure_ascii=False,indent=2))
""".replace('PAYLOAD',repr(payload))
    cells=[]
    for kind,source in [('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',desc),('code',runner)]:
        cell=dict(id=f'dev-{len(cells)}',cell_type=kind,metadata={},source=source.splitlines(True))
        if kind=='code':
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    cells[-1]['metadata']['cellView']='form'
    doc=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',
             kernelspec=dict(name='python3',display_name='Python 3'),colab=dict(name=TARGET.name)))
    TARGET.write_text(json.dumps(doc,ensure_ascii=False,indent=1)+'\n')


if __name__=='__main__':
    build()
