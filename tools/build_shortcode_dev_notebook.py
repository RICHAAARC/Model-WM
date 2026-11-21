"""Deterministic, upload-free notebook with exactly the sealed DEV code path."""
import base64
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'notebooks/lawa_shortcode_dev_colab.ipynb'


def build():
    files=set(json.loads((ROOT/'shortcode_channel/receipt.json').read_text())['sha256'])
    for pkg,names in {
        'shortcode_channel':'__init__.py receipt.json dev_assets.py dev_worker.py dev_entry.py',
        'probe_selection':'__init__.py rules.py owner.py worker_paths.py',
        'lawa_ab':'__init__.py entry.py assets.py model.py',
        'latent_control':'__init__.py io.py',
        'wam_method':'__init__.py entry.py protocol.py io.py',
        'wam_qualification':'__init__.py metered.py core.py transport.py setup.py prepare.py storage.py environment.py requirements-colab.txt',
    }.items():
        files.update(pkg+'/'+name for name in names.split())
    data=io.BytesIO()
    with ZipFile(data,'w') as z:
        for name in sorted(files):
            info=ZipInfo(name,(2026,11,22,0,0,0))
            info.compress_type=ZIP_DEFLATED
            z.writestr(info,(ROOT/name).read_bytes())
    payload=base64.b64encode(data.getvalue()).decode()
    desc="""# LaWa 短码本 · 一次 DEV 判别
选择 GPU 后全部运行，挂载自己的 Google Drive。预注册代码与分配已内嵌。
固定 8 件作品、48 个作品与码本组合，至多生成 8 次、写入 432 次、提取 1,248 次。
准备 11 分钟、模型 14 分钟、结算 1 分钟。中断后再次运行仅读取已封存状态。
结果保存在 MyDrive/Model-WM/lawa-shortcode-dev-20261122-attempt01/。权重仅在 /content。
查看 summary.json、dev_decision.json 和 slots/ 内的原分辨率图及配对读数。CAL、TEST 保持关闭。
"""
    runner="""#@title 运行冻结 DEV 或读取既有回执
import base64, io, json, pathlib, resource, sys, tempfile, time, zipfile
started = time.monotonic()
usage = resource.getrusage(resource.RUSAGE_SELF)
root = pathlib.Path(tempfile.mkdtemp(dir='/content', prefix='shortcode-dev-'))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
    z.extractall(root)
for key in list(sys.modules):
    if key.split('.')[0] in ('shortcode_channel','lawa_ab','probe_selection','wam_method','wam_qualification','latent_control'):
        del sys.modules[key]
sys.path.insert(0,str(root))
from shortcode_channel.dev_entry import run_colab
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
