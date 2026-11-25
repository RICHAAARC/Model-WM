"""Upload-free DEV-only release with a deterministic embedded source bundle."""
from pathlib import Path
import io,json,hashlib,base64
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'sd2_matched'
def files():
    # Reuse the proven dependency closure, without importing the older entry.
    old=json.loads((ROOT/'matched_loss/receipt.json').read_text())['sha256']
    names=set(old)
    packages={
      'sd2_matched':'__init__.py entry.py worker.py model.py preflight.py assets.py allocation.json PREREGISTRATION.md SOURCE_AUDIT.md prior_summary.json',
      'matched_loss':'__init__.py carrier.py',
      'probe_selection':'__init__.py rules.py owner.py worker_paths.py carrier.py',
      'lawa_ab':'__init__.py entry.py assets.py model.py',
      'latent_control':'__init__.py io.py',
      'wam_method':'__init__.py entry.py protocol.py io.py',
      'wam_qualification':'__init__.py metered.py core.py transport.py setup.py prepare.py storage.py environment.py requirements-colab.txt',
    }
    for pkg,items in packages.items():names.update(pkg+'/'+n for n in items.split())
    return names
def build():
    names=files()
    receipt=dict(frozen_at='2026-11-25',stage='DEV_ONLY',scientific_pass=None,
        sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)})
    (P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    names.add('sd2_matched/receipt.json');buf=io.BytesIO()
    with ZipFile(buf,'w') as z:
        for n in sorted(names):
            info=ZipInfo(n,(2026,11,25,0,0,0));info.compress_type=ZIP_DEFLATED
            z.writestr(info,(ROOT/n).read_bytes())
    payload=base64.b64encode(buf.getvalue()).decode()
    desc='''# LaWa × SD2.1：公平比较 DEV
选择 GPU 后“全部运行”，无需上传代码。

本入口只运行 **8 件新 DEV：8 次生成、240 次写入、24 次接口读取**。前两件接口验收计入总数；不进入 CAL/TEST。
模型构型为 SD2.1 去噪器 + 固定 SD1.4 干净解码器/LaWa；使用 Hugging Face 文档指向的社区镜像。

预计准备 5–20 分钟、模型 5–15 分钟；估计不是硬上限。准备、模型和结算独立记账，继承全部历史费用。
权重及软件仅存 /content，优先复用当前运行时可验证缓存。更换运行时后可能需要重新下载。
结果：MyDrive/Model-WM/lawa-sd21-matched-dev-20261125-attempt01/。

重复运行只读已有回执，包括失败；不会追加模型重试。
预期成功回执为 DEV_MATCHING_GATE_MET，visual_review=PENDING，scientific_pass=null。
这仅表示质量匹配和数值接口门通过。完成后请返回回执，由我们核查前两件及其余 DEV 图像，再决定是否交付正式比较入口。
'''
    runner='''#@title 运行一次冻结 DEV / 读取已存回执
import base64,io,json,pathlib,resource,sys,tempfile,time,zipfile
started=time.monotonic()
usage=resource.getrusage(resource.RUSAGE_SELF)
root=pathlib.Path(tempfile.mkdtemp(dir='/content',prefix='sd21-matched-dev-'))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
    z.extractall(root)
for key in list(sys.modules):
    if key.split('.')[0] in ('sd2_matched','matched_loss','lawa_ab','probe_selection','wam_method','wam_qualification','latent_control'):
        del sys.modules[key]
sys.path.insert(0,str(root))
from sd2_matched.entry import run_colab
result=run_colab(started=started,cpu_start=usage.ru_utime+usage.ru_stime)
print(json.dumps(result,ensure_ascii=False,indent=2))
'''.replace('PAYLOAD',repr(payload))
    cells=[]
    for kind,source in [('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',desc),('code',runner)]:
        cell=dict(id=f'sd21-dev-{len(cells)}',cell_type=kind,metadata={},source=source.splitlines(True))
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    cells[-1]['metadata']['cellView']='form'
    name='lawa_sd21_matched_dev_colab.ipynb'
    doc=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',
      kernelspec=dict(name='python3',display_name='Python 3'),colab=dict(name=name)))
    (ROOT/'notebooks'/name).write_text(json.dumps(doc,ensure_ascii=False,indent=1)+'\n')
    print(name,len(names),'embedded files',len(buf.getvalue()),'bytes')
if __name__=='__main__':build()
