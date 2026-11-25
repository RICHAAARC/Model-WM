"""Self-contained Run-all DEV only; deterministic embedded audited source."""
from pathlib import Path
import io,json,hashlib,base64
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED
from tools.build_sd2_dev_notebook import files as previous_files
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'strength_bracket'
def build():
    names=previous_files()|{'strength_bracket/'+x for x in '__init__.py rules.py carrier.py assets.py preflight.py entry.py worker.py allocation.json prior_summary.json SOURCE_AUDIT.md PREREGISTRATION.md'.split()}
    receipt=dict(frozen_at='2026-11-25',stage='DEV_ONLY',scientific_pass=None,sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)})
    (P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');names.add('strength_bracket/receipt.json')
    buf=io.BytesIO()
    with ZipFile(buf,'w') as z:
        for n in sorted(names):
            info=ZipInfo(n,(2026,11,25,0,0,0));info.compress_type=ZIP_DEFLATED;z.writestr(info,(ROOT/n).read_bytes())
    payload=base64.b64encode(buf.getvalue()).decode()
    desc='''# LaWa 尺度与普通增强：强度夹逼 DEV
选择 GPU 后“全部运行”，无需上传代码。

本入口只运行 **8 件新 DEV：8 次生成、336 次写入、0 次读取、0 次混合**。
固定粗尺度对照六个普通统一增益，只根据质量读数确定低、近、高三档全局强度，不看归属结果。
SD1.4/LaWa、码本、采样器和粗尺度规则保持固定。前两件接口图像包含在八件中。

预计准备 5–20 分钟、模型 5–15 分钟；估计不是硬时间上限。
准备、模型、结算分别记账，承接完成的 SD2.1 比较累计费用。全部失败保留。
权重和软件仅存 /content，复用当前运行时有效缓存；换运行时可能需要重新下载。

结果目录：MyDrive/Model-WM/lawa-strength-bracket-dev-20261125-attempt01/
成功选择为 DEV_BRACKET_GATE_MET，visual_review=PENDING，scientific_pass=null。
不能形成质量夹逼时为 DEV_BRACKET_STOP，不扩大增益网格或追加运行。
重复点击只读已有结果（包括失败）；本入口不能进入 CAL/TEST。

运行后返回完整回执。我们核查全部 DEV 质量和原图后，才交付冻结的 24 CAL / 160 TEST 入口。
正式比较只有“粗尺度对高档普通增强的三人均分 PNG 成员数”一个确认性主终点。
'''
    runner='''#@title 一次运行冻结 DEV / 读取已有回执
import base64,io,json,pathlib,resource,sys,tempfile,time,zipfile
started=time.monotonic()
usage=resource.getrusage(resource.RUSAGE_SELF)
root=pathlib.Path(tempfile.mkdtemp(dir='/content',prefix='strength-bracket-dev-'))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(root)
for key in list(sys.modules):
    if key.split('.')[0] in ('strength_bracket','sd2_matched','matched_loss','lawa_ab','probe_selection','wam_method','wam_qualification','latent_control'):del sys.modules[key]
sys.path.insert(0,str(root))
from strength_bracket.entry import run_colab
result=run_colab(started=started,cpu_start=usage.ru_utime+usage.ru_stime)
print(json.dumps(result,ensure_ascii=False,indent=2))
'''.replace('PAYLOAD',repr(payload))
    cells=[]
    for kind,source in [('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',desc),('code',runner)]:
        c=dict(id='strength-bracket-'+str(len(cells)),cell_type=kind,metadata={},source=source.splitlines(True))
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    cells[-1]['metadata']['cellView']='form'
    name='lawa_strength_bracket_dev_colab.ipynb'
    nb=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3'),colab=dict(name=name)))
    (ROOT/'notebooks'/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    print(name,len(names),'embedded files',len(buf.getvalue()),'bytes')
if __name__=='__main__':build()
