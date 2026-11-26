"""Self-contained formal entry; leaves the consumed DEV notebook/seal unchanged."""
from pathlib import Path
import io,json,hashlib,base64
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1];P=ROOT/'strength_compare'

def build():
    names=set(json.loads((ROOT/'strength_bracket/receipt.json').read_text())['sha256'])
    names={n for n in names if n not in ('strength_bracket/worker.py','strength_bracket/entry.py')}
    names.update('strength_compare/'+n for n in ('__init__.py gate.py rules.py worker.py entry.py allocation.json '
                  'PREREGISTRATION.md SOURCE_AUDIT.md FORMAL_ENTRY.md dev_summary.json dev_quality.json dev_visual_review.json').split())
    receipt=dict(frozen_at='2026-11-26',stage='FORMAL_CAL_THEN_TEST',scientific_pass=None,
                 selected_gains=dict(low=1.025,near=1.03,high=1.04),
                 sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)})
    (P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    names.add('strength_compare/receipt.json')
    buf=io.BytesIO()
    with ZipFile(buf,'w') as z:
        for n in sorted(names):
            info=ZipInfo(n,(2026,11,26,0,0,0));info.compress_type=ZIP_DEFLATED;z.writestr(info,(ROOT/n).read_bytes())
    payload=base64.b64encode(buf.getvalue()).decode()
    desc='''# LaWa 强度夹逼：正式公平比较
选择 GPU 后点击“全部运行”，无需上传代码。保留同一 Google Drive 中已完成的 DEV 目录。

固定 **粗尺度 + 普通增强 1.025 / 1.030 / 1.040**，不重算 DEV。
按原冻结作品运行 **24 CAL → 封存并读回验证阈值 → 160 TEST**。
全部四臂共 **184 次生成、4416 次写入、4416 次混合、18032 次读取**。
唯一确认性主比较：粗尺度对高档普通增强，三人均分 PNG 的作品级成员恢复差异。

预计准备 5–20 分钟、模型 150–270 分钟，依 GPU 与 Drive 速度变化；不是硬时限。
准备、模型、结算分别记账，继承全部历史实际与保守费用。权重及环境仅在 /content，优先复用有效缓存。
结果保存至 MyDrive/Model-WM/lawa-strength-bracket-compare-20261126-attempt01/。
原 DEV、旧结果和全部失败保留。已有正式目录（包括失败）再次打开只读，不追加运行。

完整结果可能为 COARSE_GAIN_PERSISTS_PENDING_VISUAL、ORDINARY_CATCHUP_IN_THIS_SAMPLE、INCONCLUSIVE
或 BRACKET_NOT_ESTABLISHED。scientific_pass=null、visual_review=PENDING，运行后返回完整回执接受核验。
若出现 FORMAL_STOPPED_INCOMPLETE，请保留日志；不要删除目录、换 seed 或再次试跑。
'''
    runner='''#@title 一次运行 CAL → 封存阈值 → TEST / 读取已有回执
import base64,io,json,pathlib,resource,sys,tempfile,time,zipfile
started=time.monotonic()
usage=resource.getrusage(resource.RUSAGE_SELF)
root=pathlib.Path(tempfile.mkdtemp(dir='/content',prefix='strength-bracket-formal-'))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(root)
for key in list(sys.modules):
    if key.split('.')[0] in ('strength_compare','strength_bracket','sd2_matched','matched_loss','lawa_ab','probe_selection','wam_method','wam_qualification','latent_control'):del sys.modules[key]
sys.path.insert(0,str(root))
from strength_compare.entry import run_colab
result=run_colab(started=started,cpu_start=usage.ru_utime+usage.ru_stime)
print(json.dumps(result,ensure_ascii=False,indent=2))
'''.replace('PAYLOAD',repr(payload))
    cells=[]
    for kind,source in [('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',desc),('code',runner)]:
        c=dict(id='strength-formal-'+str(len(cells)),cell_type=kind,metadata={},source=source.splitlines(True))
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    cells[-1]['metadata']['cellView']='form'
    name='lawa_strength_bracket_compare_colab.ipynb'
    nb=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3'),colab=dict(name=name)))
    (ROOT/'notebooks'/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    print(name,len(names),'embedded files',len(buf.getvalue()),'bytes')

if __name__=='__main__':build()
