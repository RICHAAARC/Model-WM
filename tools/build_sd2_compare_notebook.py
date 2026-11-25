"""Upload-free SD2 CAL/TEST release, without modifying the completed DEV."""
import base64,hashlib,io,json
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'sd2_compare'

def files():
    names=set(json.loads((ROOT/'sd2_matched/receipt.json').read_text())['sha256'])
    names.add('sd2_matched/receipt.json')
    names.update('matched_compare/'+n for n in ('__init__.py','worker.py','rules.py'))
    names.update('sd2_compare/'+n for n in (
        '__init__.py','entry.py','worker.py','gate.py','allocation.json','PREREGISTRATION.md',
        'SOURCE_AUDIT.md','dev_summary.json','dev_quality.json','dev_recomputed.json',
        'dev_visual_review.json','DEV_AUDIT_ZH.md'))
    return names

def build():
    names=files()
    receipt=dict(stage='CAL_TEST_AFTER_REVIEWED_DEV',frozen_at='2026-11-25',
        sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)})
    (P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    names.add('sd2_compare/receipt.json');buf=io.BytesIO()
    with ZipFile(buf,'w') as z:
        for n in sorted(names):
            info=ZipInfo(n,(2026,11,25,0,0,0));info.compress_type=ZIP_DEFLATED
            z.writestr(info,(ROOT/n).read_bytes())
    payload=base64.b64encode(buf.getvalue()).decode()
    desc='''# LaWa × SD2.1：24 CAL + 80 TEST 公平比较
选择 GPU 后“全部运行”，无需上传代码。使用已冻结 112 件中的剩余 104 件，不重算 DEV。
同一 Drive 中须保留已完成的 lawa-sd21-matched-dev-20261125-attempt01 文件夹；入口先核对 DEV 的质量选择、冻结划分和累计费用。

普通增益固定 **1.025**，粗尺度固定 **0.10**。模型构型仍是 SD2.1 去噪器 + 固定 SD1.4 干净解码器/LaWa。
执行 **104 次生成、1248 次写入、1248 次混合、5200 次读取**，无训练。
**24 件 CAL 全完成，阈值封存并读回核对后，才生成 80 件 TEST。** 主终点仍为三人均分 PNG 的作品级真成员增益；JPEG、非均分、误指认和质量全部保留。

DEV 实际准备 4.90 分钟、模型 7.13 分钟；正式比较预计准备 5–20 分钟、模型 70–120 分钟，取决于 GPU、网络和 Drive。
这些是估计，没有任意总时长截停。准备、模型、结算分别计时，并承接历史实际费用及保守扣账。
权重和软件仅存 /content，同一运行时优先复用经验证缓存；换运行时需重新准备。

结果保存到 MyDrive/Model-WM/lawa-sd21-matched-compare-20261125-attempt01/。
重复 Run all 只读取保存回执，包括失败，**不会自动重跑**。
完整执行应显示 104/1248/1248/5200 完成；结果可能为 NUMERIC_GATES_MET_PENDING_VISUAL_REVIEW 或 MATCHED_COMPARISON_NOT_SUPPORTED。
完成后返回完整回执。数值过门仍需图像复核；不提前宣称跨底座收益或低误报保证。
'''
    runner='''#@title 运行一次 CAL → 阈值封存 → TEST，或只读已有回执
import base64,io,json,pathlib,resource,sys,tempfile,time,zipfile
started=time.monotonic()
usage=resource.getrusage(resource.RUSAGE_SELF)
root=pathlib.Path(tempfile.mkdtemp(dir='/content',prefix='sd21-matched-compare-'))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:
    z.extractall(root)
for key in list(sys.modules):
    if key.split('.')[0] in ('sd2_compare','sd2_matched','matched_compare','matched_loss','lawa_ab','probe_selection','wam_method','wam_qualification','latent_control'):
        del sys.modules[key]
sys.path.insert(0,str(root))
from sd2_compare.entry import run_colab
result=run_colab(started=started,cpu_start=usage.ru_utime+usage.ru_stime)
print(json.dumps(result,ensure_ascii=False,indent=2))
'''.replace('PAYLOAD',repr(payload))
    cells=[]
    for kind,source in [('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',desc),('code',runner)]:
        c=dict(id=f'sd21-compare-{len(cells)}',cell_type=kind,metadata={},source=source.splitlines(True))
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    cells[-1]['metadata']['cellView']='form'
    name='lawa_sd21_matched_compare_colab.ipynb'
    nb=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',
        kernelspec=dict(name='python3',display_name='Python 3'),colab=dict(name=name)))
    (ROOT/'notebooks'/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    print(name,len(names),'embedded files')
if __name__=='__main__':build()
