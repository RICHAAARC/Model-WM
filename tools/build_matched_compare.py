"""Assemble independent one-shot package without changing the completed DEV."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def build():
    files=set(json.loads((ROOT/'matched_loss/receipt.json').read_text())['sha256'])
    files.add('matched_loss/receipt.json')
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'matched_compare').glob('*') if p.is_file() and p.name!='receipt.json')
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(files)}
    (ROOT/'matched_compare/receipt.json').write_text(json.dumps(dict(sha256=hashes),indent=2)+'\n')
    import base64,io
    from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED
    data=io.BytesIO()
    with ZipFile(data,'w') as z:
        for f in sorted(files|{'matched_compare/receipt.json'}):
            info=ZipInfo(f,(2026,11,24,0,0,0));info.compress_type=ZIP_DEFLATED
            z.writestr(info,(ROOT/f).read_bytes())
    payload=base64.b64encode(data.getvalue()).decode()
    runner='''#@title 运行冻结公平比较，或只读取已有回执
import base64, io, json, pathlib, resource, sys, tempfile, time, zipfile
started=time.monotonic()
usage=resource.getrusage(resource.RUSAGE_SELF)
root=pathlib.Path(tempfile.mkdtemp(dir='/content',prefix='matched-compare-'))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(root)
for key in list(sys.modules):
    if key.split('.')[0] in ('matched_compare','matched_loss','lawa_ab','probe_selection','wam_method','wam_qualification','latent_control'):del sys.modules[key]
sys.path.insert(0,str(root))
from matched_compare.entry import run_colab
result=run_colab(started=started,cpu_start=usage.ru_utime+usage.ru_stime)
print(json.dumps(result,ensure_ascii=False,indent=2))
'''.replace('PAYLOAD',repr(payload))
    desc='''# LaWa 等结构损失公平比较：24 CAL + 80 TEST
选择 GPU 后全部运行，无需上传。固定普通 gain=1.025 与 coarse=0.10，共享潜变量与身份。
104 次生成 / 1248 次写入 / 5200 次读出 / 1248 次混合；不重算 DEV。
CAL 24 件全完成并封存共同阈值后才生成 TEST 80 件。旧留出不解封。重开只读，无自动重跑。
承接 DEV 和全部历史费用；准备剩余 659.358 秒、模型剩余 5217.272 秒、结算剩余 59.723 秒。
权重与环境仅存 /content；Drive 保存结果和失败回执。换运行时可能需要冷启动，预算不会重置。
数值判据通过仍需原图复核，不代表低误报保证或跨底座结论。
'''
    cells=[]
    for kind,source in [('code',"from google.colab import drive\ndrive.mount('/content/drive')\n"),('markdown',desc),('code',runner)]:
        cell=dict(id=f'compare-{len(cells)}',cell_type=kind,metadata={},source=source.splitlines(True))
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    cells[-1]['metadata']['cellView']='form'
    name='lawa_matched_compare_colab.ipynb'
    doc=dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3'),colab=dict(name=name)))
    (ROOT/'notebooks'/name).write_text(json.dumps(doc,ensure_ascii=False,indent=1)+'\n')

if __name__=='__main__':build()
