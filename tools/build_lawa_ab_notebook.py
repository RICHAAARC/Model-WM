import base64
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]


def build():
    groups={'lawa_ab':['__init__.py','assets.py','entry.py','model.py','science.py','worker.py','freeze.json','schema.json','prior_receipt.json','PREREGISTRATION.md'],
            'latent_control':['__init__.py','io.py'],
            'wam_method':['__init__.py','entry.py','protocol.py','io.py'],
            'wam_qualification':['__init__.py','setup.py','prepare.py','storage.py','transport.py','environment.py','metered.py','core.py','requirements-colab.txt']}
    data=io.BytesIO()
    with ZipFile(data,'w') as archive:
        for pkg,names in groups.items():
            for n in names:
                name=pkg+'/'+n
                info=ZipInfo(name,(2026,11,19,0,0,0))
                info.compress_type=ZIP_DEFLATED
                archive.writestr(info,(ROOT/name).read_bytes())
    payload=base64.b64encode(data.getvalue()).decode()
    runner="""#@title 从固定清单执行一次 LaWa A+B
import base64, io, json, pathlib, resource, sys, tempfile, time, zipfile
start = time.monotonic()
usage = resource.getrusage(resource.RUSAGE_SELF)
root = pathlib.Path(tempfile.mkdtemp(dir='/content', prefix='lawa-package-'))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(root)
for key in list(sys.modules):
    if key.split('.')[0] in ('lawa_ab','latent_control','wam_method','wam_qualification'):
        del sys.modules[key]
sys.path.insert(0,str(root))
from lawa_ab.entry import run_colab
result = run_colab(started=start,cpu_start=usage.ru_utime+usage.ru_stime)
print(json.dumps(result,ensure_ascii=False,indent=2))
panel = pathlib.Path(result['output'])/'A_panel.png'
if panel.is_file():
    from IPython.display import Image, display
    display(Image(filename=str(panel)))
""".replace('PAYLOAD',repr(payload))
    desc="""# LaWa 同作品副本 · A+B
选择 GPU 后全部运行，挂载保有上一轮记录的 Google Drive。代码与冻结清单已内嵌。
前两件属于拟合八件。拟合决策封存后才采集八件留出，结果单独打包。作品清单及运行数据位于
MyDrive/Model-WM/lawa-ab-20261119-attempt01/。查看 A_acceptance.json、A_panel.png、
fit_decision.json 和 summary.json。模型文件只在 /content。再次运行读取已保存状态。
"""
    def cell(t,s,i):
        c=dict(cell_type=t,id=f'lawa-ab-{i}',metadata={},source=s.splitlines(True))
        if t=='code':
            c.update(outputs=[],execution_count=None)
        return c
    cells=[cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n",0),
           cell('markdown',desc,1),cell('code',runner,2)]
    cells[2]['metadata']['cellView']='form'
    target=ROOT/'notebooks/lawa_ab_colab.ipynb'
    target.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3'),colab=dict(name='LaWa A+B'))),ensure_ascii=False,indent=1)+'\n')
    return target


if __name__=='__main__':
    print(build())
