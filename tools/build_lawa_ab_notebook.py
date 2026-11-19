import base64
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]


def build(probe=False):
    groups={'lawa_ab':['__init__.py','assets.py','entry.py','model.py','science.py','worker.py','freeze.json','schema.json','prior_receipt.json','PREREGISTRATION.md'],
            'latent_control':['__init__.py','io.py'],
            'wam_method':['__init__.py','entry.py','protocol.py','io.py'],
            'wam_qualification':['__init__.py','setup.py','prepare.py','storage.py','transport.py','environment.py','metered.py','core.py','requirements-colab.txt']}
    if probe:
        groups['probe_selection']=['__init__.py','entry.py','owner.py','assets.py','carrier.py','rules.py','worker.py',
                                   'worker_paths.py','freeze.json','PREREGISTRATION.md']
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
    if probe:
        runner=runner.replace('一次 LaWa A+B','一次 LaWa 探针配对小试').replace(
            "('lawa_ab','latent_control'", "('probe_selection','lawa_ab','latent_control'").replace(
            'from lawa_ab.entry import run_colab','from probe_selection.entry import run_colab').split('panel = ')[0]
        desc="""# LaWa · 探针选择的跨身份验证
选择 GPU 后全部运行，挂载保存原 LaWa A+B 的 Drive。冻结代码与清单已内嵌。
使用旧前 8 件 DEV、新 24 件 CAL、新 8 件探索。32 次新生成，1208 次真实身份写入，
792 次 RGB 混合，3352 次提取。一次预约，失败保留，重开只读。
结果在 MyDrive/Model-WM/lawa-probe-selection-20261119-attempt01。查看 summary.json、
global.json、thresholds.json、paired_decision.json 和 exploration/。模型文件仅在 /content。
最后一格可以按作品和写法查看正式发行副本与混合图的原分辨率。"""
    cells=[cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n",0),
           cell('markdown',desc,1),cell('code',runner,2)]
    cells[2]['metadata']['cellView']='form'
    target=ROOT/'notebooks/lawa_ab_colab.ipynb'
    if probe:
        target=ROOT/'notebooks/lawa_probe_selection_colab.ipynb'
        gallery="""#@title 原分辨率视觉复核（仅本轮新探索作品）
from pathlib import Path
from zipfile import ZipFile
import json
from IPython.display import display, Image, clear_output
import ipywidgets as widgets
folder = Path(result['output'])
reports = sorted((folder/'exploration').glob('explore-*.json'))
if reports:
    work = widgets.Dropdown(options=[p.stem for p in reports],description='作品')
    arm = widgets.Dropdown(options=['margin','ordinary','global','native','confidence'],description='写法')
    view = widgets.Dropdown(options=['equal3.png','equal3.jpg','equal2.png','unequal3.png','single0.png','single1.png','single2.png'],description='副本/混合')
    group = widgets.Dropdown(options=['reg','foreign'],description='身份组')
    def show(work,arm,view,group):
        row=json.loads((folder/'exploration'/(work+'.json')).read_text())
        selected=row['choices'][arm]
        with ZipFile(folder/'archives'/(work+'.zip')) as archive:
            for name in dict.fromkeys(['clean.png','native/'+group+'/'+view,selected+'/'+group+'/'+view]):
                print(work,name)
                display(Image(data=archive.read(name)))
    display(work,arm,group,view,widgets.interactive_output(show,dict(work=work,arm=arm,group=group,view=view)))
"""
        cells.append(cell('code',gallery,3))
    target.write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        accelerator='GPU',kernelspec=dict(name='python3',display_name='Python 3'),colab=dict(name='LaWa probe selection' if probe else 'LaWa A+B'))),ensure_ascii=False,indent=1)+'\n')
    return target


if __name__=='__main__':
    import sys
    print(build('--probe' in sys.argv))
