"""Build an upload-free notebook with the exact frozen sources embedded."""
import base64
import io
import json
from pathlib import Path
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]

def build():
    buf=io.BytesIO()
    packages=('terminal_solver','latent_control','wam_method','wam_qualification')
    with ZipFile(buf,'w') as archive:
        for pkg in packages:
            for p in sorted((ROOT/pkg).iterdir()):
                if not p.is_file() or p.name.startswith(('test_','check_')):
                    continue
                if p.suffix not in ('.py','.json','.md','.npz','.txt'):
                    continue
                if pkg=='wam_method' and p.name not in ('__init__.py','protocol.py','entry.py','io.py'):
                    continue
                info=ZipInfo(pkg+'/'+p.name,(2026,11,19,0,0,0))
                info.compress_type=ZIP_DEFLATED
                archive.writestr(info,p.read_bytes())
    payload=base64.b64encode(buf.getvalue()).decode()
    runner="""#@title 读取封存进度，执行当前允许的一次阶段
import base64, io, json, pathlib, resource, sys, tempfile, time, zipfile
_started = time.monotonic()
_r = resource.getrusage(resource.RUSAGE_SELF)
_cpu = _r.ru_utime + _r.ru_stime
_dest = pathlib.Path(tempfile.mkdtemp(prefix="solver-release-", dir="/content"))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as _archive:
    for _name in _archive.namelist():
        if _dest not in (_dest / _name).resolve().parents:
            raise ValueError("Invalid package path")
    _archive.extractall(_dest)
for _name in list(sys.modules):
    if _name.split(".")[0] in ("terminal_solver","latent_control","wam_method","wam_qualification"):
        del sys.modules[_name]
sys.path.insert(0,str(_dest))
from terminal_solver.entry import run_colab
result = run_colab(started=_started,cpu_start=_cpu)
print(json.dumps(result,ensure_ascii=False,indent=2))
""".replace("PAYLOAD",repr(payload))
    desc="""# 同一共享中心的末端求解器对照

选择 GPU 后全部运行即可启动 DEV。首次运行须能读取历史封存的
`MyDrive/Model-WM/latent-control-20261116-attempt02/summary.json`。代码、码本、划分已随本页封装。
新结果保存至 `MyDrive/Model-WM/terminal-solver-20261119-attempt01/`；模型和环境保存在 `/content`。

本入口使用生成前修订的客观代理方案，无需组织或上传真人评分。
DEV 的 8 件作品在两臂各跑三档；只有布局差异与平坦区粗糙度代理满足选档规则、费用投影通过，才进入 CAL。
两个 CAL 阈值封存并读回之后才生成 TEST。任何已启动的作品不可重跑。
检查 formal/paired_results.json 的追踪、误指认与代理质量三项，以及 formal/gallery.html、figure_selection.json 的系统对比图。
将 formal 目录保留相对结构下载后可离线打开图库，点击看原图。按代理通过不等于真人质量等效。
所有技术中断均占用原次数。只续接完整封存阶段。预注册约束为准备累计 900 秒、模型累计 10,800 秒、总 CPU 14,400 核秒。
"""
    def cell(kind,source,n):
        item=dict(cell_type=kind,id='terminal-solver-'+str(n),metadata={},source=source.splitlines(True))
        if kind=='code':
            item.update(execution_count=None,outputs=[])
        return item
    disconnect="""#@title 阶段结束后释放 Colab 运行时
if result.get("status") != "GPU_REQUIRED":
    from google.colab import runtime
    print("回执已输出；正在释放运行时。如自动释放不可用，请手动断开并删除运行时。", flush=True)
    runtime.unassign()
"""
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3'),
            colab=dict(name='Terminal solver paired trial'),accelerator='GPU'),
            cells=[cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n",0),
                   cell('markdown',desc,1),cell('code',runner,2),cell('code',disconnect,3)])
    nb['cells'][2]['metadata']['cellView']='form'
    path=ROOT/'notebooks/terminal_solver_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    return path

if __name__=='__main__':
    print(build())
