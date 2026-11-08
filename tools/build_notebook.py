"""Author the thin Colab notebook using nbformat; does not execute it."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
notebook = nbf.v4.new_notebook()
notebook.metadata.update(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                         accelerator="GPU", colab=dict(name="AquaLoRA attempt01 模型阶段续接", provenance=[]))
notebook.cells = [
    nbf.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')"),
    nbf.v4.new_markdown_cell("""# AquaLoRA：attempt01 模型阶段续接

由你选择 **单GPU运行时** 并点击「全部运行」。代码从公开GitHub仓库获取；本入口从已完成的ENVIRONMENT_READY接续一次44轨迹资格检查，不进入正式CAL/TEST。

原结果目录 `MyDrive/Model-WM/aqualora-qualification-20261108-attempt01` 及 `environment-recovery-01/` 全部记录保留。新日志、轨迹与结果写入同一attempt下的 **model-stage-01/**；这不是新实验attempt。只接受最新、费用完整且0/44的ENVIRONMENT_READY回执，任何既有模型执行不重跑。

承接累计192.58095860481262秒、CPU核时129.291007秒、下载3465446284字节及其他已记费用。所有新安装、下载、CPU工作和生成继续消耗原预算；码本、seed、实验参数和44条上限不变。优先复用原恢复运行时的环境与已付费缓存，复用文件不重复计下载费用。

**切换GPU可能丢失本地环境。** 先查官方文件大小再决定是否下载；若缓存全部丢失，当前估计软件+模型约7.46GiB，超过剩余有效下载额度6.27GiB，将在大文件下载前停止。不会重置费用、换模型或缩减实验来绕过预算。

此段开始后失败、中断、超预算或重复点击均不会再次执行。数值完成后仍需审查全部8组图像，资格通过不自动宣布；模型数据史仍未核实，未接触新内容仍是用户接受的假设。"""),
    nbf.v4.new_code_cell('''import importlib, pathlib, resource, subprocess, sys, time

ENTRY_STARTED_UNIX = time.time()
own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
ENTRY_STARTED_CPU = own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime
REPOSITORY = "https://github.com/RICHAAARC/Model-WM.git"
SOURCE_BRANCH = "main"
checkout = pathlib.Path("/content/model-wm-code")
if not checkout.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", SOURCE_BRANCH,
                    REPOSITORY, str(checkout)], check=True)
else:
    subprocess.run(["git", "-C", str(checkout), "fetch", "origin", SOURCE_BRANCH], check=True)
    subprocess.run(["git", "-C", str(checkout), "merge", "--ff-only", "FETCH_HEAD"], check=True)
resolved_commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
print("本次代码版本:", resolved_commit)
if str(checkout) not in sys.path:
    sys.path.insert(0, str(checkout))
import colab_bootstrap
run_model_stage = importlib.reload(colab_bootstrap).run_model_stage
'''),
    nbf.v4.new_markdown_cell("## 一次模型阶段续接\n核对回执与环境，预检下载预算，随后下载冻结模型并运行资格检查。所有日志与已完成轨迹即时写入Drive。已有model-stage-01时只读取，不执行第二次。"),
    nbf.v4.new_code_cell('''RUN = run_model_stage(started_unix=ENTRY_STARTED_UNIX, started_cpu=ENTRY_STARTED_CPU)
print("状态:", RUN["status"])
print("完成轨迹:", RUN.get("trajectories_completed", 0), "/ 44")
print("结果目录:", RUN["output"])
print("累计资源:", RUN.get("resources", "费用未完整记录，停止"))
print("启动轨迹:", RUN.get("trajectories_started", "未核清"), "/ 44")
print("正式 CAL/TEST:", "未启动")
'''),
    nbf.v4.new_markdown_cell("## 查看全量结果\n报告、失败、全部配对读出和轨迹状态保存在model-stage-01。展示全部审查图，少于8组即未完成；逐组核查主体、布局和明显重影，未经审查的质量门保持UNRESOLVED。正式CAL/TEST未启动。"),
    nbf.v4.new_code_cell('''from IPython.display import display, Markdown, Image
output = pathlib.Path(RUN["output"])
report = output / "report.md"
if report.exists():
    display(Markdown(report.read_text()))
panels = sorted((output / "visual_review").glob("*.png"))
print("全部审查组:", len(panels), "/ 8")
for panel in panels:
    display(Markdown("**组 " + panel.stem + "**"))
    display(Image(filename=str(panel)))
'''),
]
path = root/"notebooks/aqualora_qualification_colab.ipynb"
path.parent.mkdir(exist_ok=True)
# Preserve cell identities when updating this existing notebook.
if path.exists():
    previous = nbf.read(path, as_version=4)
    if len(previous.cells) == len(notebook.cells):
        for old, new in zip(previous.cells, notebook.cells):
            new.id = old.id
nbf.validate(notebook)
nbf.write(notebook, path)
print(path)
