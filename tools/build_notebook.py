"""Build the user-run, GPU-first notebook without executing it."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata.update(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                   accelerator="GPU", colab=dict(name="AquaLoRA 首次跑通", provenance=[]))
nb.cells = [
    nbf.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')"),
    nbf.v4.new_markdown_cell("""# AquaLoRA：首次跑通

选择 **GPU运行时**，再「全部运行」。本入口在同一个运行时完成安装、加载、生成、读取水印；不需要先装CPU环境再切GPU。

**软件安装包和模型文件持久保存在 `MyDrive/Model-WM/cache/aqualora-pinned-v1/`。** 换运行时可以从Drive重建。安装失败可继续；旧失败、费用和生成结果保留。下载10GiB改为记录参考，环境准备与模型执行分别计时；不再宣称满足旧资源预算。模型加载、生成和提取阶段仍上限30分钟，44条和全部科学参数不变。

结果写入原attempt下 `first-run/session-…/`。已启动的生成不会自动重复；环境故障不会消耗未启动的生成槽位。0/44时不要把环境停止当成模型失败。

数据使用史仍未核实，未接触新内容是用户接受的假设。只做DEV资格检查，不训练、不进入CAL/TEST。"""),
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
for name in list(sys.modules):
    if name.startswith("qualification."):
        del sys.modules[name]
import colab_bootstrap
run_first_run = importlib.reload(colab_bootstrap).run_first_run
'''),
    nbf.v4.new_markdown_cell("## 连续执行\n安装及模型文件先保存到Drive，随后执行冻结的44条轨迹。所有日志与状态都会保留。"),
    nbf.v4.new_code_cell('''RUN = run_first_run(started_unix=ENTRY_STARTED_UNIX, started_cpu=ENTRY_STARTED_CPU)
print("状态:", RUN["status"])
print("阶段:", RUN.get("execution_phase", "已有结果"))
print("轨迹:", RUN.get("trajectories_completed", 0), "/44 完成，", RUN.get("trajectories_started", 0), "/44 启动")
print("结果目录:", RUN["output"])
if RUN.get("error"):
    print("具体错误:", RUN["error"])
    print(RUN.get("error_detail", ""))
print("CAL/TEST：未启动")
'''),
    nbf.v4.new_markdown_cell("## 查看已保存结果\n生成后才展示质量审查图。数值完成后还需审查全部8组主体、布局和重影，程序不会自动宣布科学通过。"),
    nbf.v4.new_code_cell('''from IPython.display import display, Markdown, Image
output = pathlib.Path(RUN["output"])
report = output / "report.md"
if report.exists():
    display(Markdown(report.read_text()))
if RUN.get("trajectories_completed", 0):
    for panel in sorted((output / "visual_review").glob("*.png")):
        display(Markdown("**组 " + panel.stem + "**"))
        display(Image(filename=str(panel)))
else:
    print("尚无生成结果。请查看上面的执行阶段和具体错误；这不是水印性能结论。")
'''),
]
path = root/"notebooks/aqualora_qualification_colab.ipynb"
if path.exists():
    old = nbf.read(path, as_version=4)
    for a, b in zip(old.cells, nb.cells):
        b.id = a.id
nbf.validate(nb)
nbf.write(nb, path)
print(path)
