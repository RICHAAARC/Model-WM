"""Author the thin Colab notebook using nbformat; does not execute it."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
notebook = nbf.v4.new_notebook()
notebook.metadata.update(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                         colab=dict(name="AquaLoRA attempt01 环境恢复", provenance=[]))
notebook.cells = [
    nbf.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')"),
    nbf.v4.new_markdown_cell("""# AquaLoRA：attempt01 环境恢复

先在「运行时 → 更改运行时类型」选择 **CPU（硬件加速器：无）**，再「全部运行」。本版只恢复软件环境，不下载真实权重、不启动生成。代码从公开 GitHub 仓库获取。

原结果目录 `MyDrive/Model-WM/aqualora-qualification-20261108-attempt01` 的失败记录保持原样；恢复记录写入其中的 `environment-recovery-01/`。只接受已记录的 Python 安装失败、0/44轨迹状态，来源或费用不清即停止。

累计原已耗24.21792459487915秒、CPU核时2.950728999999999秒、下载289791730字节及其他资源。恢复耗时仍保守扣减原30分钟GPU时间额度，下载和CPU费用继续累加；不重置预算或轨迹。码本、seed、参数及44条上限保持不变。

环境通过后停在 **ENVIRONMENT_READY**，不表示资格检查通过。恢复失败、中断或重复点击均不会再次执行。正式CAL/TEST未启动，模型数据史仍未核实。"""),
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
recover_environment = importlib.reload(colab_bootstrap).recover_environment
'''),
    nbf.v4.new_markdown_cell("## 一次环境恢复\n下载安装官方同版本Python并检查锁定依赖。使用原剩余预算；不启动模型阶段。全部新日志即时写入Drive，已有恢复目录时只读。"),
    nbf.v4.new_code_cell('''RUN = recover_environment(started_unix=ENTRY_STARTED_UNIX, started_cpu=ENTRY_STARTED_CPU)
print("状态:", RUN["status"])
print("完成轨迹:", RUN.get("trajectories_completed", 0), "/ 44")
print("结果目录:", RUN["output"])
print("累计资源:", RUN.get("resources", "费用未完整记录，停止"))
print("环境就绪:", RUN["status"] == "ENVIRONMENT_READY")
print("真实权重及生成:", "本入口未启动")
print("正式 CAL/TEST:", "未启动")
'''),
    nbf.v4.new_markdown_cell("## 查看恢复记录\n此阶段不产生图像审查组。环境就绪后停止；原attempt01的失败报告、资源回执和轨迹记录保留。完整资格检查及正式CAL/TEST均未完成。"),
    nbf.v4.new_code_cell('''from IPython.display import display, Markdown
output = pathlib.Path(RUN["output"])
report = output / "report.md"
if report.exists():
    display(Markdown(report.read_text()))
print("环境恢复结束；本入口不进入模型阶段。")
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
