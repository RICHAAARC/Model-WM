"""Author the thin Colab notebook using nbformat; does not execute it."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
notebook = nbf.v4.new_notebook()
notebook.metadata.update(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                         accelerator="GPU", colab=dict(name="AquaLoRA 44轨迹资格检查", provenance=[]))
notebook.cells = [
    nbf.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')"),
    nbf.v4.new_markdown_cell("""# AquaLoRA：44轨迹资格检查

先在「运行时 → 更改运行时类型」选择 **GPU**，再「全部运行」。代码从公开 GitHub 仓库获取，结果持续保存到 `MyDrive/Model-WM/aqualora-qualification-20261108-attempt01`。

只做加载、原生水印读出、同作品质量和成本检查；最多44条轨迹、GPU占用30分钟。不会训练或进入CAL/TEST。**未知训练历史按用户接受的新内容假设处理，仍未核实。**

已有结果目录时只读取，不覆盖或重跑。安装/下载也计入预算；失败及未完成结果同样保留。数值检查完成后仍需审查全部8组图像。"""),
    nbf.v4.new_code_cell('''import pathlib, subprocess, sys, time

ENTRY_STARTED_UNIX = time.time()
REPOSITORY = "https://github.com/RICHAAARC/Model-WM.git"
SOURCE_BRANCH = "main"
checkout = pathlib.Path("/content/model-wm-code")
if not checkout.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", SOURCE_BRANCH,
                    REPOSITORY, str(checkout)], check=True)
resolved_commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
print("本次代码版本:", resolved_commit)
if str(checkout) not in sys.path:
    sys.path.insert(0, str(checkout))
from colab_bootstrap import run_colab
'''),
    nbf.v4.new_markdown_cell("## 运行并保存\n每条状态和结果直接写入Drive；不必等最后一格。此格可能运行约30分钟，达到预算会停止。"),
    nbf.v4.new_code_cell('''RUN = run_colab(started_unix=ENTRY_STARTED_UNIX)
print("状态:", RUN["status"])
print("完成轨迹:", RUN.get("trajectories_completed", 0), "/ 44")
print("结果目录:", RUN["output"])
print("正式 CAL/TEST:", "未启动")
'''),
    nbf.v4.new_markdown_cell("## 查看结果\n报告和全部原始记录已在Drive。下面展示全部已生成审查图；数量不足8组即未完成。对照左上Reference检查各副本与混图的主体、布局和明显重影，未经审查的质量门保持UNRESOLVED。"),
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
nbf.validate(notebook)
nbf.write(notebook, path)
print(path)
