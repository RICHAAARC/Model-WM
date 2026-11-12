"""Self-contained entry: source embedded, no moving Git branch at runtime."""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
stream=io.BytesIO()
with zipfile.ZipFile(stream,"w",zipfile.ZIP_DEFLATED) as z:
    for p in sorted((ROOT/"wam_qualification").iterdir()):
        if p.is_file() and not p.name.startswith(("test_","check_")) and p.suffix in (".py",".json",".txt",".md"):
            info=zipfile.ZipInfo("wam_qualification/"+p.name, date_time=(2026,11,9,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
payload=base64.b64encode(stream.getvalue()).decode()
code='''import time, resource
_entry_started = time.time()
_r, _c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
_entry_cpu = _r.ru_utime + _r.ru_stime
import base64, io, json, pathlib, sys, tempfile, zipfile
_package = pathlib.Path(tempfile.mkdtemp(prefix="wam-frozen-entry-", dir="/content"))
_payload = PAYLOAD
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_payload))) as _z:
    for _name in _z.namelist():
        _dst = (_package / _name).resolve()
        if _package not in _dst.parents:
            raise RuntimeError("Unsafe embedded package")
    _z.extractall(_package)
# A repeated cell must use this frozen package, not an earlier imported version.
for _name in list(sys.modules):
    if _name == "wam_qualification" or _name.startswith("wam_qualification."):
        del sys.modules[_name]
sys.path.insert(0, str(_package))
from wam_qualification.separated_entry import run_colab
result = run_colab(started_unix=_entry_started, started_cpu=_entry_cpu)
print(json.dumps(result, ensure_ascii=False, indent=2))
'''.replace('PAYLOAD',repr(payload))
def cell(kind,source):
    out=dict(cell_type=kind,metadata={},source=source.splitlines(True))
    if kind=="code": out.update(execution_count=None,outputs=[])
    return out
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name="WAM qualification — separate preparation and model budgets"),
    kernelspec=dict(name="python3",display_name="Python 3"),accelerator="GPU"),cells=[
    cell("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),
    cell("markdown","# WAM qualification — separate preparation and model budgets\n\n**外部审计状态：本次 Gemini 因 agy 配额耗尽未执行；用户已明确豁免本次门槛，依据 CPU 检查发布。不得视为 Gemini 审计通过。**\n\n沿用冻结的16个作品、码本、seed及16生成/64写入/160读取。旧 attempt01 封存为环境阶段停止，所有失败与费用继续保留；本次为明确授权的新执行 `wam-qualification-20261112-attempt02`，不进入训练或 CAL/TEST。\n\n**选择一个 GPU，再全部运行。GPU 由你启动，无需上传代码。** 第一格只挂载 Drive。准备阶段包括安装、复制和下载，限1080秒墙钟/600 CPU核秒；依据真实 Colab 已完成冷启动入口353.79秒，按3倍向上取整到分钟。模型阶段从真实权重加载开始，独立限900秒GPU计时/1800秒墙钟/600 CPU核秒；两阶段均保留原20秒停止余量，总费用另行累计，原跨执行总预算继续约束。\n\n历史 CPU 已从最终低值18.62核秒补到监控已观察的至少78.36核秒，累计至少2404.914067核秒。无法追溯的尾部明确保留未知标记；为总预算审核，旧 WAM CPU600核秒额度全部保留扣账，测量费用和保守预留分别展示，不伪造精确总数。\n\n**恢复范围：** 当前运行时有可用环境则实际校验后复用；安装中断后保留已装包，从固定安装包补齐缺项。准备阶段仅在费用已封存且剩余准备预算足够时可再次运行，次数和累计预算不重置。换运行时后 `/content` 环境和本地文件可能消失，必须重新检查和恢复；Drive历史缓存可读取，不能把旧 READY 当成当前可用。无法封存费用的硬中断会停止自动恢复。进入模型加载后禁止再次执行，即使尚未生成第一张图。\n\n真实权重和新增下载只放 `/content`；优先复用当前运行时文件，再读取已付费 Drive 缓存。Drive只写小型记录、结果与失败日志，不新增权重、不删除旧失败。每次准备调用的日志保存在 `invocations/`，旧记录不覆盖。\n\n输出：`MyDrive/Model-WM/wam-qualification-20261112-attempt02/`。查看 `history_correction.json`、`summary.json`中的 `phases`、`resources`、`cumulative_resources`和`cumulative_budget_debits`。CPU预检与代码审计不代表真实GPU资格通过；预算、来源、数值或质量失败仍会停止。\n"),
    cell("code",code)])
for i,c in enumerate(notebook['cells']): c['id']=f'wam-{i:02d}'
destination=ROOT/"notebooks/wam_qualification_colab.ipynb"
destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
print(destination)
