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
result = run_colab(started_unix=_entry_started, started_cpu=_entry_cpu, engineering_closure=True)
print(json.dumps(result, ensure_ascii=False, indent=2))
'''.replace('PAYLOAD',repr(payload))
def cell(kind,source):
    out=dict(cell_type=kind,metadata={},source=source.splitlines(True))
    if kind=="code": out.update(execution_count=None,outputs=[])
    return out
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name="WAM qualification — attempt02 engineering closure 01"),
    kernelspec=dict(name="python3",display_name="Python 3"),accelerator="GPU"),cells=[
    cell("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),
    cell("markdown", """# WAM qualification — attempt02 engineering closure 01

本入口承接同一 attempt02 的失败 0001–0004，下一次记录为 **0005**。领取本次标记后，无论成功或失败，再次点击只读；不自动追加运行。

**选择 GPU 后全部运行，无需上传代码。** 第一格仅挂载 Drive；GPU 由你点击启动。尽量保留当前运行时以复用环境和缓存，换运行时后会实际验证并恢复。

准备预算本次追加 **1560 秒墙钟、660 CPU 核秒**，累计准备上限为 **2640 秒／1260 CPU 核秒**。依据已完成的准备耗时 494.82 秒和历史 CPU 199.42 核秒，按三倍加60秒再向上取整到分钟；详细依据随记录封存。原预算文件与历史扣账不改写。已有准备扣账1082.184秒／412.111核秒，追加后剩1557.816秒／847.889核秒，进入本格后的耗时仍扣除。

模型预算仍900秒GPU代理计时／1800秒墙钟／600 CPU核秒，承接原模型扣账；全局累计上限不变。安装、复制、下载、封存均计费。停止余量60秒包含清理与最终封存，费用不足立即停止；不能保证任意冷启动都完成。

官方小型元数据与组件依赖先验证，再安装环境和恢复大权重。兼容环境复用；缓存命中不等于免安装、免复制。环境、实时通信及新权重仅放 /content；Drive 保存封存回执、结果和防重试标记，旧缓存只读，不新增权重归档。

作品、码本、seed、参数与 **16生成／64写入／160提取**保持冻结。第一个作品仍按1生成→4写入→10提取执行并计入总数，然后按现有成本预测继续或停止，无额外试跑。不训练，不进入CAL/TEST。

累计历史费用全部保留：当前总墙钟4411.522秒，CPU实测下界2765.070核秒；保守CPU扣账3002.255核秒。未知历史尾部继续标注，准备预算追加不代表旧预算通过。

输出：`MyDrive/Model-WM/wam-qualification-20261112-attempt02/`，新增日志在 `invocations/0005/`。请保留全部失败回执。

CPU回归与完整合成接口检查通过：实际worker、生成、写入、提取、PNG/JPEG、混合、评分和账本均覆盖。**未加载真实权重、未启动GPU；CPU通过不代表真实链路或资格通过。本次未新增Gemini审计。** 数值门槛与16组视觉审查仍按冻结方案判定。
"""),
    cell("code",code)])
for i,c in enumerate(notebook['cells']): c['id']=f'wam-{i:02d}'
destination=ROOT/"notebooks/wam_qualification_colab.ipynb"
destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
print(destination)
