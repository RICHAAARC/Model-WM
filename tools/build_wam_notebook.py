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
result = run_colab(started_unix=_entry_started, started_cpu=_entry_cpu, technical_continuation=True)
print(json.dumps(result, ensure_ascii=False, indent=2))
'''.replace('PAYLOAD',repr(payload))
def cell(kind,source):
    out=dict(cell_type=kind,metadata={},source=source.splitlines(True))
    if kind=="code": out.update(execution_count=None,outputs=[])
    return out
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name="WAM qualification — attempt02 technical continuation 01"),
    kernelspec=dict(name="python3",display_name="Python 3"),accelerator="GPU"),cells=[
    cell("code","from google.colab import drive\ndrive.mount('/content/drive')\n"),
    cell("markdown","# WAM qualification — attempt02 technical continuation 01\n\n本入口仅执行用户授权的同一 attempt02 **一次技术续接**，保留失败 0001、0002、0003；下一次记录为 0004。领取续接标记后，无论准备或模型阶段是否失败，再次点击均只读。\n\n**GPU 仍由你启动，无需上传代码。** 第一格仅挂载 Drive；请选择 GPU 后全部运行。尽量保留现有运行时以复用环境和缓存；换运行时后会实际检查恢复，冷启动超剩余预算仍停止。\n\n已补齐固定官方版本的图像预处理配置，并在模型阶段前检查真实组件索引和配置。作品、码本、seed、参数以及16生成/64写入/160读取上限不变，无额外预热，不进入训练或 CAL/TEST。\n\n准备预算仍1080秒墙钟/600 CPU核秒；模型预算仍900秒GPU代理计时/1800秒墙钟/600 CPU核秒。保留20秒停止余量和跨执行总预算，**不重置额度**。\n\n三次失败的实测下界继续累计：attempt02 572.369144秒墙钟、188.376089 CPU核秒；含旧历史共3953.353626秒墙钟、至少2593.290156 CPU核秒，原保守CPU扣账2750.475211核秒。历史封存尾部没有实测终点，仍标注未知；从原预算额外保留准备40秒、模型20秒的墙钟/CPU费用，并计入GPU占用代理扣账，不冒充实测或新增预算。扣留后准备阶段在停止余量前剩475.984074秒墙钟、379.668502 CPU核秒。\n\n新入口把封存主体写盘纳入计时；最终回执自身写入预扣原20秒余量，返回前核验实际封存耗时，超出则保留费用并停止。查看 resources/phases 的测量值及 budget_resources/phase_budget_debits/cumulative_budget_debits 的保守扣账。\n\n实时通信、环境与真实权重仅在 /content；Drive 保存封存回执、结果和防重试标记，旧失败不删除。输出保持 MyDrive/Model-WM/wam-qualification-20261112-attempt02/，本次日志位于 invocations/0004/。\n\n56项CPU工程检查、真实组件索引下的完整合成加载及缺配置负例通过。**本修复未新增 Gemini 审计；CPU通过不代表真实GPU兼容或资格通过。** 实际质量、预算和视觉审查仍按冻结条件判定。\n"),
    cell("code",code)])
for i,c in enumerate(notebook['cells']): c['id']=f'wam-{i:02d}'
destination=ROOT/"notebooks/wam_qualification_colab.ipynb"
destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
print(destination)
