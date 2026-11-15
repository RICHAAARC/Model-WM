"""Build a frozen upload-free Colab entry; never execute the experiment."""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
stream=io.BytesIO()
with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
    for package in ['wam_method','wam_qualification']:
        for p in sorted((ROOT/package).iterdir()):
            if p.is_file() and not p.name.startswith(('test_','check_')) and p.suffix in ['.py','.json','.txt','.md']:
                info=zipfile.ZipInfo(package+'/'+p.name,date_time=(2026,11,15,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,p.read_bytes())
payload=base64.b64encode(stream.getvalue()).decode()
code='''#@title 启动本次冻结比较（再次点击只读，不追加运行）
import time, resource
_entry_started = time.monotonic()
_r = resource.getrusage(resource.RUSAGE_SELF)
_entry_cpu = _r.ru_utime + _r.ru_stime
import base64, io, json, pathlib, sys, tempfile, zipfile
_package = pathlib.Path(tempfile.mkdtemp(prefix="wam-writer-frozen-", dir="/content"))
_payload = PAYLOAD
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_payload))) as _z:
    for _name in _z.namelist():
        if _package not in (_package / _name).resolve().parents:
            raise RuntimeError("Unsafe embedded package")
    _z.extractall(_package)
for _name in list(sys.modules):
    if any(_name == p or _name.startswith(p + ".") for p in ["wam_method", "wam_qualification"]):
        del sys.modules[_name]
sys.path.insert(0, str(_package))
from wam_method.entry import run_colab
result = run_colab(started=_entry_started, cpu_start=_entry_cpu)
print(json.dumps(result, ensure_ascii=False, indent=2))
'''.replace('PAYLOAD',repr(payload))
description='''# WAM writer A/B/C — one frozen comparison

**选择 GPU 后全部运行；无需上传任何代码或权重。GPU 由你启动。**
第一格仅挂载 Drive。当前入口承接历史费用，创建新工作包
`MyDrive/Model-WM/wam-writer-abc-20261115-attempt01/`。
旧 `wam-qualification-20261112-attempt02/summary.json` 必须仍在原目录且已封存。

A 为原始 WAM；B 为普通混合训练；C 只增加一个任务级排序目标。
仅训练写入器，提取器/JND/写入强度冻结。B/C 各500次完整更新，
seed=2026110902；N16、K2、T4、128位跨作品载荷，PNG/JPEG85。
256 TRAIN图片、64 CAL组、128 TEST组已一次抽样冻结。
所有A/B/C校准阈值封存后才进入TEST，不选检查点、不调参、不追加实验。

这是此前审定的新方法比较，**不是重跑旧16/64/160资格小试**。
旧资格结果和失败记录不改写。本包不使用SD生成器，不进入VAE/MOLM迁移。

固定追加预算：GPU分配6小时、墙钟8小时、CPU6核小时、结果20GiB；
准备30分钟、训练每臂60分钟、评估180分钟，保留60秒封存余量，所有上限同时有效。
安装、下载、Drive写入和CPU工作均计入总费用。前16个正式更新计入500次，
仅用其耗时预测后续能否完成，预算不足即停止，不追加试跑。

权重、下载缓存、图片、环境及实时通信只放`/content`；旧付费缓存可只读复用。
Drive只保存封存日志/费用/结果及本次训练得到的B/C最终写入器。
运行时丢失后不能假定环境还在。源图片瞬时传输失败最多补取同一URL一次，
两次回执保留；来源不明、格式错误、资源不足或模型失败立即停止。
重复点击只读已保存记录，不重新领取训练或TEST次数。

通过条件：C比B净增至少13/128组联合PNG+JPEG catch-one，配对单侧p<.05；
联合错误归属数不高于B且95%上界<=5%；同时满足冻结的原生恢复、质量非劣和
计算开销<=5%条件。数值通过后仍需原分辨率视觉审查，不能自动声称创新成立。
0/128错误的95%上界约2.31%，不能据此声称1%低误报。

来源已做完整解码与精确重复检查；预训练重叠和近重复仍未经核实。
CPU合成流程与官方随机状态梯度接口已验证；**尚未加载真实权重或运行GPU，
不保证真实训练能在预算内完成。本次未新增Gemini审计。**

完整预注册：[PREREGISTRATION](https://github.com/RICHAAARC/Model-WM/blob/main/wam_method/PREREGISTRATION.md)。
检查输出中的`summary.json`、`calibration.json`、`method_result.json`、
`paired_results.json`以及每组原分辨率图像；全部失败与缺项均保留。
'''

def cell(kind,text,index):
    r=dict(cell_type=kind,id=f'wam-writer-{index:02d}',metadata={},source=text.splitlines(True))
    if kind=='code':r.update(execution_count=None,outputs=[])
    return r

nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(
    colab=dict(name='WAM writer A-B-C — frozen attempt01'),
    kernelspec=dict(name='python3',display_name='Python 3'),accelerator='GPU'),cells=[
    cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n",0),
    cell('markdown',description,1),cell('code',code,2)])
nb['cells'][2]['metadata']['cellView']='form'
target=ROOT/'notebooks/wam_writer_abc_colab.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print(target)
