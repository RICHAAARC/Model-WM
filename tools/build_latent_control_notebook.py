"""Create upload-free frozen Colab; never execute model work."""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def build():
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for package in ['latent_control','wam_qualification','wam_method']:
            for p in sorted((ROOT/package).iterdir()):
                if not p.is_file() or p.name.startswith(('test_','check_')):continue
                if package=='wam_method' and p.name not in ['__init__.py','entry.py','io.py','protocol.py']:continue
                if p.suffix not in ['.py','.json','.txt','.md','.npz']:continue
                item=zipfile.ZipInfo(package+'/'+p.name,date_time=(2026,11,16,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(item,p.read_bytes())
    payload=base64.b64encode(stream.getvalue()).decode()
    code='''#@title 启动这一次冻结实验（重复点击只读）
import time, resource
_started = time.monotonic()
_r = resource.getrusage(resource.RUSAGE_SELF)
_cpu_start = _r.ru_utime + _r.ru_stime
import base64, io, json, pathlib, sys, tempfile, zipfile
_package = pathlib.Path(tempfile.mkdtemp(prefix="latent-control-frozen-", dir="/content"))
_payload = PAYLOAD
with zipfile.ZipFile(io.BytesIO(base64.b64decode(_payload))) as _zip:
    for _name in _zip.namelist():
        if _package not in (_package / _name).resolve().parents:
            raise RuntimeError("Unsafe embedded package path")
    _zip.extractall(_package)
for _name in list(sys.modules):
    if any(_name == p or _name.startswith(p + ".") for p in ["latent_control", "wam_method", "wam_qualification"]):
        del sys.modules[_name]
sys.path.insert(0, str(_package))
from latent_control.entry import run_colab
result = run_colab(started=_started, cpu_start=_cpu_start)
print(json.dumps(result, ensure_ascii=False, indent=2))
'''.replace('PAYLOAD',repr(payload))
    description='''# 生成内共享响应控制：一次冻结小试

**由你选择 GPU 后“全部运行”；无需上传代码、数据或权重。** 第一格仅挂载 Drive。
结果持续封存至 `MyDrive/Model-WM/latent-control-20261116-attempt01/`。
旧 `wam-writer-abc-20261115-attempt01/summary.json` 须保留在原目录且已封存。

## 方法与数据

同一作品共享一次扩散去噪，在冻结 VAE 的输入 latent 写入用户身份。
C：两次固定响应计算后共用逆映射；B：普通逐副本 L-BFGS，最多32次目标/梯度。
相同作品、码本、强度、画质门及反向上限；所有内部线搜索和失败都计费。
N16、K2、T1；64个可编码身份最多6 bit身份容量，不是64 bit独立载荷。

seed=2026111601；136个新作品：TRAIN=0、DEV8、CAL64、TEST64，已一次抽样冻结。
实际 uint8 RGB 以¼、½、¾混合，再评估PNG/JPEG85。外来原生副本属于正式拒识风险族。
两个CAL阈值封存并读回后才进入TEST，不换seed、码本、设置或追加次数。

## 费用与停止

准备最多30分钟；模型加载到全部结果封存最多6小时（内含60秒封存预留）；
总CPU12核小时、RAM32GiB、VRAM16GiB、结果10GiB。准备费用单列并保留完整总费用。
DEV8是真实分母的一部分，用其实际完整耗时投影剩余工作；超预算即停止，不补跑。
全部旧失败及累计费用承接。预训练权重、缓存、环境及实时通信仅在`/content`；
Drive保存封存回执、源代码、原分辨率结果和失败记录。可复用仍存在且通过检查的缓存。
运行时丢失后不会凭旧READY认定环境存在；重复点击只读，未封存中断保留完整预算扣账。

## 结果判定

C至少48/64作品通过所有混合/codec，较B净增≥8/64且配对单侧p≤.05；联合错误指认0/64；
同时满足冻结画质及完整成本不增条件。0/64风险的95%上界约4.57%，不是零风险。
数值通过仍需原分辨率视觉和机制证据复核，不自动宣布科学或创新PASS。

CPU接口、真实组件索引/随机初始化组件和合成控制流程可在本地验证；
**交付时未加载真实权重或启动GPU，不保证真实方法通过或在预算内完成。**
预训练内容史仍UNVERIFIED，未接触内容仅是用户接受的假设。本次未新增Gemini审计。

[完整预注册](https://github.com/RICHAAARC/Model-WM/blob/main/latent_control/PREREGISTRATION.md)
· [冻结清单](https://github.com/RICHAAARC/Model-WM/blob/main/latent_control/manifest.json)

检查 `summary.json`、`method_result.json`、`calibration.json`、`paired_results.json`，以及
`works/` 中全部原分辨率图像、写入轨迹、响应谱和误差；失败不会被清理或从分母剔除。
'''
    def cell(kind,text,index):
        r=dict(cell_type=kind,id=f'latent-control-{index:02d}',metadata={},source=text.splitlines(True))
        if kind=='code':r.update(execution_count=None,outputs=[])
        return r
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name='Generation-internal shared response — frozen attempt01'),
        kernelspec=dict(name='python3',display_name='Python 3'),accelerator='GPU'),cells=[
        cell('code',"from google.colab import drive\ndrive.mount('/content/drive')\n",0),
        cell('markdown',description,1),cell('code',code,2)])
    nb['cells'][2]['metadata']['cellView']='form'
    path=ROOT/'notebooks/latent_control_colab.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    print(path)

if __name__=='__main__':build()
