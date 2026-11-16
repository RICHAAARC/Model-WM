"""Build the explicitly authorized attempt02; never execute its GPU cell."""
import ast
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def build():
    nb = json.loads((ROOT/'notebooks/latent_control_colab.ipynb').read_text())
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for package in ['latent_control', 'wam_qualification', 'wam_method']:
            for p in sorted((ROOT/package).iterdir()):
                if not p.is_file() or p.name.startswith(('test_', 'check_')):continue
                if package == 'wam_method' and p.name not in ['__init__.py', 'entry.py', 'io.py', 'protocol.py']:continue
                if p.suffix not in ['.py', '.json', '.txt', '.md', '.npz']:continue
                item = zipfile.ZipInfo(package+'/'+p.name, date_time=(2026,11,16,0,0,0))
                item.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(item, p.read_bytes())
    code = ''.join(nb['cells'][2]['source'])
    lines = code.splitlines(True)
    lines = [('_payload = '+repr(base64.b64encode(stream.getvalue()).decode())+'\n')
             if line.startswith('_payload = ') else line for line in lines]
    code = ''.join(lines).replace('from latent_control.entry import run_colab',
        'from latent_control.rerun import run_colab').replace('启动这一次冻结实验', '清理旧输出并启动一次大显存 GPU 重跑')
    compile(code, 'rerun notebook', 'exec')
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = '''# 大显存 GPU 重跑：attempt02

选择 **A100（优先）或其他显存超过16 GiB的 GPU**，然后“全部运行”。无需上传文件。
首次运行会先读取旧累计费用，再删除 `MyDrive/Model-WM/latent-control-20261116-attempt01/`。
只保留累计费用与操作数量，不保留该次失败堆栈、图像或日志；其他历史实验目录不删除。
GPU不符合要求时，在清理、下载和模型启动前停止。删除由你运行此 notebook 时执行。

新输出：`MyDrive/Model-WM/latent-control-20261116-attempt02/`。
这次授权一个新的完整执行预算：准备30分钟、模型6小时（含60秒封存预留）、CPU12核小时、RAM32GiB、结果10GiB。
显存上限改为当前单卡实际容量；旧费用不退还。重复点击只读，不自动产生第三次运行。
旧 `attempt01/summary.json` 或已经写好的 `attempt02-carry.json` 必须在原路径，无需手工操作。

作品、码本、seed=2026111601、512px/float32、25步、B/C机制、136作品与所有操作上限保持冻结。
DEV/000在旧运行中已生成；CAL/TEST未进入。两个CAL阈值封存后才进入TEST。
本轮仍须满足原来的质量、拒识、配对增益及完整成本通过条件。
权重只存 `/content`；同一运行时尽量复用包和权重，更换运行时后重新检查环境与缓存。

本入口没有修复梯度图的显存占用，而是执行你授权的更大显卡重跑。
CPU入口检查不等于真实GPU验证；更大显卡也不保证数值通过或全部预算内完成。
[冻结方法与通过条件](https://github.com/RICHAAARC/Model-WM/blob/main/latent_control/PREREGISTRATION.md)
'''.splitlines(True)
    nb['metadata']['colab']['name'] = 'Latent control — larger GPU rerun attempt02'
    nb['metadata']['colab']['gpuType'] = 'A100'
    path = ROOT/'notebooks/latent_control_rerun_colab.ipynb'
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n')
    print(path)

if __name__ == '__main__':build()
