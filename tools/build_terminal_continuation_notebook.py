"""Build the continuation from the published, self-contained DEV notebook."""
import ast
import base64
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def build():
    nb = json.loads((ROOT/'notebooks/terminal_solver_colab.ipynb').read_text())
    runner = ''.join(nb['cells'][2]['source'])
    call, = [n for n in ast.walk(ast.parse(runner)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode']
    original = ast.literal_eval(call.args[0])
    data = io.BytesIO(base64.b64decode(original))
    with ZipFile(data, 'a') as archive:
        for name in ('__init__.py','checkpoints.py','entry.py','worker.py'):
            path = ROOT/'terminal_continuation'/name
            info = ZipInfo(str(path.relative_to(ROOT)), (2026,11,19,0,0,0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    runner = runner.replace(original, base64.b64encode(data.getvalue()).decode())
    runner = runner.replace('from terminal_solver.entry import', 'from terminal_continuation.entry import')
    runner = runner.replace('("terminal_solver",', '("terminal_continuation","terminal_solver",')
    nb['metadata']['colab']['name'] = 'Terminal solver – continue from sealed DEV'
    nb['cells'][1]['source'] = """# 末端求解器对照 · 从已封存的 DEV 续接

选择 GPU，挂载原 Google Drive 后全部运行。请只在一个运行时打开本入口。
代码、码本、划分已内嵌，权重保存在 /content。入口读取
MyDrive/Model-WM/terminal-solver-20261119-attempt01 中的停止回执、8 件 DEV 的结果与选档。
准备、模型时间、CPU、显存、内存和结果体积继续记账。旧的总额度和预测不再触发叫停。

CAL 的 64 件作品完成后，两臂分别封存阈值并回读，再执行 64 件 TEST。
续接的回执、逐作品结果、paired_results.json 和原分辨率 gallery.html 均在原目录的 continuation-01/。
意外断线后重新打开本页。两臂已封存的作品直接复用。已开始但未封存的作品保留为缺失，
同阶段的未启动作品仍可执行；CAL 有缺失时阻止 TEST，TEST 有缺失时无法判定通过。
闭合实验再次点击只读结果。按冻结的追踪、误归属与客观图像代理报告比较结果，真人评审暂缓。
""".splitlines(True)
    nb['cells'][2]['source'] = runner.splitlines(True)
    nb['cells'][3]['source'] = ['if result.get("status") not in ("GPU_REQUIRED", "ACTIVE_IN_THIS_RUNTIME", "ACTIVE_IN_OTHER_RUNTIME"):\n',
                                  '    from google.colab import runtime\n', '    runtime.unassign()\n']
    target = ROOT/'notebooks/terminal_solver_continue_colab.ipynb'
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n')
    return target


if __name__ == '__main__':
    print(build())
