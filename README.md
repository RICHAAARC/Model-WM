# Model-WM

[在 Colab 打开：AquaLoRA 首次跑通](https://colab.research.google.com/github/RICHAAARC/Model-WM/blob/main/notebooks/aqualora_qualification_colab.ipynb)

选择单GPU运行时后全部运行。同一个运行时内安装软件、加载模型、生成、读取水印并保存结果；不再要求CPU装好再切GPU。

软件wheel、固定Python归档和冻结模型文件持续保存在 MyDrive/Model-WM/cache/aqualora-pinned-v1/。新运行时从这份缓存重建；成功安装的回执不再被当成软件文件本身。Python3.10.14用内置venv离线建环境，避免uv再次隐式下载Python。

本次按用户“我需要先跑通”的指令修订工程资源政策：10GiB下载上限改为历史参考，不阻断首次生成；环境准备与模型阶段分别计时，原成本及本次成本全部保留。模型阶段上限30分钟，每次入口墙钟60分钟；CPU15核分钟、显存24GiB、RAM32GiB、结果1GiB的停止保护继续有效。不再声称满足最初包含全部准备成本的资源预算。

模型、float32、25步、分辨率、码本、seed、44条轨迹、媒体处理和质量门不变。新增结果在原attempt01下 first-run/session-…/；旧失败记录不改。安装/下载故障可恢复，短暂网络故障最多3次传输尝试并保留日志。任何已有STARTED/FAILED/COMPLETED生成不自动重做。

报告明确显示 SOFTWARE_SETUP、MODEL_DOWNLOAD、MODEL_EXECUTION，以及具体底层报错。没有图像时不提示质量审查失败；生成完成后仍需查看全部8组主体、布局和重影，不自动宣布资格通过。

数据使用史仍未核实，未接触新内容是用户接受的假设。这里只做DEV资格检查，不训练、不进入CAL/TEST、不宣称抗串谋或创新成立。

[输入](qualification/manifest.json) · [修订记录](AMENDMENT.md) · [工程检查](LOCAL_VALIDATION.md)

参考：[官方 AquaLoRA](https://github.com/Georgefwt/AquaLoRA/tree/62fb5cf186050dd4e3bce0bfe15068872fc57d62) 与 [官方 ppft 权重](https://huggingface.co/georgefen/AquaLoRA-Models/tree/98688b2a1e762339593ee8fe96ed13762f06b732/ppft_trained)。权重不随代码重发布，原许可继续适用。
