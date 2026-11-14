> **2026-11-14 WAM engineering closure:** the Colab link now targets attempt02 invocation 0005, retaining all four failures and costs. One user-started execution, no extra trial. See [scope and additive preparation budget](wam_qualification/ENGINEERING_CLOSURE_01.md).

# Model-WM

## WAM 资格入口（2026-11-14）

[在 Colab 打开：WAM 一次性资格验证](https://colab.research.google.com/github/RICHAAARC/Model-WM/blob/main/notebooks/wam_qualification_colab.ipynb)

选择单 GPU 后全部运行，首格仅挂载 Drive，无需上传代码。冻结16个作品、64次写入、160次提取；承接全部费用。准备预算追加1560秒墙钟／660 CPU核秒，累计上限2640／1260；模型仍900秒GPU代理计时／1800秒墙钟／600 CPU核秒，原模型费用继续扣账。旧44条 AquaLoRA与WAM四次失败记录保留；本次领取后再次打开只读。新权重与环境仅放/content，Drive只写封存结果和回执。

CPU回归及完整合成接口检查通过，覆盖真实worker、生成／写入／提取、媒体和评分链路。此前运行已加载过WAM真实权重，但完整真实链路未通过，本修订未加载真实权重或运行GPU；本次未新增Gemini审计，历史审计不代表本版本通过。没有训练或CAL/TEST，不作数值资格、抗串谋或创新PASS；未知训练史仍按用户接受的未接触新内容假设披露。

[预注册](wam_qualification/PREREGISTRATION.md) · [冻结输入](wam_qualification/manifest.json) · [Gemini审稿](wam_qualification/review/gemini_final.txt) · [使用说明](wam_qualification/README.md)

## 历史 AquaLoRA 入口

以下保留原第2轮入口与修订说明；该轮44条已完成，数值资格门失败，不是WAM的新运行入口。

[在 Colab 打开：AquaLoRA 完整重跑（第2轮）](https://colab.research.google.com/github/RICHAAARC/Model-WM/blob/main/notebooks/aqualora_qualification_colab.ipynb)

选择单GPU运行时后全部运行。同一个运行时内安装软件、加载模型、生成、读取水印并保存结果；不再要求CPU装好再切GPU。

软件wheel、固定Python归档和冻结模型文件持续保存在 MyDrive/Model-WM/cache/aqualora-pinned-v1/。新运行时从这份缓存重建；成功安装的回执不再被当成软件文件本身。Python3.10.14用内置venv离线建环境，避免uv再次隐式下载Python。

本次按用户“我需要先跑通”的指令修订工程资源政策：10GiB下载上限改为历史参考，不阻断首次生成；环境准备与模型阶段分别计时，原成本及本次成本全部保留。模型阶段上限30分钟，每次入口墙钟60分钟；CPU核时只记录、不触发15核分钟硬停止；显存24GiB、RAM32GiB、结果1GiB的停止保护继续有效。不再声称满足最初包含全部准备成本的资源预算。

模型、float32、25步、分辨率、码本、seed、44条轨迹、媒体处理和质量门不变。用户已授权整轮重跑并清理旧产物。新增结果在原attempt01下 rerun-02/session-…/；旧轮停止摘要与账本保留，旧图像、审查图和指标产物已清理。安装/下载故障可恢复，短暂网络故障最多3次传输尝试并保留日志。第二轮使用独立账本重新运行44条，不读取旧轮指标。本轮已有STARTED/FAILED/COMPLETED生成不自动重做；再次点击不会意外开启第三轮。

报告明确显示 SOFTWARE_SETUP、MODEL_DOWNLOAD、MODEL_EXECUTION，以及具体底层报错。没有图像时不提示质量审查失败；生成完成后仍需查看全部8组主体、布局和重影，不自动宣布资格通过。

数据使用史仍未核实，未接触新内容是用户接受的假设。这里只做DEV资格检查，不训练、不进入CAL/TEST、不宣称抗串谋或创新成立。

[输入](qualification/manifest.json) · [修订记录](AMENDMENT.md) · [工程检查](LOCAL_VALIDATION.md)

参考：[官方 AquaLoRA](https://github.com/Georgefwt/AquaLoRA/tree/62fb5cf186050dd4e3bce0bfe15068872fc57d62) 与 [官方 ppft 权重](https://huggingface.co/georgefen/AquaLoRA-Models/tree/98688b2a1e762339593ee8fe96ed13762f06b732/ppft_trained)。权重不随代码重发布，原许可继续适用。
