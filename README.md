# Model-WM

AquaLoRA 的 44 轨迹 Colab 资格检查。仅检验加载、原生消息读出、同作品质量和成本；不训练，不执行正式 CAL/TEST，不宣称抗串谋或创新通过。

用户于 2026-11-08 接受“新内容未被所选 checkpoint 使用”的工作假设。实际训练、选择和评估使用史仍未核实；本入口不再以此阻断运行。

[在 Colab 打开](https://colab.research.google.com/github/RICHAAARC/Model-WM/blob/main/notebooks/aqualora_qualification_colab.ipynb)

选择GPU运行时，然后全部运行。第一格挂载Drive；后续代码从本仓库取得。默认结果目录为：

`MyDrive/Model-WM/aqualora-qualification-20261108-attempt01/`

每条轨迹的预先占位、状态、图像和提取结果及时落到Drive；最终报告位于 `report.md`，总状态位于 `summary.json`。发生失败、中断或预算停止时仍保留已完成内容。再次运行只读取已有attempt，不能清空目录来重跑；若需要修复后新的执行，需另行明确范围。

全部44条及数值门完成后，仍需检查 `visual_review/` 中全部8组的主体、构图和明显重影；未审查时为UNRESOLVED。不会自动宣布资格通过。完整基线仍需独立决定，吞吐估计不构成其启动授权。

固定输入见 [manifest.json](qualification/manifest.json)，此次假设和执行方式修订见 [AMENDMENT.md](AMENDMENT.md)。模型参数、解码预处理、PNG/JPEG和质量门来自此前已采用的44轨迹方案；没有为Colab降低步数、精度、分辨率或门槛。

本仓库的本地验证是输入/计数/媒体/预算控制和notebook格式检查，**尚未在Colab加载真实权重或生成**。真实依赖安装、CUDA兼容、权重接口、内存和吞吐均由唯一一次资格检查记录；出现问题停止，不暗中换版本或重跑。运行时日志含最终依赖版本和代码commit。

参考实现：[AquaLoRA](https://github.com/Georgefwt/AquaLoRA/tree/62fb5cf186050dd4e3bce0bfe15068872fc57d62)、[ppft权重](https://huggingface.co/georgefen/AquaLoRA-Models/tree/98688b2a1e762339593ee8fe96ed13762f06b732/ppft_trained)。原模型和基础SD1.5各自的许可继续适用。这里只下载推理所需文件，不重发布权重。
