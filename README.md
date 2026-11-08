# Model-WM

AquaLoRA 的 44 轨迹 Colab 资格检查。仅检验加载、原生消息读出、同作品质量和成本；不训练，不执行正式 CAL/TEST，不宣称抗串谋或创新通过。

用户于 2026-11-08 接受“新内容未被所选 checkpoint 使用”的工作假设。实际训练、选择和评估使用史仍未核实；本入口不再以此阻断运行。

[在 Colab 打开](https://colab.research.google.com/github/RICHAAARC/Model-WM/blob/main/notebooks/aqualora_qualification_colab.ipynb)

**当前 notebook 是 attempt01 的一次 CPU 环境恢复入口。选择 CPU（硬件加速器：无），然后全部运行。** 第一格只挂载Drive；后续代码从本仓库取得。原结果目录为：

`MyDrive/Model-WM/aqualora-qualification-20261108-attempt01/`

原目录的文件保持原样。恢复日志和累计资源写入其中的 `environment-recovery-01/`，该子目录不是新实验attempt。只接受已记录的Python安装失败和0/44状态；缺失记录、来源不清、已有轨迹或预算不足均停止。恢复目录先占位，失败、中断和重复点击不会再次执行。

同一官方 Astral CPython 3.10.14 / 20240814 构建改由 GitHub Release asset API 显式获取，随后让uv使用本地解释器；禁止uv自动下载解释器及HTTP重试，没有镜像或版本回退。完成锁定依赖安装、`pip check`、版本记录后停止在 **ENVIRONMENT_READY**。本入口不下载模型权重、不生成、不进入CAL/TEST；环境就绪不等于资格检查通过。

累计原墙钟/GPU时间额度24.21792459487915秒、CPU核时2.950728999999999秒、下载289791730字节及其他已记费用。新恢复耗时继续保守扣减原30分钟GPU时间额度，即使恢复在CPU上进行；CPU和新下载继续累加，RAM/设备峰值取跨段最大值，结果大小按整个attempt统计且不低于原回执。不重置码本、seed、参数、轨迹计数或预算。

全部44条及数值门完成后，仍需检查 `visual_review/` 中全部8组的主体、构图和明显重影；未审查时为UNRESOLVED。不会自动宣布资格通过。完整基线仍需独立决定，吞吐估计不构成其启动授权。

固定输入见 [manifest.json](qualification/manifest.json)，此次假设和执行方式修订见 [AMENDMENT.md](AMENDMENT.md)。模型参数、解码预处理、PNG/JPEG和质量门来自此前已采用的44轨迹方案；没有为Colab降低步数、精度、分辨率或门槛。

LoRA 张量接口修复保持不变。**31项本地CPU工程检查通过**，包括13项环境恢复检查、14项原控制检查和4项锁定版本加载回归；另已实际取得官方Python安装包、运行3.10.14并用锁定uv0.9.5创建独立环境。入口会更新已有代码checkout并重新载入入口模块。检查方法和证据边界见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。

**本次未执行Colab恢复、加载真实权重或启动GPU**。CPU检查不代表Colab网络、CUDA、真实权重、内存或吞吐已验证；出现问题停止并保留记录。运行时日志含最终依赖版本和代码commit。

参考实现：[AquaLoRA](https://github.com/Georgefwt/AquaLoRA/tree/62fb5cf186050dd4e3bce0bfe15068872fc57d62)、[ppft权重](https://huggingface.co/georgefen/AquaLoRA-Models/tree/98688b2a1e762339593ee8fe96ed13762f06b732/ppft_trained)。原模型和基础SD1.5各自的许可继续适用。这里只下载推理所需文件，不重发布权重。
