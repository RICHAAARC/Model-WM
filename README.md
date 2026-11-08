# Model-WM

AquaLoRA 的 44 轨迹 Colab 资格检查。仅检验加载、原生消息读出、同作品质量和成本；不训练，不执行正式 CAL/TEST，不宣称抗串谋或创新通过。

用户于 2026-11-08 接受“新内容未被所选 checkpoint 使用”的工作假设。实际训练、选择和评估使用史仍未核实；本入口不再以此阻断运行。

[在 Colab 打开](https://colab.research.google.com/github/RICHAAARC/Model-WM/blob/main/notebooks/aqualora_qualification_colab.ipynb)

**当前 notebook 是同一 attempt01 的一次模型阶段续接入口。由用户选择单GPU运行时，然后点击全部运行。** 第一格只挂载Drive；后续代码从本仓库取得。原结果目录为：

`MyDrive/Model-WM/aqualora-qualification-20261108-attempt01/`

原目录及 `environment-recovery-01/` 的文件保持原样。续接日志、状态、图像和全部配对结果写入其中的 **`model-stage-01/`**，该子目录不是新实验attempt。只接受已核实回执链上的、费用完整的ENVIRONMENT_READY和0/44状态；来源不清、已有模型执行、预算不足均停止。模型阶段先占位，失败、中断和重复点击不会再次执行；硬中断后读取真实账本中的STARTED/FAILED，不用旧摘要的0覆盖它们。

优先复用恢复运行时的独立Python环境，并用包元数据和pip check核对恢复时的完整版本记录。环境缺失时优先复用其已付费wheel、官方Python归档、解释器和uv；只补齐缺失软件。缓存来源、格式或版本不符即停止，不自动换版本或镜像。已付费缓存不重复计下载，新文件全部计入原预算。

承接**最新累计**墙钟/GPU时间额度192.58095860481262秒、CPU核时129.291007秒、下载3465446284字节及其他已记费用。新安装、下载、CPU工作和生成时间全部累计；RAM/设备峰值取跨段最大值，结果大小按整个attempt统计。不退回原24.22秒起点，不重置码本、seed、参数、轨迹计数或预算。

启动大文件下载前，对冻结官方资产做HEAD大小预检。当前模型文件合计4,836,415,535字节（4.50GiB）；剩余有效下载额度6,735,101,044字节（6.27GiB）。**若切换GPU后软件缓存全部丢失**，按已观测软件费用估计合计约7.46GiB，入口将停止于下载预算预检，不能保证完成44条。不会通过重置费用、换模型或缩减实验来绕过预算。

全部44条及数值门完成后，仍需检查 `visual_review/` 中全部8组的主体、构图和明显重影；未审查时为UNRESOLVED。不会自动宣布资格通过。完整基线仍需独立决定，吞吐估计不构成其启动授权。

固定输入见 [manifest.json](qualification/manifest.json)，此次假设和执行方式修订见 [AMENDMENT.md](AMENDMENT.md)。模型参数、解码预处理、PNG/JPEG和质量门来自此前已采用的44轨迹方案；没有为Colab降低步数、精度、分辨率或门槛。

LoRA 张量接口修复保持不变。**43项本地CPU工程检查通过**，包括12项模型续接检查、13项环境恢复检查、14项原控制检查和4项锁定版本加载回归；另通过真实pip离线缓存复用、官方Python归档离线恢复、官方资产HEAD预检和实际Drive回执链只读检查。入口会更新已有代码checkout并重新载入入口模块。检查方法和证据边界见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。

**本次交付未执行Colab模型阶段、下载或加载真实模型权重、启动GPU。实际GPU运行由用户启动。** CPU检查和文件头预检不代表真实权重兼容性、GPU内存或吞吐已验证；出现问题停止并保留记录。不会进入CAL/TEST，运行时日志含所用版本和代码commit。

参考实现：[AquaLoRA](https://github.com/Georgefwt/AquaLoRA/tree/62fb5cf186050dd4e3bce0bfe15068872fc57d62)、[ppft权重](https://huggingface.co/georgefen/AquaLoRA-Models/tree/98688b2a1e762339593ee8fe96ed13762f06b732/ppft_trained)。原模型和基础SD1.5各自的许可继续适用。这里只下载推理所需文件，不重发布权重。
