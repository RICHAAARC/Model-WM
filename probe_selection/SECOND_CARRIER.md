# 第二载体接口核查 · Guidance Watermarking + VideoSeal（SD2 配置）

选择作者 ICLR 2026 的生成时梯度引导系统，VideoSeal 在这里是冻结读者。检查作者源码修订 d1f8e48946f830087baf32356e5ce0eeec68e27e。LaWa A+M、Guidance B+M 的目标是分别发行副本并各自校准。

| 接口 | 已核查的证据 | 尚待验证 |
|---|---|---|
| 同作品不同身份发行 | 官方 generate 接受 prompt、初始 latents、逐样本 key，加载调度器后在去噪过程中使用该 key 的梯度。每次应克隆同一初始状态、重置调度器及所有随机源，串行发放；作者方法将 key 存为实例属性。 | 多身份图及其真实 RGB 混合在相同主体和布局下可用，需要真实发行验证。 |
| 连续读数 | 官方 videoseal_w.yaml 为 256 位。使用 detector.preprocess 后接 decode_message，可返回阈前实数及可选白化。已在 CPU 上实例化官方 VideoSealDetector，用合成检测器和单位白化矩阵执行 RGB→预处理→连续 256 维→同一个软评分函数。 | 真实提取器、训练得到的白化器与 SD2 尚未在此轮联合加载。作者的权重与白化路径需绑定并核实；无 key 的默认 forward 会访问未赋值的 pval_0bit，应使用上述连续接口。 |
| 候选写法 | 官方 SD2 配置 wm_scale=250、top_percent=0.1、max_norm=0.3。源码将 wm_scale 用于水印梯度的噪声更新。可在保持同一起始 latent、相同步数和更新数量的条件下，构造相邻早/晚活动步之间的引导预算转移。 | 该动作是待实现的第二适配器。实际每步梯度范数不同，应按实测引导能量与输出质量配平。不能直接移植 LaWa 的尺度系数、码长或阈值。 |
| 原生与共同机制 | M 关闭时保持原生带水印 guidance 的完整配置，不把 wm_scale 设为零。开启时共用同作品探针真实 RGB 合成、独立码表、质量筛选、成员—无辜选择和先选后发行规则。 | 原生旁路与适配器在真实 SD2 输出上的数值一致性、独立阈值下的配对收益仍待验证。 |

复查命令为 python -m probe_selection.check_carriers_cpu LAWA_SOURCE SD14_METADATA GUIDANCE_SOURCE，使用固定 CPU 依赖和随机张量。第二载体当前证据为**官方生成代码静态核查 + 作者读者接口合成 CPU 调用**。没有第二载体真实生成、同权重迁移或通用收益结论。仅在 LaWa 的配对数值信号和主体/自然度复核都支持时，另行安排第二载体真实验证。

参考 [论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b9501301a7a0f99e57406cc901f4e3ce-Abstract-Conference.html)、[发行与引导更新](https://github.com/EnoalG/Guidance-Watermarking-for-Diffusion-Models/blob/d1f8e48946f830087baf32356e5ce0eeec68e27e/guidance-watermarking-for-diffusion-models/watermarking/watermark.py)、[连续读者](https://github.com/EnoalG/Guidance-Watermarking-for-Diffusion-Models/blob/d1f8e48946f830087baf32356e5ce0eeec68e27e/guidance-watermarking-for-diffusion-models/detector/detector.py)、[官方 SD2 入口](https://github.com/EnoalG/Guidance-Watermarking-for-Diffusion-Models/blob/d1f8e48946f830087baf32356e5ce0eeec68e27e/guidance-watermarking-for-diffusion-models/scripts/sd2.sh)。
