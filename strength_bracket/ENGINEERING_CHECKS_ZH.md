# 强度夹逼 DEV 工程 CPU 核验

2026-11-25。15 项针对性 CPU 检查全部通过。测试命令在 `colab_release/` 执行：

```bash
../audits/wam-qualification-preparation-20261109/cpu-venv/bin/python -m unittest strength_bracket.test_cpu -v
```

当前完整日志为 `cpu-checks.log`；第一次因工程文件尚未重封而失败的日志保留为 `cpu-checks-attempt01.log`，第二次通过的日志保留为 `cpu-checks-attempt02.log`。后续补入当前运行目录坏缓存边界后再次通过，未删除历史检查记录。

## 已覆盖

- 一次冻结的 192 件：8 DEV、24 CAL、160 TEST；身份分配、全部历史清单摘要及 prompt/seed 不重复，码本与原冻结码本相同。没有运行抽样脚本或改写 allocation。
- DEV 全局强度仅由质量代理选择；有限正数检查、非单调/缺档停止、边界不能把同一强度占两档。
- 160 件质量区间和三态判定；4/5 个无反向胜作品的精确单侧符号检验边界；误指认、质量或完成度门槛；零赢上界。不把不显著写成等价。
- 实际 DEV owner、metered 子进程、worker 和封存/重开路径，以确定性的纯色图像 fixture 完整走通 8 次生成、336 次写入、0 次读数；保存全部 336 张模拟身份图。Fixture 只验证接口与文件/记账流程，不模拟水印效果。
- 六处 uniform hook 输出与异常后清理；错误 RGB 类型/尺寸停止；首个写入失败保留 STARTED、ERROR 与 traceback，再次进入不能重做。
- 对已保存官方 SD1.4 小配置执行真实 CLIP tokenizer、DDIM scheduler、UNet/VAE/text 配置解析；用 meta device 构造相应组件结构，不加载真实参数。错误 latent 缩放、cross-attention 维度及 prediction_type 拒绝。
- 原费用逐项高水位承接、分阶段费用求和、结果体积封存、只读重开一致、未知尾部不捏造费用、未完成 JSONL 尾行处理。
- 注入 KeyboardInterrupt 后 supervisor 回收子孙进程并封存费用；没有重新开始模型操作。
- 缓存来源与 `/content` 限制；旧运行空缓存保留并记 skip，能够继续找到有效旧缓存；当前运行 partial/empty/来源不明缓存必须停止，不以邻居缓存覆盖或绕过。
- notebook 第一格只挂载 Drive；54 个内嵌文件与当前源码逐字节一致，receipt 完整有效；不包含权重和正式比较 worker。

## 工程修复

仅修改 `strength_bracket/assets.py` 的缓存复用边界：旧运行已损坏缓存不是当前运行可复用资产，记录原因后跳过；当前运行目录先执行原有 `cached()` 校验，保留原有不可替换/不可重试约束。历史文件不删除，费用规则不变。随后重建工程 receipt 与内嵌 notebook；没有重抽作品、seed 或身份，没有改变增益网格与统计判据。

## 证据范围

这些结果允许交付 DEV 工程入口，不代表真实 GPU、真实权重、真实水印写入或质量夹逼已通过。正式入口仍只允许用户手动运行 DEV；CAL/TEST 未启动。现有两组 80 件实验结论与费用保持原样。测试中的纯色图像不进入科学数据集或论文实验分母。
