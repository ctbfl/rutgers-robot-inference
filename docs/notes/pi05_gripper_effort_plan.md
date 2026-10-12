# π0.5 row1：单维 gripper effort 微调方案

调研与实施日期：2026-09-06。外置实现位于 `/common/users/jh2400/pi05-effort`，普通版和 RTC 版已完成真实数据、完整 base 权重的 smoke。数据统计来自本机全部 parquet，架构结论以当前本机代码为准，并交叉核对官方 OpenPI 源码。

## 实施记录

最终 smoke 使用 4 个真实 row1 帧、完整 `pi05_base` 权重，固定 flow noise/time/delay、关闭图像增强，进行 120 步全量过拟合；另执行原始随机采样和图像增强的 OpenPI `train_step`。这是可训练性诊断，不代表真实插入成功率。

| 版本 | 初始固定 loss | 120 步后 loss | 通过检查数 | 实验目录 |
|---|---:|---:|---:|---|
| 普通 + effort，GPU 0 | 0.0526561 | 8.86916e-6 | 51 | `/common/users/jh2400/training_results/pi05_effort/260906_122040_row1_h30` |
| training-time RTC + effort，GPU 1 | 0.0474079 | 8.85478e-6 | 54 | `/common/users/jh2400/training_results/pi05_rtc_effort/260906_122040_row1_h30` |

最终完整记录在各目录的 `smoke_v3/result.json` 和 `smoke_v3/smoke.log`，两路进程均以 0 退出。验收包含原模型完整前向等价性、effort 两层真实梯度、实际七维动作对 effort 的敏感性、RTC 前缀固定、随机训练步、完整 checkpoint 保存恢复、optimizer 计数和原始/EMA 分支权重恢复，以及恢复后前向一致性。

在首次并发保存中 RTC 进程曾无 Python traceback 退出。最终实现通过外置 `checkpoint_guard.py` 串行化两路完整 checkpoint 的保存/恢复，降低主机内存的并发暂存峰值；随后两路完整 smoke 均通过。它不修改模型更新、loss 或 checkpoint 保留策略。临时 smoke 权重已回收，日志、结果和 checkpoint 元数据保留。

正式入口 `train_effort.py` 要求相同 variant 的成功 smoke，并校验实现/依赖/归一化资产指纹未改变。两路正式任务于 2026-09-06 12:42 启动，各自从 `pi05_base` 重新初始化，使用原 108 条数据、batch 32、10,000 steps。当前进程和目录记录在 `/common/users/jh2400/pi05-effort/active_runs.json`；实际进度见各实验目录的 `train.log`。权重保留 2000/4000/6000/8000/9999，只有最新 training state 留存。

建议：从 `pi05_base` 初始化，在 action expert 的 `adarms_cond` 上加一个 `1 → 32 → 1024` 的连续 effort 残差 MLP，末层零初始化。新增 33,856 个参数。普通版和 training-time RTC 的无 effort row1 基线均已训练完成，此次只新增两个对应的 effort 版本，分别对比已有 checkpoint；沿用原 108 条轨迹和训练设置。

## 已定位的代码和数据

| 对象 | 本机位置 / 实际配置 |
|---|---|
| 标准 OpenPI | `/common/home/jh2400/projects/openpi`；origin 为 `https://github.com/Physical-Intelligence/openpi.git`；HEAD `650c5b0`，工作区有本地修改 |
| training-time RTC | `/common/users/jh2400/pi05-rtc`；这是导入 OpenPI 的外置子类包，并非第二份完整 OpenPI |
| 普通 row1 训练配置 | `openpi/src/openpi/training/config.py` 中的 `pi05_row1_h30` |
| 当前 RTC 训练配置 | `pi05-rtc/pi05_train_rtc/config.py` 中的 `pi05_row1_rtc_h30` |
| RURI wrapper | `src/ruri/server/wrappers/pi05/pi05.py` 和 `pi05_train_rtc.py` |
| 原始数据 | `/common/users/jh2400/dataset/tight_insertion_row_1_260904_filtered` |
| OpenPI 使用的数据 | `/common/users/jh2400/dataset/tight_insertion_row1_v21` |
| 初始化权重 | `gs://openpi-assets/checkpoints/pi05_base/params` |

## 已确认的 row1 基线 checkpoint

两份基线保存在新的 `training_results` 目录下，不能仅搜索旧的 `openpi_checkpoints` 目录。

- 普通 π0.5：`/common/users/jh2400/training_results/pi05/260905_020728_row1_h30/checkpoints/pi05_row1_h30/row1_h30/9999`
- Training-time RTC：`/common/users/jh2400/training_results/pi05_train_rtc/260905_020728_row1_rtc_h30/checkpoints/pi05_row1_rtc_h30/row1_rtc_h30/9999`

已读取各 run 的 `launch.sh`、`train.log`、W&B 本地存档 `files/config.yaml`、checkpoint `_CHECKPOINT_METADATA`、`params/_METADATA` 和 norm assets，确认：

- 两者都有 `2000 / 4000 / 6000 / 8000 / 9999`；最终 checkpoint 同时保留 `params`、`assets` 和 `train_state`，参数目录有 OCDBT 数据文件。
- 日志记录完成 10,000 iterations、最终 checkpoint 成功提交，保存线程没有报错。2026-09-05 日志中的最终保存时间分别为普通版 18:13:39、RTC 版 17:50:24。
- 两者均从 `pi05_base` 恢复权重，使用 `tight_insertion_row1_v21`。启动脚本记录为全部 108 条 / 146,810 帧，实际 loader 未指定 episode 子集。
- 存档配置一致：H=30、batch=32、seed=42、10,000 steps、全量微调、EMA=0.999、warmup=1,000、LR=5e-5→5e-6、workers=12、`discrete_state_input=False`。AdamW 为 b1=0.9、b2=0.95、eps=1e-8、weight_decay=1e-10、gradient clip=1。
- RTC 存档的 delay_probs 为 `[0.15,0.05,0.1,0.2,0.2,0.15,0.1,0.05]`。其 params 有 `tok_time_proj/kernel`（1024×1024）和 bias（1024）；普通版没有。两份参数树和已记录的输入 transforms 均没有 effort 分支。
- 两份 norm_stats.json 逐字节一致，SHA-256 为 `5573a1ada4cbf4e9e69172de717c28e1a5b4ee4ce08666b9ecec24e557ef83f1`；state/action 统计均为七维。

这足以确认训练完成和基线身份；本轮没有执行完整 checkpoint restore 或实机推理，不把元数据检查等同于逐个权重数据块的完整性测试。

examples、RTC README 和 wrapper 的一些默认名称、checkpoint、prompt 仍指向旧 E1 实验。此次以实际 row1 训练配置为准，部署时必须一起更新。row1 的任务文本是：

> insert the row of five objects, from left to right, into the bottom row of holes on the manipulation-net board.

## 数据核查

`meta/info.json` 和全部 108 份 episode parquet 一致：108 episodes、146,810 frames、30 Hz。用户所说的约 100 条，在当前目录中实际为 108 条。

原始字段 `observation.joint_effort` 是 float32、shape `(7,)`，最后一个名字为 `gripper.effort`。唯一新增网络输入为当前帧的 `observation.joint_effort[..., -1:]`，不取前六个关节 effort，不读取未来 effort，也不预测 effort。

| 全量数据统计 | 值 |
|---|---:|
| gripper effort 最小 / 最大 | -1.535 / 0.597 |
| q01 / q99 | -1.047 / 0.196 |
| 均值 / 标准差 | -0.390688 / 0.395732 |
| 中位数 | -0.295 |
| 非有限值 | 0 |
| 不同数值数 | 1,738 |
| 负值比例 | 85.29% |
| 零值比例 | 0.277% |
| episode 内相邻帧相等比例 | 6.74% |
| 最长连续不变时间 | 0.233 秒 |
| 与 gripper position 的 Pearson 相关系数 | 0.691 |

结论：此信号有变化，值得尝试；不能把负值截为零或直接取绝对值。上述统计不证明它包含足以提升插入成功率的信息，也不证明传感器与图像完全同步。

本地 `single_piper/normalization.py:recording_observation` 将此维定义为 signed gripper force（N），来自 follower telemetry 的 `gripper_force`；前六维才是关节 torque（N·m）。这里遵循现有采集约定，未独立校准传感器。夹爪反馈不能直接视为末端六维接触力，尤其不能保证检测到所有轴向插入阻力。

## 为什么仅仅拼接 state 无效

当前普通 row1 和 RTC row1 都设了 `pi05=True, discrete_state_input=False`。

1. `ModelTransformFactory` 将该开关传给 `TokenizePrompt`；False 时 tokenizer 不读取 state。
2. `Pi0.embed_suffix` 仅在 `not self.pi05` 时使用 `state_proj(obs.state)`。
3. 因此当前 π0.5 网络不直接编码原有七维 state。state 仍用于 DeltaActions / AbsoluteActions，以及推理中相对动作还原，这与进入神经网络是两件事。

把 effort 拼进去而不加显式网络读取路径，不会让当前模型用到它。把 `discrete_state_input` 改成 True 并拼接 effort 能提供另一种输入路径，但同时启用了全部 proprioception 和另一种 prompt 格式，不能单独归因于 effort。第一轮保留 False。

官方实现也区分了这两条状态路径：[Pi0Config](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py)、[Pi0 模型](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)。

## 最小网络修改

原始路径：`flow timestep → time MLP → adarms_cond → action expert 的各层 adaRMSNorm`。

新增路径：`当前 gripper effort → 单维归一化 → Linear(1,32) → SiLU → Linear(32,1024) → 加到 adarms_cond`。

公式：

```text
ê = clip(2 * (e - q01) / (q99 - q01 + 1e-6) - 1, -3, 3)
g(ê) = W_out · SiLU(W_in · ê + b_in) + b_out
c(t,e) = c_base(t) + g(ê)
```

当前 `gemma_300m` 的 action expert width 是 1024；代码应从 `self.action_in_proj.out_features` 取宽度。参数数目为 `1×32 + 32 + 32×1024 + 1024 = 33,856`。

`W_in` 正常随机初始化；`W_out` 和 `b_out` 全零。这样在加载相同 base 权重、给定相同输入和噪声时，新分支初始贡献为零，设计上保持原模型输出。不要把两层都置零，也不要再叠加一个零初始化乘法 gate，避免阻断学习。首步内层梯度为零可以是正常现象，应检查末层先获得梯度、更新后内层也获得梯度。

核心结构示意如下；实际实现见外置包 `pi05_effort/model.py`：

```python
class Pi05GripperEffort(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        if not self.pi05:
            raise ValueError("requires pi05=True")
        width = self.action_in_proj.out_features
        self.effort_mlp_in = nnx.Linear(1, 32, rngs=rngs)
        self.effort_mlp_out = nnx.Linear(
            32, width,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def embed_suffix(self, obs, noisy_actions, timestep):
        tokens, mask, ar_mask, cond = super().embed_suffix(
            obs, noisy_actions, timestep
        )
        # 输入管线已完成归一化；固定 slot 7，不能读 padded state[-1]。
        effort = jnp.clip(obs.state[..., 7:8], -3.0, 3.0)
        residual = self.effort_mlp_out(nnx.swish(self.effort_mlp_in(effort)))
        return tokens, mask, ar_mask, cond + residual.astype(cond.dtype)
```

选择这个位置的原因：复用现成的连续条件接口、不增加 token 或注意力长度、不改变已有权重形状；新增信号经过后续网络可以影响七维动作中的任何维度。这里的融合方法是针对本项目的设计建议，并非声称官方已经验证了 gripper effort 的效果。[官方 adaRMSNorm 源码](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py) 支持上述接口判断。

普通 π0.5 的 loss 和 sampling 都调用 `self.embed_suffix`，一次覆盖即可同时接入训练与推理；不修改 flow-matching loss、time MLP 或 action projection。

## 数据通路：借用已有 state 的第八个槽位

为了减少基础设施改动，建议在自定义 data transform 中把 effort 放到已有 observation state 容器，而不是新增全局 `Observation` 字段：

```text
原始 state:   [q1,q2,q3,q4,q5,q6,gripper_position]        shape 7
拼接后 state: [q1,q2,q3,q4,q5,q6,gripper_position,effort] shape 8
Normalize → PadStatesAndActions(32)
网络读取: obs.state[..., 7:8]                            shape B×1
```

这是传输容器上的约定，不是把 effort 变成 action。原始/对外 action 始终七维，内部 action 仍 pad 到 32；`action_dim` 保持 32，不能改成 8。DeltaActions / AbsoluteActions 的 mask 保持 `make_bool_mask(6, -1)`，只处理前六个关节，夹爪 position action 保持绝对值。

具体改动：

1. 自定义 Piper data config 的 repack 保留 `observation.joint_effort`。
2. 自定义 inputs transform 在构造 `state` 时只拼接该字段的最后一维。需要放在原始 repack 和 `LiberoInputs` 丢弃字段之前，或由自定义 inputs transform 接替 `LiberoInputs`。
3. 创建新八维 state 的 norm assets：精确复制已有 checkpoint 的前七维 state 统计及全部 action 统计，仅向 state 的 mean/std/q01/q99 数组追加当前 108 条训练轨迹的单维 effort 统计。原七维统计文件不能直接用于八维输入，也不要重新估计已有 action 统计而引入额外差异。无需新增统计 key，已有 `compute_norm_stats.py` 的 state/actions 结构即可兼容。
4. 推理加载 effort checkpoint 自己的八维统计；保留原基线 assets。裁剪仅作用于 effort，不改原七维 state/action 归一化。保留超出 q01/q99 的一定幅度，以免把接触极值全压在 ±1。
5. 新配置验证 `discrete_state_input=False`、原始 state shape `(7,)`、原始 effort shape `(7,)` 和有限值。缺失新字段不能静默补零。

未来若改为独立 `gripper_effort` 字段，需要同时改 Observation dataclass、from_dict、preprocess_observation、inputs_spec、norm stats 和 Policy output unnormalization 等路径；仅让 dict 多一个 key 会被当前代码丢掉。这不作为此次最小实现首选。

## 权重加载与代码组织

沿用 RTC 的外置包模式，已新增 `/common/users/jh2400/pi05-effort`，包内包含 `model.py`、`config.py`、`transforms.py`、`weight_loader.py` 和训练入口。`Pi0Config.create()` 返回新模型；训练入口复用 OpenPI `scripts/train.py:main`。

默认 `CheckpointWeightLoader` 仅允许补充缺失的 LoRA 参数，不能原样用于新分支。自定义 loader 应严格继承 base 所有原参数，仅允许新命名空间 `effort_mlp_in/.*`、`effort_mlp_out/.*` 从初始化模型补齐；若启用 LoRA，再增加对应允许项。必须检查参数树和形状，避免用宽泛 `.*` 掩盖 base 缺失。

推理必须以新模型 config 加载 checkpoint。`BaseModelConfig.load` 默认会 intersect 参数树，普通 config 可能静默丢掉 effort 参数，因此新 wrapper 需核验模型类型/参数和 eight-dimensional state stats，并让普通 wrapper 拒绝这种 checkpoint。

## RURI 推理接线

当前 `SinglePiper.get_observation()` 只输出 state/images；`get_recording_observation()` 才通过 `recording_observation(packet)` 输出 effort。应让新策略的推理 observation 从同一个 follower telemetry packet 提取 state 和 effort，再附加摄像头数据；不要为两个字段分别读不同 packet。

增加 effort wrapper 的 input mapping、metadata、`_to_openpi_observation` 和 warmup 输入。当前 `_to_openpi_observation` 显式重建字典，仅在外部输入附加字段不会自动传入 OpenPI。原始推理契约建议继续发送七维 `observation.state` 和七维 `observation.joint_effort`，在共享训练/推理 transform 中统一抽取最后一维，避免在线拼接与离线拼接不一致。

需要检查采集 force 的更新时间/有效性与推理观测时刻；recording helper 校验有限值并不等价于校验每个传感器的独立时间戳。

## 微调与实验设计

无 effort 的普通版和 RTC 版直接使用上面已经完成的 checkpoint。只训练新增的 effort 普通版和 effort RTC 版，两者仍从 `pi05_base` 开始，匹配已有基线的训练预算：

| 项目 | 设置 |
|---|---|
| 模型 | π0.5，H=30，action_dim=32，discrete_state_input=False |
| 数据 | 与已有基线相同的全部 108 个训练 episodes / 146,810 帧 |
| batch / steps / seed | 32 / 10,000 / 42 |
| LR | warmup 1,000；peak 5e-5；cosine decay_steps=10,000；最终 5e-6 |
| optimizer / EMA | AdamW；clip_gradient_norm=1；EMA=0.999 |
| checkpoint | 每 2,000 steps 保存；主比较固定最终 step 9999 |
| 冻结 | 与现有 row1 baseline 一样，不冻结；MLP 同时训练 |

保留全量微调是为了与已完成实验设置对齐，不表示这是约 100 条数据上的最优选择。此次先不改变冻结策略、LoRA 或训练时长。

此次保持原数据范围，不再切成 100 train + 8 validation。已有基线见过全部 108 条，事后从中划八条不能作为未见过的验证集，也会使新的 effort 训练与已有基线不再配对。独立效果评估使用新实机 trials 或另外采集的 held-out trajectories；原训练集的离线误差仅作为诊断。

对照关系固定为：已有 `pi05_row1_h30/9999` vs 新普通 effort 版；已有 `pi05_row1_rtc_h30/9999` vs 新 RTC effort 版。只新增两个训练 run，不重训无 effort 基线，不需要为此次实验修改 loader 的数据拆分功能。

10,000 steps × batch 32 约相当于 2.18 次遍历全部训练帧，但 action chunk 重叠，这不是独立样本数量。新增分支在父模型初始化之后创建，并保留已有随机数路径，避免无意改变 loss 的采样或图像增强方式。

离线诊断使用固定 observation/noise/timestep 的 flow loss，补充只看真实七维 action 的误差指标；原模型 loss 包括 padding 维度，不能只靠总 loss 判定效果。主比较固定最终 9999，不用实机测试集反复挑 checkpoint；普通和 RTC 的 loss 定义不同，应各自与对应 effort 版本比较。

实机在相同初始摆放、推理步数、重规划周期和执行 chunk 长度下交替运行 A/B。至少先做每组 20–30 次完整 row1 trials，报告五个全插完的成功率、每个物体的插入成功率、完成时间、重试/掉落次数及不确定性；小幅差异需更多 trials 或训练 seeds 才能确认。

对 B 的同一 checkpoint 补充离线 effort 置为训练中位数、以及跨样本置乱的敏感性检查。性能下降只能说明模型依赖该输入，不能单独证明使用了接触信息：本数据 effort 与 gripper position 的相关性为 0.691。若要进一步支持“接触信息”的解释，应增加相同分支输入 gripper position 的训练对照。

若将来为了快速试验，改为从现有 9999 checkpoint 继续训练 effort 模型，那属于另一个训练预算：不能把“多训练若干 steps + effort”的收益全部归因于 effort，需要另设同样继续训练步数的无 effort 对照。此次按用户原目标，从同一 `pi05_base` 训练 effort 版本即可复用已有两份基线。

## 与 training-time RTC 的衔接

已有普通版和 RTC 版均已完成，因此新增版本对应为 π0.5+effort 和 RTC+effort，可按资源情况先后训练。不要用普通 π0.5 对比 RTC+effort 来判断 effort 收益。

本地 `Pi05TrainRTC.embed_suffix_rtc()` 直接调用 `super().embed_suffix()`。只在其子类覆盖 `embed_suffix` 不会自动让 RTC 的训练/条件采样调用该覆盖。RTC+effort 必须覆盖 `embed_suffix_rtc`，对它返回的 `cond` 加同一个 MLP residual，并确认所有实际 loss/sample 路径经过该方法；保留 `tok_time_proj` 和 delay-conditioned loss。首次从 base 初始化时，loader 同时允许新 effort 参数及 RTC 参数，原 base 参数仍严格校验。

模型一次规划只看到当前 effort，不能在一个已发出的 action chunk 内获得未来触觉反馈。H=30 在 30 Hz 下覆盖 1 秒，因此必须控制两组实际执行多少步再观测，而不是只看网络 horizon。RTC 中已承诺前缀也不能因新 effort 被即时改写；effort 对后续动作的帮助受实际观测/推理延迟约束。[Training-Time Action Conditioning 原论文](https://arxiv.org/abs/2512.05964) 讨论的是该动作前缀条件机制，并未验证本方案的 effort 融合。

## 实施验收点

- 精确抽取 effort 最后一维；相同 observation 经训练与推理 transforms 后一致；内部 state[7] 有值、state[8:] 为 padding；对外 actions 为七维。
- 相同 base 参数、noise、观测下，零初始化分支保持 baseline 输出；首次加载严格验证所有原参数。
- 真实样本 loss 的 effort 末层梯度非零，更新后内层也可学习；训练后固定图片/噪声、改变 effort 时输出会变化。零初始化时输出不随 effort 变是预期行为。
- checkpoint 保存/恢复后仍保留分支和相同 norm assets；普通 wrapper 不能静默加载成 baseline；effort wrapper 缺传感器输入报错。
- 若扩展 RTC，分别检查训练、无前缀采样和有前缀采样；承诺前缀保持不变，后缀可受 effort 影响。

本次已完成全量 parquet 审计和代码路径调研，并在 OpenPI 的现有 Python 环境中用 CPU 验证了：

- 真实数据拼接到八维 state 后，现有 DeltaActions / AbsoluteActions 往返正确，quantile normalization 和 32 维 padding 保留 effort 于 slot 7；action 的第八维以后仍全为 padding。
- 独立 NNX MLP 的参数量确为 33,856、初始输出全零；一次合成目标更新时末层获得非零梯度，更新后内层也获得非零梯度，输出开始随 effort 改变。

上述是早期 CPU 检查。后续已完成完整 π0.5 的 GPU smoke，详见本文开头的实施记录。真实任务成功率仍需训练后在实机上与已有基线比较。RURI 客户端/服务端的在线 effort 接线属于后续部署工作，当前训练与 smoke 使用共享的 OpenPI 数据 transforms。
