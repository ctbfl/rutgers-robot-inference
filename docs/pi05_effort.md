# Pi05 row1 gripper-effort 推理

普通版使用 `Pi05EffortWrapper`，training-time RTC 版使用
`Pi05TrainRTCEffortWrapper`，两者都在
`src/ruri/server/wrappers/pi05/pi05_effort.py`。
wrapper 从 checkpoint 自带的 assets 读取归一化统计，复用训练时的
`PiperEffortInputs` 和 `Pi05EffortConfig`，严格恢复全部参数。
选错普通/RTC 模型、有参数缺失或额外参数、checkpoint 未完成保存都会报错。

## 数据接口

| RURI 字段 | 格式 | 含义 |
| --- | --- | --- |
| `observation.state` | float32 `(7,)` | 现有 Piper 数据集约定的 6 个关节位置和夹爪开度；沿用控制器的 `normalize_telemetry`，不是直接发送弧度/米 |
| `observation.joint_effort` | float32 `(7,)` | 录制时同格式的六个关节力矩 + 夹爪力；策略只读取最后一项 |
| `observation.images.top` | uint8 `(H,W,3)` RGB | 顶部相机 |
| `observation.images.wrist` | uint8 `(H,W,3)` RGB | 手部相机，对应训练集的 `images.hand` |
| `prompt` | str | row1 任务原文，见下方命令 |
| 输出 `action_chunk` | float32 `(30,7)` | 已经还原到现有 Piper 控制器接收的动作空间 |

夹爪 effort 使用 telemetry 的 **带符号 `gripper_force`，单位 N**，沿用录制管线。
保留原始正负号；客户端不做额外归一化、不取绝对值。
服务端把 `effort[-1]` 拼到七维 state 后面，用 checkpoint 的分位数统计归一化，
pad 到模型宽度；模型再取这一维并裁剪到 `[-3,3]` 后送入 effort MLP。
客户端 state 和 action 始终是七维。缺失 effort 会报错，不会自动补零。

`SinglePiperConfig(include_effort_observation=True)` 使 `get_observation()`
从**同一个 telemetry 包**取得位置和 effort，使用与录制相同的
`recording_observation()`。默认关闭，现有普通客户端保持原来的数据接口。

## 启动服务

在此仓库根目录执行。使用已经装有 OpenPI 的环境；本机默认路径由脚本设置。
模型侧还需要 `/common/users/jh2400/pi05-effort` 和 `/common/users/jh2400/pi05-rtc`。
迁移机器时通过 `--effort-package`、`--train-rtc-package` 指定这两个目录，
并使用匹配的 OpenPI 源码/依赖版本。正常服务只读取 checkpoint 的 `params/` 和
`assets/`，不读取数据集，也不需要 training state。

普通 effort 最终 checkpoint：

```bash
CKPT=/common/users/jh2400/training_results/pi05_effort/260906_122040_row1_h30/checkpoints/pi05_row1_effort_h30/row1_plain_effort_h30/9999
GPU=0 PORT=5561 bash examples/shell_scripts/serve_pi05_effort.sh plain "$CKPT"
```

training-time RTC effort 使用相同入口的 `rtc` 分支。2026-09-07 检查时 RTC 尚在续训，
下面的 2000 是已经完整保存的调试 checkpoint；评测最终效果时换成完成后的 9999。
当前 GPU 1 用于 RTC 训练，推理示例使用空闲 GPU 0。

```bash
RTC_CKPT=/common/users/jh2400/training_results/pi05_rtc_effort/260906_122040_row1_h30/checkpoints/pi05_row1_rtc_effort_h30/row1_rtc_effort_h30/2000
GPU=0 PORT=5562 bash examples/shell_scripts/serve_pi05_effort.sh rtc "$RTC_CKPT"
```

服务默认使用 10 个 denoising steps；启动时完成 warmup 才监听端口。
RTC warmup 会同时编译无前缀和有前缀两条路径。
普通入口 `server_pi05.py` 和原来的 `server_pi05_train_rtc.py` 不加载 effort 架构，
这些 checkpoint 应使用上面的专用入口。

## 机器人端

更新机器人端的 RURI 代码，在机器人已有运行环境中执行：

```bash
PROMPT='insert the row of five objects, from left to right, into the bottom row of holes on the manipulation-net board.'
python examples/client_single_piper_blocking.py \
  --policy-endpoint tcp://SERVER_HOST:5561 \
  --include-effort-observation --prompt "$PROMPT" \
  --control-hz 30 --execute-actions-per-chunk 5
```

RTC 使用已有调度器，它会继续执行旧 chunk，同时发送对齐的剩余动作和延迟估计：

```bash
python examples/client_single_piper_rtc.py \
  --policy-endpoint tcp://SERVER_HOST:5562 \
  --include-effort-observation --prompt "$PROMPT" \
  --control-hz 30 --execution-horizon 5 --rtc-initial-delay-steps 4
```

`SERVER_HOST` 替换为推理机器地址，CAN/相机参数按现有机器人配置传入。
这里显式传 row1 prompt，因为现有通用客户端默认 prompt 仍是旧的单物体 E1 任务。
这些命令沿用现有 scheduler 的机器人启动与动作执行流程。

## 自己写 inference code

普通请求只需要多传 effort；ZMQ 编码、图像变换和动作还原由现有库负责：

```python
import numpy as np
from ruri.client.utils import inference_client

# state7 是现有控制器的七维 observation.state；effort7 是录制格式的测量值。
# top_rgb/wrist_rgb 均为 HWC uint8 RGB。
request = {
    "observation.state": np.asarray(state7, dtype=np.float32),
    "observation.joint_effort": np.asarray(effort7, dtype=np.float32),
    "observation.images.top": top_rgb,
    "observation.images.wrist": wrist_rgb,
    "prompt": "insert the row of five objects, from left to right, into the bottom row of holes on the manipulation-net board.",
}
with inference_client.connect("tcp://SERVER_HOST:5561") as policy:
    result = policy.infer(request)
    actions = result["action_chunk"]  # (30, 7)，直接交给现有 scheduler/controller
```

若已使用 `SinglePiperController`，在共享 args 中设置
`include_effort_observation=True`，直接用它的 `get_observation()` 构造请求即可。
不要另外读取第二次 state 再和第一次 effort 拼接；每次重新规划都发送当前测量。

自己实现 RTC 时，在上述请求基础上加入：

```python
request.update({
    "context.rtc.prev_chunk_left_over": previous_chunk[consumed:].copy(),
    "context.rtc.consumed_steps": consumed,
    "context.rtc.estimated_inference_delay_steps": estimated_delay,
})
```

剩余 chunk 第 0 行必须与这次 observation 的控制时间对齐，动作仍是未经过模型归一化的
七维控制器动作。首个请求没有旧 chunk，直接走无前缀路径即可。
训练支持 delay 0–7；wrapper 对超范围估计会警告并裁剪。
固定的是延迟覆盖的动作前缀，后续动作会使用新的 effort 重新生成。
实际执行时仍按请求发出后**真实经过的控制步数**丢弃过期动作，已有 `RTCScheduler`
已处理这部分；不要再对它返回给控制器的队列重复裁剪。

建议先固定现有无-effort 对照所用的 prompt、相机、控制频率、执行 horizon 和
denoising steps，再换 checkpoint/wrapper 做对照。上面 horizon=5 是调用示例，
正式比较应沿用之前对照实验的设置。训练 loss 和单帧 effort 敏感性只能证明模型在使用
新输入，是否提高插入成功率需要同条件的机器人评测。

## 本地验证

接口回归测试：

```bash
PYTHONPATH=src /common/home/jh2400/projects/openpi/.venv/bin/python -m unittest \
  tests.test_pi05_effort_wrapper tests.test_single_piper \
  tests.test_blocking_scheduler tests.test_rtc_scheduler
```

`tests/smoke_pi05_effort_wrapper.py` 用真实 row1 样本和 checkpoint 验证完整推理。
覆盖训练/服务输入变换逐项一致、固定 RNG 重复一致、其他六维 effort 不影响动作、
改变夹爪 effort 会影响动作、输出七维，以及 RTC 前缀还原和后缀 effort 敏感性。
运行 smoke 不连接机器人、不启动常驻服务。

2026-09-07 验证结果：39 项接口/控制器/调度器测试通过；普通版 9999 和 RTC 版
2000 的真实 checkpoint smoke 均通过。RTC 固定前缀在还原后动作空间中的最大绝对
误差为 `1.91e-6`。两种模型都验证了只有最后一维 effort 会影响动作。
详细记录位于 `/common/users/jh2400/pi05-effort/wrapper_validation/`。
