# Pi05 关节力矩历史 + 训练期 RTC 推理（whole_v1）

wrapper：`src/ruri/server/wrappers/pi05/pi05_torque.py` 里的 `Pi05TrainRTCTorqueWrapper`。
服务入口：`examples/server_pi05_torque.py`，启动脚本：`examples/shell_scripts/serve_pi05_torque.sh`。

模型来自 `/common/users/jh2400/pi05-torque`（照 TA-VLA 的做法，把 7 维测量力矩的
**1.8 秒历史**接到 π0.5 action expert 的 adaRMS 条件上）。训练过程和设计取舍见
`/common/users/jh2400/model_training/pi05_train_rtc/whole_v1_torque_4gpu_bs128/README.md`。

**完整的客户端契约在服务端 metadata 里**，写客户端时以它为准：

```bash
python examples/list_policy_servers.py        # 找 name = pi05_train_rtc_torque 那一项
```

`inputs` 逐个键写了形状、dtype、单位和来源；`outputs` 写了 `action_chunk` 的含义和所有
响应字段；`policy.client_guide` 是分步骤的客户端写法。

## 和 pi05_effort 的区别

`pi05_effort`（row1 夹爪力）要的是**当前一帧** `observation.joint_effort` `(7,)`。
这个模型要的是**一段带时间戳的历史**，两者不兼容，发错会直接报错。

| RURI 字段 | 格式 | 含义 |
| --- | --- | --- |
| `observation.state` | float32 `(7,)` | `normalize_telemetry(P)`，和历史最新一行来自同一个 telemetry 包 |
| `observation.joint_effort_history` | float32 `(N, 7)` | 客户端缓冲的测量力矩，**旧的在前**；6 个关节力矩 N·m + 夹爪力 N，带符号、原始值 |
| `observation.joint_effort_history_t` | float64 `(N,)` | 每行的采集时间（秒），严格递增，同一时钟，比如 `P['monotonic_ns'] / 1e9` |
| `observation.images.top` | uint8 `(H,W,3)` RGB | 顶部相机 |
| `observation.images.wrist` | uint8 `(H,W,3)` RGB | 手部相机，对应训练集的 `images.hand` |
| `prompt` | str | **必填**，目标孔位，只能是 `A0`–`E4` 之一，原样 |
| `context.rtc.*` | 可选 | 和 `pi05_train_rtc` 完全一样 |
| 输出 `action_chunk` | float32 `(30,7)` | 绝对目标，和 `observation.state` 同一套约定，每行一个 30 Hz 控制步 |

## 为什么要客户端发带时间戳的缓冲

模型的力矩历史定义在时间上：以当前为 0，往前 0.2、0.4 …… 1.8 秒，共 10 帧
（30 fps 下的帧偏移 `-54, -48, …, -6, 0`）。

但现有调度器都不是按这个节奏读机器人的：阻塞调度器每个 chunk 读一次，RTC 调度器每次请求
在推理线程里读一次。服务端自己根本攒不出这个时间栅格。所以由客户端按自己的节奏
（≥30 Hz）缓冲，连同时间戳发上来，**服务端按训练时的偏移挑行**——偏移量只在服务端一个地方定义。

挑行方式是 **sample-and-hold**：对每个偏移，取时间不晚于 `t_最新 + 偏移` 的最新一行
（留 1 ms 容差，所以严格 30 Hz 的缓冲能逐帧复现数据集的索引）。这就是录制时的行为——
录制器在每个 30 Hz 帧锁存当时最新的包。

**比缓冲还早的偏移，用最旧的一行填。** 这和训练完全一致：LeRobot 把越界的历史索引夹到
episode 起点，所以每条 episode 的前 1.8 秒，训练时看到的就是"重复第一帧"。客户端只要
**在 episode 开始时清空缓冲**，就和训练对得上，不需要任何特殊处理。

## 客户端现状（写客户端的 agent 需要知道）

- `MITTelemetryReceiver` **只保留最新一个包**，新包直接覆盖。所以历史只有在有人以
  ≥30 Hz 轮询它时才存在。
- 没有新包时 `latest_engaged()` 会**再次返回同一个包**，按包里的 `sequence` 去重。
- 接收器**不是线程安全的**，而 `RTCScheduler` 本来就在推理线程里调 `get_observation()`。
  所有 telemetry 读取应集中在一处（比如一个采样线程负责缓冲并缓存最新包，其他读者读缓存）。
- `SinglePiperController.get_observation()` 内部自己读包，**不返回包和时间戳**，目前也没有
  公开接口能拿到。需要新增一个。
- 每次请求，`observation.state` 和缓冲最新一行必须来自**同一个包**；相机帧用
  `get_camera_observation()`。不要把缓冲和 `get_observation()` 的 state 拼在一起——它读的
  可能是更新的一个包。

## 响应里的诊断字段

每次响应都带着服务端实际用了什么，建议客户端全部记日志：

| 字段 | 正常值 | 不正常说明什么 |
| --- | --- | --- |
| `torque.slots_from_oldest_row` | episode 开始 1.8 秒后为 0 | 一直不为 0：缓冲太短，或每次请求前被清空了 |
| `torque.max_hold_ms` | 30 Hz 下 < 40 | > 100：缓冲采样太稀，分辨不出 0.2 秒的栅格 |
| `torque.clipped_values` | 0 | > 0：力矩远超训练范围，检查单位和通道顺序 |
| `torque.history_rows` / `torque.history_span_s` | 约 60 行 / 约 2 秒 | — |

服务端会直接拒绝：形状不对、NaN、时间戳不严格递增、时间戳不是 float64（整数的
纳秒和秒分不清，float32 在 monotonic 秒量级上分辨不了毫秒）、|力矩| > 200
（训练集里最大才 18.3 N·m，超过这个一定是单位错了）、prompt 不在 `A0`–`E4` 里、
只发了单帧 `observation.joint_effort`。

服务端**检查不了**：最新一行和 state 是不是同一个包、通道顺序对不对、episode 开始时有没有清缓冲。

## 启动服务

```bash
bash examples/shell_scripts/serve_pi05_torque.sh                     # 默认第 10000 步，GPU 0，端口 5563
GPU=1 PORT=5564 bash examples/shell_scripts/serve_pi05_torque.sh \
  /common/users/jh2400/training_results/pi05_train_rtc/260915_whole_v1_torque_rtc_h30_4gpu_bs128/checkpoints/pi05_whole_v1_torque_rtc_h30/whole_v1_torque_rtc_h30_4gpu_bs128/5000
```

启动要恢复参数、再编译两次（无前缀 + 有前缀），等到 ready 行再连。默认名字
`pi05_train_rtc_torque`，用 `NAME=` 改。端口 5555–5562 已被其他脚本占用。

想和无力矩对照组比，公平的是**第 5000 步**（和无力矩第 7499 步看过同样多数据，2.19 epoch）。

wrapper 从 `/common/users/jh2400/pi05-torque` 导入模型包，**不能**用训练 run 目录下的
`source/` 快照：那份的输入 transform 会拒绝所有带 RTC 前缀的推理请求（已在
pi05-torque 79aff65 修复，只影响推理）。加载时会探测这一点，用错了直接报错。

## 本地验证

```bash
PYTHONPATH=src /common/home/jh2400/projects/openpi/.venv/bin/python -m unittest tests.test_pi05_torque_wrapper
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /common/home/jh2400/projects/openpi/.venv/bin/python \
  tests/smoke_pi05_torque_wrapper.py --output /tmp/torque_smoke.json
```

2026-09-16 第 10000 步的真实 checkpoint smoke 全部通过：

- 客户端风格的 30 Hz 缓冲经过 wrapper 后，模型收到的 77 维 state 与训练管线**逐位相同**
  （episode 中段和开始后第 3 帧都是）
- 推理 56 ms（10 步去噪），采样动作与录制值平均差 0.50°
- 只换力矩历史，动作平均变 0.49°
- RTC 固定前缀在原始动作空间的最大误差 7.6×10⁻⁶；换力矩时固定行完全不动，后面的行跟着变
- 通过真实 ZeroMQ 服务端和 `inference_client` 往返正常

记录在 `/common/users/jh2400/pi05-torque/wrapper_validation/`。
