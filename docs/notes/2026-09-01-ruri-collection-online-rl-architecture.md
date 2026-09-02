# RURI Piper 与 LeRobot 数据采集架构（统一版本）

状态：已按本版本在 `client/ruri-piper-lerobot` 分支实施。

## 当前只解决什么

本轮范围严格限制为：

1. 让 RURI 自包含当前已验证的 Piper MIT 控制链，不再运行时依赖旁边的
   `piper_teleop_agx` repo。
2. 新增基于现有 `SinglePiperController` 的 leader/follower teleop controller。
3. 将现有 LeRobot robot / teleoperator plugin 变成依赖 RURI 硬件事实的薄 adapter。
4. 统一 inference 与 collection 的相机参数、Piper normalization 和 telemetry packet。
5. 将官方 `pyAgxArm` 作为可选 submodule vendor。

不在本轮引入 Online RL recorder、reward API、replay buffer、transition/sample
dataclass、相机时间戳或新的 hardware/runtime 分层。

## 保持现有三层主体

RURI 继续保持现有概念：

- server 侧 `policy_wrapper` 负责把具体模型适配成 RURI policy server；这个层保留。
- client 侧 scheduler 负责 inference timing 和 action chunk 语义。
- client 侧 controller 负责硬件生命周期、观测和最终 target 下发。

远程 inference 连接只是 client utility：

```python
from ruri.client.utils import inference_client

with inference_client.connect(args) as policy:
    scheduler.run(
        controller=SinglePiperController,
        policy=policy,
        args=args,
    )
```

不存在 LocalPolicy 层，也不把 inference client class/factory 作为 Scheduler 参数。

## Action 的唯一语义

policy server 返回完整 chunk。Scheduler 决定：

- chunk 中执行多少行；
- RTC 的剩余 chunk 和 consumed steps；
- rolling aggregation 或 temporal ensemble；
- minimum jerk；
- 最终的 defensive clipping。

Controller 收到的就是最终 target。它只做：

1. shape、finite 和 range 验证；
2. 固定的 normalized-to-hardware 坐标转换；
3. 下发 target。

`send_action(target)` 返回 `None`，不会静默裁剪，也不会制造
“requested/accepted/executed action”三套语义。Scheduler 日志记录最终下发 target；发生
防御性裁剪时另记 pre-clip action。

未来可训练 policy 应优先使用有界 action distribution。梯度问题属于 policy/training
设计，不进入 Controller。

## Controller 布局

保持 controller 为硬件核心，不改成新的 `hardware/`、`runtime/` 大重构：

```text
src/ruri/client/controllers/
├── single_piper/
│   ├── single_piper.py
│   ├── normalization.py
│   ├── camera_params.py
│   ├── cam_params.json
│   ├── mit_io.py
│   ├── mit_process.py
│   └── mit/
│       ├── policy_controller.py
│       ├── leader_follower.py
│       ├── telemetry.py
│       ├── gravity.py
│       └── assets/
└── single_piper_leader_follower_teleop/
    ├── config.py
    └── controller.py
```

`SinglePiperLeaderFollowerTeleopController` 继承 `SinglePiperController`。虽然它控制
leader 和 follower 两台 Piper，最终操作对象仍是一台 follower arm，因此仍属于
single-Piper controller 家族。

当前约定：

- right/secondary arm 是 leader；
- left/main arm 是 follower；
- 两者都用 USB-CAN 稳定 hardware ID 解析，不能靠临时的 `can0/can1` 猜；
- packaged `leader_follower` MIT worker 是两个 CAN socket 的唯一 owner；
- teleop worker 在 100 Hz 控制，RURI/LeRobot observer 只读 localhost telemetry。

单臂 inference 继续使用 `SinglePiperController` 和 packaged
`policy_controller` worker。

## LeRobot collection

不 fork 或 vendor LeRobot。两个 adapter 仍是独立 distribution，但与 RURI 同 repo：

```text
integrations/
├── lerobot_robot_piper_mit/
└── lerobot_teleoperator_piper_mit/
```

它们不复制：

- CAN/arm discovery；
- camera serial/model discovery；
- `cam_params.json` 和 RealSense control/warm-up；
- Piper normalization ranges；
- telemetry receiver。

采集仍保持两个进程：

1. RURI teleop controller 进程拥有双 CAN 和 MIT loop。
2. `lerobot-record` 进程通过 RURI observer 读取 telemetry 与两路相机，不打开 CAN。

每个 30 Hz dataset frame：

1. latch 当前最新的同一个 MIT packet；
2. 从该 packet 生成 follower state 和 follower target action；
3. 读取当前最新的 head D435 与 wrist D415 RGB frame；
4. adapter 映射成旧数据集兼容的 `joint*.pos`、`gripper.pos`、`top`、`hand`。

LeRobot `frame_index` 是唯一 dataset 时间。camera/telemetry timestamp 不写进 dataset。
Dataset action 是 MIT loop 给 follower 的 target，不是 controller 回报的“实际执行值”。

episode、task、video、resume 和 Dataset v3 写入继续由官方 `lerobot-record` 管理。

## 依赖

官方 <https://github.com/agilexrobotics/pyAgxArm> 固定为：

```text
vendor/pyAgxArm @ 799b8412fbe8b9156bc9892d3dbeb2df7e98be71
```

使用 Piper：

```bash
git submodule update --init vendor/pyAgxArm
pip install -e '.[piper]'
```

`piper` extra 只包含 SDK 的普通 Python dependencies。RURI base 可在不拉 submodule、
不装 Piper dependencies 时导入。SDK 只在实际建立 Piper arm 时 lazy import。

LeRobot 不属于 RURI 的 `piper` extra，也不属于两个 adapter 的自动 dependency。用户
必须自己控制 LeRobot、PyTorch/CUDA 和 RealSense 的安装，然后把 adapter 安装进该环境。

## 为未来 Online RL 保留的唯一事实

当前不新建 Online RL 抽象。以后如果在同一 controller 上做 rollout：

- Scheduler 继续产生最终 target；
- Controller 继续直接 `send_action(target)`；
- 每个 30 Hz frame 使用该时刻最新 robot state 和两路 camera frame；
- 上层采集器以 LeRobot frame/step 为时间索引。

reward、done、human intervention、replay buffer 和 actor/learner 等需求等到真实任务明确后，
再在 RURI 之外或 integration/workflow 层设计，避免现在凭空增加实体。
