# RURI Controller 讨论与实现记录（2026-08-30）

## 项目定位与命名

RURI（Rutgers Robot Inference）是用于 robot policy inference 的轻量分层。

- Python 包名继续使用 `ruri`。
- 命令名由 `rri-client` / `rri-server` 修正为 `ruri-client` / `ruri-server`。
- Client 代码按 `controllers/` 与未来的 `schedulers/` 分层；业务逻辑不集中到 `main.py`。
- Scheduler 今天没有实现。

远端 policy server 位于 `tcp://172.16.68.130:5555`。已成功进行一次 metadata 查询，确认 metadata 能提供 client 编写所需的信息，包括：

- 输入字段：`observation.state`、`observation.images.top`、`observation.images.wrist`、`prompt`
- policy：`pi05`
- config：`pi05_tight_insertion_E1`
- checkpoint：服务端的 `9999` checkpoint

metadata 当前只用于发现和说明，不作为严格运行时 schema 校验。遵循“如无必要勿增实体”的原则。

## 三层边界

### Server / Policy Wrapper

- 持有模型、checkpoint 和模型特定预处理。
- 处理 resize、crop、归一化、tensor layout 等 policy-specific 图像预处理。
- 使用 checkpoint 自己的统计信息做模型侧 normalization / denormalization。
- 不拥有本地机器人硬件，不决定本地动作执行时钟。

### Inference Scheduler

- 从 Controller 取得标准 RURI observation。
- 异步请求远端 inference，并维护 action chunk / action queue。
- 负责 replan、chunk 替换、action aggregation 和固定频率 control tick。
- Scheduler 是动作队列的唯一所有者。
- Scheduler 需要独立的实时 action executor，不能被相机读取或远程 inference 阻塞。
- 每个 control tick 从队列取一行动作，直接调用 `controller.send_action(action)`。

### Robot Setup Controller

- 拥有相机、CAN 和持久 MIT 控制进程的生命周期。
- Scheduler 与 Controller 保持同进程函数调用，不增加 RPC、序列化或中间动作队列。
- Controller 不决定 action chunk 何时请求、替换或聚合。
- `send_action()` 是立即下发的热路径：只做硬件 convention 转换和安全检查，然后交给 MIT loop。
- `connect()` 只连接 observation 设备并被动探测 CAN，不使能或移动机械臂。
- `start_arm()` 是显式运动门：启动、使能并执行参考 MIT controller 已有的安全回零流程。

目标实时链路：

```text
Policy inference ──补充 action chunk──▶ Scheduler action queue
                                          │ 每个 control tick 取一行
                                          ▼
                              Controller.send_action(action)
                                          │ 立即转换并发送
                                          ▼
                               persistent MIT loop @ 100 Hz
```

抽象层只能划分责任，不能给耦合实现增加额外的数据中转或调度延迟。

## Single Piper 硬件约定

本机 setup：

- 一个 D435，作为 head/top camera。
- 一个 D415，作为 wrist camera。
- 一个当前使用的 Piper arm，通过 SocketCAN 连接。
- 当前上电的是 left arm，同时是默认 main arm；right arm 未上电。

相机使用持续 streaming，再从后台流取得最新帧，而不是每次 observation 重启相机：

- `observation.images.top`：D435，序列号 `827112071860`
- `observation.images.wrist`：D415，序列号 `002422064073`
- 图像格式：原生 `640x480`、HWC、RGB、`uint8`
- Client 只做硬件 canonicalization，不做模型特定 resize/crop/normalization。

机器人 state/action 使用这台机器的数据采集 convention：

- 六个 joint：`[-100, 100]`
- gripper：`[0, 100]`
- 该转换属于 Controller，因为它与这台 Piper 的关节范围和采集方式绑定。
- 上传到 Server 前保留这一机器 convention；Server 再按 checkpoint 统计做模型侧处理。

## CAN 稳定身份与注册

不能使用 `can0` / `can1` 作为机械臂身份，因为它们可能随启动顺序、USB 端口或枚举顺序改变。

当前采用两阶段识别：

1. 从 sysfs 读取 USB-CAN 的 vendor、product 和硬件 serial，生成稳定 hardware ID。
2. 被动监听并要求出现完整 Piper feedback signature（状态帧 `0x2A1` 加完整 joint feedback IDs）。

注册结果：

| 稳定 CAN hardware ID | Arm side | Role | 当前临时接口 |
| --- | --- | --- | --- |
| `usb:1d50:606f:0042002F4759530820353131` | left | main | `can1` |
| `usb:1d50:606f:002B00464759530920353131` | right | secondary | `can0` |

默认流程不是先假设 left，而是：

1. 找到唯一具有有效 Piper feedback 的总线。
2. 用稳定 hardware ID 反查 left/right 和 role。
3. 两个 Piper 同时活跃、没有活跃 Piper 或适配器未注册时均拒绝猜测。

限制：这里稳定的是 USB-CAN 适配器身份，不是 Piper 本体序列号。它不受 `can0` / `can1` 重命名或更换 USB 端口影响；如果物理更换适配器或把左右适配器互换，则需要重新注册。当前 Piper 协议/固件信息没有提供一个已经确认可用且全局唯一的机械臂本体序列号。

## 已实现 Controller

主要文件：

- `src/ruri/client/controllers/robot_setup_controller.py`
- `src/ruri/client/controllers/single_piper/config.py`
- `src/ruri/client/controllers/single_piper/discovery.py`
- `src/ruri/client/controllers/single_piper/hardware_registry.py`
- `src/ruri/client/controllers/single_piper/normalization.py`
- `src/ruri/client/controllers/single_piper/mit_io.py`
- `src/ruri/client/controllers/single_piper/mit_process.py`
- `src/ruri/client/controllers/single_piper/single_piper.py`
- `tests/test_single_piper.py`

当前公开能力：

- 自动发现唯一 D435 与 D415，并按型号赋予固定角色。
- 使用稳定 USB-CAN ID 和 Piper feedback 双重确认 arm。
- `get_camera_observation()` 获取已经 streaming 的最新相机帧。
- `get_observation()` 返回标准 RURI state/top/wrist 字段。
- 显式启动并持有参考项目 `piper_teleop_agx/mit_policy_controller.py`。
- `send_action()` 接受一行机器 convention action，立即转换为 rad/m 并发送给 MIT 进程。
- 本地 command/telemetry 使用 loopback UDP；MIT 子进程是唯一实际 CAN socket owner。
- command watchdog 超时后执行参考控制器的 hold/home 安全恢复。

Controller 当前不维护 action chunk 或 action queue。以后也不应在 Controller 内增加 Scheduler queue。

## 今天的真实硬件测试

### 相机

- D435 与 D415 均成功持续 streaming。
- 60 对图像耗时约 `1.953 s`，约 `30.72 Hz`。
- head/wrist 各得到 60 个不同帧。
- 最大双相机时间戳偏差约 `14.351 ms`。
- 输出均为 `(480, 640, 3)`、`uint8`。
- 测试后两个相机均正常释放。

### CAN 探测

- 两个接口均配置为 `1 Mbps`，总线状态 `ERROR-ACTIVE`，没有 CAN bus error。
- 当前 `can0` 没有 Piper 帧；当前 `can1` 有完整 Piper feedback。
- 稳定 ID 自动推断结果为 `left/main`，临时接口为 `can1`。
- 一次 1 秒被动测试在 `can1` 收到 3041 帧。

### MIT 控制流

- 显式启动后进入 `CAN_CTRL` / `MOVE_MIT`，完成约 5 秒安全回零。
- 得到完整 state 和两路真实图像 observation。
- 将测得的当前位置作为目标，以 30 Hz 连续注入 1 秒，共 30 个 scheduler-side action tick；没有故意要求额外运动。
- MIT 日志记录共 129 个底层 command 周期、0 overruns。
- 停止注入后，约 `0.310 s` 触发 command watchdog，随后完成安全回零。
- 最终 home residual 为 `0.016 rad`。
- MIT 子进程已经退出，相机已经释放，没有遗留控制进程。

安全状态说明：参考 MIT controller 退出后不会擅自 torque-off；left arm 仍上电并保持最后的 MIT 指令。下次操作前仍应按真实硬件动作对待并确保有人在机械臂旁。

## 自动化测试状态

- 8 个 Single Piper 单元测试全部通过。
- `compileall` 通过。
- 覆盖相机唯一性、CAN Piper signature、稳定 ID 跨接口重命名、双 Piper 歧义拒绝、未注册适配器拒绝、归一化 round-trip、标准 observation 和 action 注入。
- `.venv` 用于轻量单元测试，但没有安装 LeRobot/RealSense 可选依赖。
- 真实相机测试使用 `/home/omen/miniforge3/envs/lerobot/bin/python` 并设置项目 `PYTHONPATH=src`。

## 明天继续的议题

1. 先讨论 Scheduler 的最小接口，不直接开始堆实现。
2. Scheduler 内将远程 inference/refill 与实时 action executor 分开。
3. 明确 control frequency、队列低水位、action chunk 替换和耗尽策略。
4. 保持 Scheduler→Controller 为同进程直接调用，Controller 不增加动作队列。
5. 优化 `send_action()` 热路径，减少不必要的复制、字典和临时对象。
6. 增加与耦合参考实现可直接比较的指标：send latency、control tick jitter、command age、missed deadline 和 MIT overruns。
7. 再决定是否需要 observation timestamp / action sequence；除非它解决实际同步或诊断问题，不新增协议实体。
8. 处理最终 CLI 入口：源码配置已改名为 `ruri-client` / `ruri-server`，但目前没有实现 Scheduler/client `__main__`，现有 `.venv/bin` 也尚未重新安装刷新。

当前所有改动仍在本地工作区，尚未提交。
