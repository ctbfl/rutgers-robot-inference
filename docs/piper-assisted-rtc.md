# 双臂人工接管与远程 Pi0.5 RTC

入口 `examples/client_dual_piper_rtc.py` 使用已真机确认的双臂圆圈 controller，
把本地 `CirclePolicy` 替换为现有 `inference_client.connect()` 返回的远程 policy。
不需要修改服务端、训练模型或增加 action 维度；也不依赖圆圈 FK/IK 或 SciPy。

```bash
python examples/client_dual_piper_rtc.py \
  --policy-endpoint tcp://172.16.68.130:5557
```

端口使用现有 RTC 推理的实际值。连接后在终端输入任务 prompt，保持与训练任务一致。
输入任务后，本地按注册 USB ID 识别两臂、上电并回 home，然后开始推理。
运行中按 q 暂停，输入新任务继续；暂停时输入 home 回零，switch PORT 切换同一主机的模型端口。
Esc 或 Ctrl-C 请求 HOLD 并退出，不再使用命令行 --prompt / --seconds。
`--simulate` 只替换硬件与相机，仍走真实服务器连接。不会自动清除电子急停。

`examples/client_single_piper_assisted_rtc_interactive.py` 是同一交互实现的原入口；
两个入口共用配置、控制器、RTC 调度器与终端逻辑。

## 代码职责

- `SinglePiperAssistedController` / MIT worker：本地两臂、相机、共享 reference、右臂接管、左臂插值。
- `AssistedScheduler(rtc_during_policy=True)`：复用 `_ActionTimeline` 和原 RTC 输入构造函数，
  管理异步推理、延迟对齐、旧 chunk 丢弃和接管边界。
- `InteractiveAssistedRTC`：终端交互、持续采样与调度，结束时请求 HOLD，再关闭 controller。
- `inference_client` / MessagePack / ZeroMQ / 服务端 wrappers：保持原实现。

底层设置与通过真机体验的圆圈一致：policy reference 上限 0.5 rad/s，总 reference 上限
4 rad/s，右臂 Kp=3/Kd=0.5，左臂 Kp=10/Kd=0.8，各自重力前馈持续有效。
双臂 home 也分别使用这些增益，避免右臂从 enable 进入 home 时突然变硬。
夹爪同步和力反馈保留。[交接公式和时序](piper-shared-reference.md)由 worker 执行。

## 网络协议不变

服务器收到：

| 字段 | 内容 |
| --- | --- |
| `type` | `metadata` 或 `infer`，由原 inference client 添加 |
| `observation.state` | 左臂实测关节和夹爪，原归一化的 7 维向量 |
| `observation.images.top` / `.wrist` | 原 top 与左臂 wrist RGB 图像 |
| `prompt` | 当前任务原文 |
| `observation.joint_effort` | 仅在 policy metadata 声明该输入时发送 |
| `context.rtc.prev_chunk_left_over` | 正常自主阶段旧 action chunk 的剩余部分 |
| `context.rtc.consumed_steps` | 同一 chunk 已消费的行数 |
| `context.rtc.estimated_inference_delay_steps` | 原有窗口估计的推理延迟 |

响应仍为 `action_chunk: (H, 7)` 和现有 `rtc.*` 状态。右臂 pose、alpha、authority、
intervention id 和左右插值状态都不加入网络输入。发送 RTC prefix 后，仍要求服务器
返回 `rtc.applied=true`；不接受把拒绝 RTC 的响应默默当作普通推理执行。

## RTC 在接管边界的处理

1. **正常 policy**：与普通 RTC client 一样，在旧队列继续执行时发送完整剩余 chunk。
   响应以请求时的 action timestep 为锚，丢弃推理期间已经过期的行。prefix 是待提交的
   policy 目标序列；底层 MIT 跟踪误差不会被伪装成额外的 action 维度。
2. **进入 blend / human**：worker 更新 reference revision，scheduler 清空旧队列，
   丢弃旧 revision 的在途结果。policy 继续基于最新左臂观测推理，暂不发送旧 RTC prefix；
   worker 在卸载和纯重力阶段不执行这些 proposal。
3. **recover**：以最新观测重新推理，无旧 prefix；新 proposal 可以参与恢复阶段的跟踪。
4. **重新进入 policy**：即使 worker 为保证 reference 连续没有再次修改 revision，
   scheduler 仍建立新的 RTC 序列，丢弃恢复阶段的未执行 proposal。第一条新 chunk 无 prefix，
   后续恢复正常 RTC 衔接。暂停、更换 prompt 或 checkpoint 同样清空历史。

圆圈示例仍使用 `rtc_during_policy=False`，保持其单步即时重规划方式；原固定增益导纳路径
也继续不用这些 prefix。这是客户端的调度策略，服务端协议不变。

## 数据与验证

不启动 recorder，不创建 episode、dataset、视频、effort 或 intervention 数据文件。
控制遥测与调度状态只暂存在内存中，状态变化输出到终端。

验证包括：正常阶段与原 RTCScheduler 的动作序列、prefix 和延迟丢帧对照；介入时丢弃旧
在途结果；恢复时重新建立 RTC；实际本地 TCP/MessagePack 服务、模拟双臂 MIT 子进程、
UDP 遥测、合成相机帧及完整接管/恢复联测，并检查没有生成数据文件。
这些测试验证协议与控制接线，不替代真实 Pi0.5 权重、相机和机械臂联测。
