# 双臂人工接管与远程 Pi0.5 RTC

入口 `examples/client_dual_piper_rtc.py` 使用双臂圆圈 controller，
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

## 夹爪触发接管

旧关节偏移入口与交互式 RTC 已存档于 `6bec532`。当前两个双臂入口默认
`human_trigger="gripper"`，**仅右夹爪闭合触发接管**，关节偏移不再触发。

1. 自主跟随时右夹爪先用 0.2 N 张开，到位后以最大 0.05 N 的弱回弹保持全开目标，左夹爪继续执行 policy。
   全开读数在 2 mm 容差内稳定 50 ms 后准备好；终端显示 ready。
2. 实测开度 ≤ **90 mm** 即触发（全开目标 100 mm），无速度或持续时间要求，
   不使用历史峰值，可缓慢捏合，无需完全闭合。
   触发动作不会传给左夹爪；左夹爪目标固定在触发时实测开度。
3. `gripper_align`：右夹爪用 `grip_force` 主动对齐左夹爪实时开度，命令速度
   20 cm/s（独立 `gripper_alignment_speed`，不使用 policy 的 4 cm/s 夹爪速度）。
   双臂关节仍共用 reference，现有 reference 速度在 0.1 s 内减至零，
   暂停执行 policy proposal。右夹爪命令及实测开度都进入 2 mm 容差并稳定
   50 ms 后，**直接 alpha=1、右臂纯重力**，不经过原 0.5 s 卸载。
   对不齐时保持对齐阶段并显示状态，不按计时器盲目进入 gravity，也不因此 abort。
4. 左臂仍用 0.5 s 插值接到右臂实时关节位置；夹爪同步、力反馈恢复。
   右臂静止 0.5 s、左臂 reference 追上后，再用原 0.5 s 恢复跟随。
5. 恢复 policy 时，**只有右夹爪先用 0.2 N 重新全开，再切回弱回弹**，左夹爪继续当前 policy 开度。
   必须重新达到全开才接受下一次闭合，自动对齐和张开不会触发接管。

夹爪打开、对齐和接管阶段写入本地状态，不增加远程 observation/action 字段。
这套夹爪接管、重新张开与 20 cm/s 对齐已通过双臂圆圈真机体验；两个远程入口
同样启用 `gripper_max_m=.1`，共用该实现。双臂圆圈的完整控制边界与参数见
[圆圈说明](piper-shared-reference.md)。

## 夹爪量程与已有 policy

本地夹爪命令及反馈按 **0–100 mm ↔ policy 0–100** 双向映射。采用此前确认的
名义量程线性迁移：同一个归一化值表示相同开合比例，例如 policy 输出 `10`，
旧固件对应 8 mm，新入口对应 10 mm；当前实测 10 mm 也回传 `10`。
输出 `100` 对应 100 mm，不再在旧 80 mm 上限截断。

转换位于本地 controller；左臂 observation、action、home 及夹爪命令使用同一量程。
RTC 剩余 chunk 保持 policy 的原归一化值，不额外缩放。右夹爪的全开与手势状态
不会替代左臂传给 policy 的夹爪读数。2026-09-21 全局 nominal、两臂夹爪 clamp
及单臂/teleop 的 calibration 副本也已统一为 0–100 mm；普通推理、遥操作和采集
使用同一映射。RTC 的局部适配按当前 nominal 计算，默认不再额外缩放。
历史数据与模型归一化统计不重写；两臂文件保留旧测量值作为来源记录。

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
| `observation.state` | 左臂实测 7 维向量；关节归一化不变，夹爪按本地 0–100 mm 映射到 0–100 |
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
2. **进入 gripper_align / human**：worker 更新 reference revision，scheduler 清空旧队列，
   丢弃旧 revision 的在途结果。policy 继续基于最新左臂观测推理，暂不发送旧 RTC prefix；
   worker 在夹爪对齐和纯重力阶段不执行这些 proposal。
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
另验证远程 action `10` 实际产生 10 mm 的模拟左夹爪目标与反馈、RTC prefix 不缩放，
交互入口连续两次接管后右夹爪重新张开至 100 mm、左夹爪保持任务开度、RTC 恢复。
这些测试验证协议与控制接线，不替代真实 Pi0.5 权重、相机和机械臂联测。
