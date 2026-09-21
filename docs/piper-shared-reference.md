# 固定 MIT + Policy + Human Intervention：第一版

当前验收对象只有右臂 leader。新入口 `examples/client_single_piper_shared_circle.py` 设置
`leader_only=True, camera_enabled=False`，按注册 USB ID 连接右臂；不连接左臂、不打开相机。
普通 teleop、单臂 policy 和数据收集仍使用各自原有入口。

## 逻辑放在哪里

- `single_piper_assisted/reference.py`：纯 NumPy 数学状态，维护六维 reference、velocity 和 α，无 SDK、相机或 policy 依赖。
- `single_piper_assisted/control.py`：在 MIT 周期调用它，负责会话、模式、action 新鲜度和 reference revision。
- `single_piper/mit/assisted.py`：硬件读取和发送。`LeaderRig` 只创建一个右臂 driver，使用 V189 握手和右臂重力标定。
- `schedulers/assisted.py`：异步推理和丢弃旧 chunk，不计算 torque 或 α。
- `examples/piper_circle_policy.py`：普通 `PolicyWrapper`，只消费最新实测关节状态并输出位置 proposal。SciPy FK/IK 只属于示例。

已有 assisted 双臂适配器复用这个 reference 核心；本轮没有做双臂真机验收。

## 每个周期

默认 100 Hz，在物理关节坐标（rad）下计算：

```text
interaction = Kp_L * (q_L - r) - bias
effective   = sign(interaction) * max(abs(interaction) - deadzone, 0)
v_h         = effective / B
alpha       = timed_authority(interaction, dt)  # 独立滤波、持续确认和时间曲线见下文
v_p         = clip(K_r * (policy_proposal - r), -v_policy_max, v_policy_max)
v           = clip((1 - alpha) * v_p + alpha * v_h, -v_max, v_max)
r_next      = clip(r + dt * v, joint_lower, joint_upper)
v_applied   = (r_next - r) / dt
```

interaction 使用上一周期已发送的 r。时间抖动时积分 dt 上限为两个 MIT 周期；关节边界投影后重新计算
`v_applied`，避免到边界后仍发送朝边界外运动的速度。

底层每次发送 `p_des=r_next, v_des=v_applied, kp=3, kd=0.5, t_ff=gravity(q)`。
不根据 α 改 Kp/Kd，不再额外叠加软件 `-d*qdot`。SDK 固件接口仍是已适配的 V189；扭矩相关参数沿用
当前 pinned SDK 与现有 calibration 的 legacy 单位，不能当作独立力传感器测量。

通用 controller 默认每关节 deadzone=0.08、bias=0、B=2，K_r=2/s，
v_max=v_policy_max=0.15 rad/s。当前右臂圆圈示例使用 v_max=4 rad/s、v_policy_max=0.5 rad/s、B=0.1。

### 人工权重的时间曲线

对去除 bias 的关节偏差 `interaction / Kp_L` 做时间常数 **50 ms** 的低通滤波，
只用于 authority 判断，不在快速 admittance 支路加入这个滤波延迟。

1. **确认**：任一关节的滤波偏差绝对值连续超过 **5° 达 0.5 秒**。
   确认期间保持 **policy**、α=0，不丢弃 policy chunk；条件中断立即清空确认计时。
   判断每个关节的最大值，不把多个小偏差的向量范数当成 5°。
2. **blend**：确认后，从当前 α 连续线性上升，累计 **0.5 秒有效介入证据**才到 1。
   维持条件低于进入门槛；信号消退后暂停上升，不能继续自动升满。
3. **human**：α 严格等于 1，policy 对 reference 的贡献严格为零，推理仍在后台继续。
4. 没有有效介入证据后，前 **0.5 秒**保持当时的权重；随后进入 **recover**，
   再用 **0.5 秒**从该权重连续降到零。未升满时也从实际权重下降，不能跳到 1。
5. **recover** 期间重新介入也需要通过 5° / 0.5 秒确认，确认后从当时权重重新上升。

配置为 `human_trigger_angle_deg=5`、`interaction_filter_s=0.05`、`human_confirm_s=0.5`、
`human_takeover_s=0.5`、`human_release_delay_s=0.5`、`human_release_s=1.0`。
从滤波信号超过进入门槛到完全接管，连续有效证据下约需 1 秒；滤波还会引入短暂检测延迟。

例如完成接管后，若人工静止时间为 s：

```text
s <= 0.5 s:        alpha = 1
0.5 < s < 1.0 s:   alpha = (1.0 - s) / 0.5
s >= 1.0 s:        alpha = 0
```

interaction 使用弹簧形变估计，不是手部接触传感器。维持判断为滤波后的 interaction 经过原有
deadzone 后，范数超过 `interaction_release=0.02`。5° 不改变跟手死区 0.08 或 B=0.1。
机械臂自身运动不再作为维持接管的证据。若拖动过轻、reference 已追上手而形变低于维持门槛，
同样会开始释放计时；仅凭编码器不能完全区分轻触、松手与模型残差，手感仍需右臂实测。
这组进入门槛在原 22 秒无人日志上没有触发；日志回放不能替代修改后的无人真机验收。

夹爪单独处理，不把米和关节 torque proxy 混入同一个 norm。圆圈 policy 保持最新实测夹爪开度；
本示例不是夹爪干预/力反馈验收。

## Policy 始终使用最新状态

自动干预不会退出 POLICY，也不会停推理。权重阶段变化时 reference revision 增加，未执行旧 chunk、
旧 revision 的在途结果失效；恢复阶段重新从最新状态查询，不执行人工独占期间保留的旧 proposal。
人工独占时 policy 贡献为零，上升和下降期间新 proposal 按 1−α 参与混合。policy observation 最长有效期默认 1 秒。
等待新推理的 keepalive 只维持通信，不刷新旧 action 的有效期。

第一版不向模型发送 RTC 已执行前缀：policy proposal 经过限速和人类混合后不等于实际执行 reference，
不能把 proposal chunk 冒充已执行轨迹。新返回的无前缀 chunk 从下一个可用调度时刻执行，避免短 chunk
因推理延迟在返回时已经过期。普通 RTC scheduler 不受这个 assisted 模式分支影响。

## 圆圈示例

基座坐标 +X 向前、+Z 向上。TCP 暂用 URDF 的夹爪中点，即 link6 局部 +Z 的 135.8 mm 处；
这是几何模型值，安装额外工具时用 `--tcp-offset X Y Z` 改为实际工具变换。

默认 home TCP 约 `[0.19143, 0, 0.22502] m`，圆心约 `[0.24143, 0, 0.37502] m`。
画基座 YZ 平面内半径 10 cm 的竖直圆，圆周角速度 0.5 rad/s（约 12.57 秒一圈）。
示例通过 `--reference-speed` 将融合后的关节 reference 速度上限设为 4 rad/s；
`--policy-speed` 独立限制 policy 支路为 0.5 rad/s。MIT 发送层的速度裁剪同步设为 4 rad/s。姿态是软约束：上方仰起、下方略向下看；
朝向的 yaw 根据本次 TCP 近端目标计算，避免 home 腕部奇异位置出现不必要的绕腕解。
IK 从当前实测 q 开始，并保留很小的关节连续性正则项。

圆周目标的相位按单调时钟持续前进。每次推理从最新实测 TCP 沿直线朝当时的圆周目标迈一小步，
再解 IK。默认 proposal 的 TCP 步长不超过 50 mm（0.10 m/s × 0.5 s lookahead）。这个速度是
policy 参数；最终电机跟随的是关节空间限速/融合后的 r，实际 TCP 路径不保证严格直线。

先检查从 home 到圆的接近路径、整圈 FK/IK 误差、相邻关节解、圆周关节速度与右臂 gravity 前馈上限。
检查不涉及碰撞几何；这条新路径仍需真机观察。程序运行中不会停下看图。

## 运行

使用已安装 Piper 依赖的环境；额外安装 `pip install -e '.[piper-circle]'` 提供示例的 SciPy。
在 repo 根目录运行：

```bash
# 离线几何检查：默认模式，不创建 controller，不连接 CAN。
PYTHONPATH=src:vendor/pyAgxArm python examples/client_single_piper_shared_circle.py

# 仿真完整进程/UDP/推理链路；第 3 秒约束右臂 J1 两秒，随后释放。
PYTHONPATH=src:vendor/pyAgxArm python examples/client_single_piper_shared_circle.py \
  --simulate --simulate-intervention --seconds 35

# 真机：右臂上电，MIT 回 home，然后永续执行，Ctrl-C 请求 HOLD 并退出。
PYTHONPATH=src:vendor/pyAgxArm python examples/client_single_piper_shared_circle.py --execute
```

`--seconds 60` 可限制 policy 测试时长（从 home 完成后计时）；默认 0 表示永续。
`--radius` 单位米、`--period` 单位秒，`--reference-speed` 是融合后的关节 reference 速度上限（rad/s），`--policy-speed` 是 policy 支路上限。默认不绑定某个 canN；指定 `--leader-can-interface` 时仍校验右臂身份。
真机入口复用工作站 writer lock 并检查其他已知控制进程，避免与现有 teleop 同时发送。

正常结束持有当前位置并退出进程，不关闭扭矩；退出后不再持续重算主机重力前馈。运行中硬件故障沿用
MIT recovery：有健康反馈时先 HOLD、再低速 home；无法恢复时发送 electronic stop，不自动重启。
保留每路反馈、状态、速度、reference tracking error（0.35 rad）和扭矩上限。策略异常先请求 HOLD。

JSONL 默认写到 `logs/right_shared_circle_时间.jsonl`，包含配置、几何检查、q、r、rdot、α、interaction、
policy 时间/圆周目标、推理结果采纳/丢弃原因及 MIT telemetry。
终端显示 `authority` 阶段、α、进入确认计时 `confirm`、上升计时 `rise` 和静止计时 `quiet`；
日志增加 `filtered_interaction_error_rad`、`intervention_trigger_active`、`human_rise_s`，另有 `human_exclusive`
与加权后、最终限速前的 `applied_policy_velocity`，人工独占时这个向量严格为零。右臂实测状态来自 `leader`；`follower.q`
保持 null，并有 `leader_only=true, follower_present=false` 标记。action 记录实际应用 reference，而非未执行 proposal。

## 首版离线验证（3 cm、0.15 rad/s 配置）

完整测试覆盖数学公式、固定 MIT 参数、右臂单 driver、启动/回 home、通信与反馈故障、action 失效、
短 chunk 的持续更新，以及圆圈 FK/IK。35 秒仿真中注入两秒干预，α 最高约 0.96，释放后回到 0；
reference 速度不超过 0.15 rad/s，旧 revision 结果被丢弃。接近圆后模拟 TCP 对移动目标的误差约 4.3 mm。
这里的仿真 rig 是简化运动模型，不验证电机动力学、实际重力精度或接触稳定性；还需右臂真机调试。

当前默认已调整为半径 10 cm、融合后的关节 reference 上限 4 rad/s、policy 支路上限 0.5 rad/s、圆周角速度 0.5 rad/s；
新路径已通过离线 IK 检查，圆周所需最大关节速度约 0.207 rad/s。上面的 35 秒仿真统计属于初版参数。

右臂示例的 `--admittance-damping` 默认现为 0.1，相比 B=2，同样有效 interaction 产生的
人支路速度提高 20 倍。充分人工接管时，0.16 rad（约 9.2°）单关节形变即可给出 4 rad/s；
接近手的位置后自然减速至 deadzone，4 rad/s 是上限，不是强制最低速度。Kp/Kd 保持固定；α 使用上述时间曲线。
这是 reference 数学响应的调整，实际拖动手感仍以真机体验为准。修改后重启示例生效。
