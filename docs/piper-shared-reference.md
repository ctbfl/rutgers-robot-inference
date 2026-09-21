# 圆圈：Policy、右臂纯重力交接与左臂跟随

单臂版本已经过用户真机体验确认。入口 `examples/client_single_piper_shared_circle.py` 使用
`leader_only=True, camera_enabled=False`，按注册 USB ID 连接右臂，不连接左臂、不打开相机。
进入时启用 MIT 并回 home。普通 teleop、单臂 policy、数据收集和双臂 assisted 使用各自原有路径。

旧版固定阻抗、通过导纳推动共享 reference 的圆圈实现已存档于 commit `b73950b`。
新双臂入口 `examples/client_dual_piper_shared_circle.py` 设置 `gravity_handoff=True`，复用同一个
右臂交接机制并增加左臂插值。左臂复用原 single Piper 的固定 MIT 增益和重力前馈。
未开启重力交接选项的双臂 assisted 继续使用 `SharedReference`。

## 代码边界

- `single_piper_assisted/gravity_handoff.py`：纯数值的进入确认、权重过渡、静止窗口和 reference 状态。
  复用 `SharedReference` 的状态/状态报告接口，不调用它的导纳更新逻辑。
- `single_piper_assisted/control.py`：在 MIT 周期调用数值控制器，管理会话、watchdog、action 新鲜度与 revision。
- `single_piper_assisted/follower_handoff.py`：左臂从共享目标接到实时右臂读数的插值与限速。
- `single_piper/mit/assisted.py`：`LeaderRig` 打开一个右臂 driver，使用 V189 握手、右臂重力标定及 MIT 发送。
- `schedulers/assisted.py`：异步推理与旧 chunk 丢弃，不计算 alpha 或电机力矩。
- `examples/piper_circle_policy.py`：使用最新实测关节状态的普通 policy。SciPy FK/IK 只属于示例。

## 电机控制律

MIT 始终运行在同一个模式。重力前馈不退出；alpha 改变 leader 阻抗和拖动阻尼：

```text
tau = gravity(q)
    + (1-alpha) * [Kp * (r-q) + Kd * (v_ref-qd)]
    - alpha * D_h * qd
```

默认 `Kp=3, Kd=0.5, D_h=leader_kd=0.2`。

- alpha=0：原来的 policy 位置/速度跟踪。
- 0<alpha<1：MIT `kp=(1-alpha)*Kp, kd=(1-alpha)*Kd`，
  `t_ff=gravity(q)-alpha*D_h*qd`；前馈经过原有扭矩边界检查。
- alpha=1：直接调用已验证的 `leader_follower.send_leader`，SDK `kp=kd=0`、
  `v_des=0`，只输出重力补偿和软件阻尼。没有位置弹簧、导纳死区或 B 参数。

SDK 中的 p_des 字段仍存在，但零刚度时不产生位置恢复力。此时内部 r 只记录实测 q，
`reference_active=false`；shadow reference 的变化不是位置伺服命令，`reference_qd=0`。
反馈速度上限仍然有效。SDK 与 calibration 继续使用 legacy 扭矩单位，不是独立力传感器测量。

## 进入和退出

默认控制周期 100 Hz。权重过渡使用 `smoothstep(u)=3u²-2u³`，首尾变化率为零。

1. **policy / 进入确认**：对去 bias 的 `q-r` 做 50 ms 低通滤波。任一关节偏差连续
   超过 5° 达 0.1 秒才开始交接；中断则清空确认计时。确认期间 policy 正常运行，alpha=0。
   判断每关节绝对偏差的最大值，不把多个小偏差相加，也不把 policy 自己的运动当介入。
2. **blend / 卸掉阻抗**：用 0.5 秒把 alpha 从当前值平滑升到 1。原 reference 速度
   同时平滑减至零，不再追逐新的 policy 目标。确认完成后不再用逐渐消失的弹簧力判断是否继续卸载。
3. **human / 纯重力**：alpha=1，policy 继续推理但不控制电机。每个关节在最近 0.5 秒内的
   `max(q)-min(q)` 均不超过其小阈值，才确认静止。默认阈值均为 0.1°。
   统计从进入纯重力阶段开始，不把此前卸载阶段算成静止。缓慢持续运动也会累积到窗口范围中。
   手握着不动或颤动低于门槛，同样计入恢复；这不是手部接触检测。
4. **recover / 恢复**：刚开始时 alpha 仍为 1，将 r 对齐当前实测 q、reference velocity 清零，
   使旧 revision 和旧 action 失效。随后用 0.5 秒把 alpha 降到 0，policy 基于当前状态重新推理，
   reference 按 policy 速度限制向新目标移动。恢复结束保留这条新轨迹，不再额外清空它。
5. **恢复途中重新介入**：重新走 5° / 0.1 秒的偏差确认，然后从当前 alpha 开始卸载。
   此时不使用关节运动窗口维持人工控制，避免 policy 的运动打断自身恢复。

显式操作员 HUMAN 请求跳过进入确认，但仍使用 0.5 秒卸载过程；显式 HUMAN 不自动恢复。
HOLD、home 和故障处理仍沿用已有生命周期，不被自动交接覆盖。

主要配置：`human_trigger_angle_deg=5`、`human_confirm_s=0.1`、`interaction_filter_s=0.05`、
`human_takeover_s=0.5`、`human_release_delay_s=0.5`、`human_release_s=1.0`、
`human_quiet_range_deg=(0.1,)*6`。最后一个字段支持逐关节阈值；示例 `--quiet-angle` 同时设置六个值。
0.1° 是初始静止判据，历史重力试验中存在关节反馈完全不变的窗口；它尚未经过新版本多姿态实测验收。
配置中的 `admittance_damping`、`interaction_deadzone`、`interaction_release` 留给双臂旧控制器，
重力交接路径不使用它们。

## 双臂交接的细节

单一 policy 从左臂最新实测状态推理，100 Hz worker 生成 reference；两臂在同一个 worker 周期
发送各自的 MIT 命令。SDK 写入顺序执行，并非硬件时钟级同步。
两臂使用相同 joint coordinates，各自在自己的基座坐标系中画同一个圆；不是两个 TCP 追逐
世界坐标中的同一个点。圆圈示例仅支持 identity joint mapping。

| 阶段 | 右臂 leader | 左臂 follower |
| --- | --- | --- |
| policy / 进入确认 | 跟踪共同 r | 跟踪共同 r |
| blend：0.5 s 卸载 | r 速度平滑刹停，阻抗随 alpha 下降 | 同一 r 与速度，保留固定阻抗 |
| human 前 0.5 s | 原 teleop 的纯重力补偿与阻尼 | 从最后已发送的共同 r 插值到实时 qL |
| human 后续 | 纯重力补偿与阻尼 | 每周期跟随实时 qL，受关节范围及 4 rad/s 限制 |
| recover：0.5 s | 阻抗平滑恢复，跟随共同 r | 固定阻抗，跟随共同 r |

左臂始终使用原 single Piper 的 `follower_kp=10, follower_kd=0.8`、
`calibration_follower.json` 与同一个 `send_position_impedance`：

```text
tauF = gravityF(qF) + KpF*(rF-qF) + KdF*(vF-qdF)
```

启用、home、运行、HOLD 和故障回 home 均保留这个前馈。左臂不会进入右臂那样的零刚度
手拖模式；“保留重力前馈”和“卸掉刚度进入纯重力模式”是两个不同的概念。
此前直接把左臂 t_ff 清零会让位置误差承担重力支撑，已撤销这个偏离原控制器的改动。
两臂收到相同目标不代表实测关节位置完全一致，仍存在正常跟踪误差。

记进入 human 的时刻为 t0，**上一周期发给左臂的目标**为 r0。第一帧仍发送 r0。
接下来令 `u=clip((t-t0)/0.5, 0, 1)`，采用五次平滑函数：

```text
s(u) = 10u^3 - 15u^4 + 6u^5
rF_des(t) = (1-s(u))*r0 + s(u)*clip(qL(t), common_bounds)
rF(k+1) = rF(k) + clip(rF_des-rF(k), -vmax*dt, vmax*dt)
vF(k+1) = (rF(k+1)-rF(k))/dt
```

插值终点每周期更新，不冻结 t0 时的右臂读数。五次函数首尾一阶、二阶导数为零；
配合此前共享 r 的刹停，平滑进入跟随。左臂 MIT 同时收到实际插值位置与对应速度，
不会错误地使用右臂纯重力阶段的零 reference velocity。
0.5 秒是插值时长，不是实测左臂必须精确到位的保证。若限速生效，继续有界追赶，
不跳目标也不因 gap 本身 abort；到达时间相应延长。日志 `follower_handoff_limited` 会标记限速。

**恢复同样处理 gap**：右臂静止 0.5 秒后，还需插值完成、左臂已发 reference 与右臂有界目标
逐关节相差不超过静止门槛（默认 0.1°）。满足后把共同 r 设为左臂最后已发 reference，
此刻右臂 alpha 仍为 1，故不会突然产生位置恢复力；左臂目标也没有跳变。
随后丢弃旧 revision，从当前实测位置重新 query policy，两臂重新收到共同 r、共同速度，
右臂在 0.5 秒内恢复阻抗。若左臂尚在限速追赶，右臂继续纯重力，等待跟上后再恢复。
判据针对命令的连续性，不要求左右实测 q 完全一致；MIT 的正常跟踪误差可以存在。

恢复中再次介入仍使用 5° / 0.1 秒确认；下一次左臂插值重新从当时已发送目标开始。
圆周虚拟点和 policy 推理在所有阶段持续运行。夹爪沿用原双臂同步与力反馈通道，不参与六关节插值。

```bash
# 默认只检查双方关节范围、IK 路径及双方重力标定。
PYTHONPATH=src:vendor/pyAgxArm python examples/client_dual_piper_shared_circle.py
# 不连接硬件，演练完整进程与交接。
PYTHONPATH=src:vendor/pyAgxArm python examples/client_dual_piper_shared_circle.py \
  --simulate --simulate-intervention --seconds 9
# 按注册 USB 身份连接左右臂、上电、双臂回 home，再开始永续圆圈。
PYTHONPATH=src:vendor/pyAgxArm python examples/client_dual_piper_shared_circle.py --execute
```

如果上一次故障留下 `EMERGENCY_STOP(0x1)`，普通启动在 enable 前报错，不会自动复位。
需要复位后重新启动时，显式运行 `--execute --reset-stop`：只对当前报告急停锁存的臂调用
一次 SDK `reset()`，等待时间戳前进且状态 NORMAL、err_code=0，再按原流程启用、回 home。
它不重试运行故障、不复位健康臂；复位未获健康反馈则停止启动。启动错误附带 worker 日志，
示例同时把异常写入 JSONL 的 `failure` 行，避免只剩 setup 而丢失首次失败原因。

双臂日志为 `logs/dual_shared_circle_时间.jsonl`。新增 `follower_reference_q`、
`follower_reference_qd`、`follower_handoff_progress`、`follower_handoff_limited`；
human 时 `reference_q` 是右臂 shadow pose，左臂命令必须查看 `follower_reference_q`。
`reference_active=false` 在此仅表示右臂位置弹簧卸载，左臂仍在执行跟随。
终端 `left_blend` 显示插值进度，`limited` 显示是否触发 reference 限速。

## Policy 与 reference

自动交接不退出外层 POLICY 会话，不停止推理。进入 blend、human、recover 时 revision 改变，
旧 chunk 和在途结果失效。人工阶段的 policy proposal 不执行；恢复阶段重新从最新实测位置推理。
policy observation 最长有效期默认 1 秒，keepalive 不会刷新旧 action 的有效期。

圆圈入口使用 `ruri_gravity_handoff_v1`，保持不带 RTC prefix 的单步重规划。
新的远程入口在正常 policy 阶段使用原 RTC chunk 衔接，在介入及恢复阶段清空 prefix，
见 [远程 RTC 接入说明](piper-assisted-rtc.md)。双臂旧 `ruri_shared_reference_v1` 路径保持原调度方式。
夹爪是独立的位置/力通道；本轮不是夹爪力反馈验收。

## 圆圈与运行

圆心为 home TCP 向前 5 cm、向上 15 cm，约 `[0.24143, 0, 0.37502] m`；
基座 YZ 平面半径 10 cm，角速度 0.5 rad/s（约 12.57 秒一圈）。TCP 暂用 URDF 夹爪中点，
沿 link6 +Z 偏移 135.8 mm，可用 `--tcp-offset` 修改。姿态为软约束，上方抬头、下方低头。

圆周目标的相位始终前进。policy 从当前实测 TCP 沿直线朝当时目标迈出一小步再解 IK，
默认 TCP lookahead 速度 0.10 m/s、步长不超过 50 mm。policy 关节 reference 速度上限 0.5 rad/s，
发送层 reference 上限保持 4 rad/s；纯重力拖动不靠 reference 速度限制跟手，实测关节速度检查仍为 4 rad/s。
关节空间执行不保证 TCP 走精确直线。离线 IK 审查不等同碰撞检查。

离线 IK 审查保持 0.05 mm 的位置残差门槛；运行时允许 0.5 mm。人工拖到其他姿态后，
若当前直线步长的 IK 不收敛，圆圈 policy 依次用 1、1/2、1/4、1/8 步长重试。
都无解时返回当前实测姿态对应的有界 proposal，标记 `ik_status=waiting_ik`；这是本次不再前进的
普通 policy 结果，不请求 worker HOLD，不打断纯重力拖动。下一次 query 继续从最新状态尝试，
圆周相位仍持续前进。它不保证从所有姿态都能找到绕开不可达区域的路径。
非有限观测、意外程序异常和硬件故障仍沿用原有故障处理。

```bash
# 默认只离线审查 FK/IK，不创建 controller，不连接 CAN。
PYTHONPATH=src:vendor/pyAgxArm python examples/client_single_piper_shared_circle.py

# 完整进程/UDP/推理仿真；第 3 秒将 J1 偏移 0.2 rad 并约束 2 秒。
# 约束期间不移动也会触发静止恢复，符合“握着不动也算静止”的定义。
PYTHONPATH=src:vendor/pyAgxArm python examples/client_single_piper_shared_circle.py \
  --simulate --simulate-intervention --seconds 9

# 右臂上电、MIT 回 home、永续圆圈，Ctrl-C 请求 HOLD 并退出。
PYTHONPATH=src:vendor/pyAgxArm python examples/client_single_piper_shared_circle.py --execute
```

当前环境 Python 为 `/home/omen/miniforge3/envs/lerobot-061/bin/python`，额外依赖 SciPy 在
`piper-circle` extra 中。`--seconds 60` 可限制 home 后的运行时间，默认 0 为永续。
`--quiet-angle 0.1` 设置静止窗口范围，单位度。修改后需重启脚本。

入口持有现有硬件 writer lock，并按 USB 身份校验右臂。保留每路反馈、状态、速度与前馈扭矩检查；
右臂 `LeaderRig` 不再因 q 与 reference 偏差超过 0.35 rad 而 abort，也没有替换成加权门槛。
双臂重力交接也不使用该 raw-error abort；未开启重力交接的双臂原有控制器仍保留原检查。
运行故障沿用 MIT hold/home recovery，不能健康恢复时发送 electronic stop；不自动重启。
正常退出持有当前位置再断开，不关闭扭矩；退出后没有持续运行的主机重力循环。
调度器关闭时若 worker 已在故障恢复，不再发送 HOLD；恢复状态切换造成的清理超时只记录提示，
保留原始故障，随后由 controller.disconnect 等待恢复完成。

## 日志与验证

JSONL 保存到 `logs/right_shared_circle_时间.jsonl`，包含配置、FK/IK 审查、q、r、alpha、
policy 圆周目标、推理结果及 MIT telemetry。`follower.q=null`，`leader_only=true`。
终端每 0.1 秒显示 authority、alpha、confirm、rise、quiet、kp_scale 和窗口最大 motion。
同时显示 `ik=ok/reduced_step/waiting_ik`；JSONL 的 circle 中记录求解残差、步长比例与失败原因。
日志中的 `human_motion_range_deg` 是逐关节窗口范围，`leader_impedance_scale=1-alpha`，
`reference_active` 区分伺服目标和纯重力阶段的 shadow reference。
`applied_policy_velocity` 在卸载及纯重力阶段为零；卸载时 reference 的平滑刹停记录在 `reference_qd`。

离线测试验证持续进入确认、微动/慢速拖动和静止窗口、权重连续性、零刚度时重置 reference、
旧 chunk 失效、恢复结束不打断新轨迹，以及 MIT 三项控制律和纯重力端点与原 teleop 路径一致。
单 driver、启动、home、通信与状态故障以及现有双臂行为保留回归测试。
双臂新增离线检查覆盖：卸载阶段目标一致、实时移动终点、交接首帧无跳变、限速追赶、
恢复与再次介入、左右独立速度发送、左臂固定阻抗与重力前馈（含启用、home、HOLD），
与右臂纯重力端点。另用相同 reference 和反馈逐字段比较左臂 MIT 命令与原 single Piper 发送器。
完整双臂脚本仿真已走通 policy → blend → human → recover → policy。
仿真 rig 是简化运动模型，不验证真实接触动力学或拖动手感；新增双臂交接尚待真机验收。
