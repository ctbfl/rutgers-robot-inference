# RURI Blocking Scheduler 实现记录（2026-08-31）

## 已确认的上层接口

外部只负责选择组件并提供一个完整的全局 `args`：

```python
with inference_client.connect(args) as policy:
    BlockingScheduler().run(
        controller=SinglePiperController,
        policy=policy,
        args=args,
    )
```

`inference_client.connect(args)` 读取 server 地址并返回已连接的 `policy`；Scheduler 和 Controller 继续从同一个 `args` 对象读取各自关心的字段。

## 阻塞执行语义

`BlockingScheduler.run()` 内部负责：

1. 接收由 `inference_client.connect()` 建立、已完成 metadata readiness 的 `policy`。
2. 构造并调用 `controller.start()`，等待相机、CAN、MIT worker 和机械臂全部 ready。
3. 循环获取 observation，由 Scheduler 注入 `args.prompt` 后调用 `policy.infer()`，再解析 `action_chunk`；完整 horizon 以 server metadata 的 `outputs.output_chunk_size` 为准。
4. 以 `control_hz` 逐行调用 `controller.send_action()`，完整消费当前 chunk 后才请求下一个 chunk。
5. 退出或异常时执行 `controller.stop()`；policy connection 生命周期属于外层 `with`。

`max_chunks=None` 时持续运行；设置整数时执行对应数量的 chunk。

## 实现位置

- `src/ruri/client/schedulers/blocking.py`
- `src/ruri/client/utils/inference_client.py`
- `src/ruri/client/_args.py`
- `src/ruri/client/controllers/dummy.py`
- `src/ruri/client/controllers/robot_setup_controller.py`
- `src/ruri/client/controllers/single_piper/single_piper.py`

`SinglePiperController.start()` 组合现有的 `connect()` 和 `start_arm()`；`stop()` 复用幂等的 `disconnect()`。底层的 observation-only `connect()` 和显式 `start_arm()` 仍保留。

## 验证与限制

- 相关单元测试全部通过。
- 覆盖 policy-before-Controller readiness、两轮 chunk 顺序执行、异常清理、远程请求组装和 Single Piper 完整 start gate。
- 已用 `DummyController` 对当前远程 Pi0.5 server（checkpoint `6000`）完成两轮真实 inference：每个 chunk 截取 3 行，共下发 6 个 `(7,)` action；全部为有限值，最终 dummy state 与最后一行动作一致。两次 `timing.wrapper_ms` 约为 184 ms 和 68 ms。
- 尚未用真实机械臂运行这个 Scheduler。
- 这是刻意保持简单的阻塞基线：远程 inference 期间不会继续发送 action。如果 inference 间隔超过 MIT `command_timeout_s`，可能触发 watchdog。以后可在 Scheduler 内增加独立 executor，外部 `run(controller, policy, args)` 接口不需要改变。
