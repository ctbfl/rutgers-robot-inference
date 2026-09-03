# Optional integrations

These packages adapt RURI controllers to frameworks that users install and
manage separately. They deliberately do not install LeRobot, PyTorch, CUDA, or
RealSense dependencies.

The unsuffixed adapters target LeRobot 0.6.1 and are the active implementation.
Prepare LeRobot yourself, then install RURI and these two adapters into that
environment:

```bash
git submodule update --init vendor/pyAgxArm
pip install -e '.[piper]'
pip install -e integrations/lerobot_robot_piper_mit
pip install -e integrations/lerobot_teleoperator_piper_mit
```

The previous LeRobot 0.5.2 implementations remain available for teams that
have not migrated:

```bash
pip install -e integrations/lerobot_robot_piper_mit_lerobot0_5_2
pip install -e integrations/lerobot_teleoperator_piper_mit_lerobot0_5_2
```

Install exactly one matching adapter pair in an environment. Both generations
register the same stable `piper_mit_observer` and `piper_mit` type names, so
installing both generations together is unsupported.

The LeRobot process opens only the two cameras and the local telemetry socket;
it never opens CAN. LeRobot 0.6.1 connects the teleoperator before the robot,
which the active adapter supports while preserving RURI's same-frame telemetry
latch.
