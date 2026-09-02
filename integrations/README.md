# Optional integrations

These packages adapt RURI controllers to frameworks that users install and
manage separately. They deliberately do not install LeRobot, PyTorch, CUDA, or
RealSense dependencies.

For LeRobot 0.5.2, first prepare the LeRobot environment yourself. Then install
RURI and the two adapters into that environment:

```bash
git submodule update --init vendor/pyAgxArm
pip install -e '.[piper]'
pip install -e integrations/lerobot_robot_piper_mit
pip install -e integrations/lerobot_teleoperator_piper_mit
```

Start the CAN-owning RURI teleop controller in one terminal, then run
`lerobot-record` in another. The LeRobot process opens only the two cameras and
the local telemetry socket; it never opens CAN.
