Rutgers Robot Inference (RU-RI, ruri) is a lightweight robot inference layer.

Three modules make up the full inference stack:

- `policy_wrapper` runs inference. You write it yourself, generally a new one per model. Think of it as calling the model directly as an API.

- `inference_scheduler` manages timing. It reads sensor data from the robot controller, passes it to the policy wrapper, then composes the returned actions and hands them downstream to the robot controller for execution.

- `robot_setup_controller` owns the hardware and exposes an abstract interface to the layers above it, covering camera sensors, arm control, and so on.

## Choosing a remote policy

List the currently available policies and their complete metadata from the
default internal menu:

```bash
python examples/list_policy_servers.py
```

Then pass the selected endpoint explicitly to a client. Clients intentionally
have no default policy port:

```bash
python examples/client_single_piper_blocking.py \
  --policy-endpoint tcp://arcla.cs.rutgers.edu:5555 \
  --execute-actions-per-chunk 25
```

The server-declared `outputs.output_chunk_size` is authoritative; clients do
not request or override the returned action horizon.
