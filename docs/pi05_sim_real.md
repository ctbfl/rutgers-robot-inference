# Pi05 sim/real RTC inference

`Pi05SimRealWrapper` serves the `pi05_sim_real_rtc_inference_v1` exports using
the existing RURI protocol. The server selects `--domain real` or `--domain sim`.
Clients continue to send `observation.state` (7), RGB uint8 top/wrist images,
`prompt`, and the existing `context.rtc.*` fields. Responses contain a `(30, 7)`
absolute robot action chunk and the existing RTC execution offset. Client code
and schedulers do not change. The training camera called `hand` corresponds to
RURI's `observation.images.wrist`.

Single-target task prompts in this recipe are **`A0` through `E4`**. Use the
canonical identifier, e.g. `--prompt E1`. Row-of-five data retains its original
instruction. A per-request prompt overrides the server default; an older long
single-target instruction is not the prompt used by this training recipe.

## Checkpoint contract

Copy a complete inference export:

```
10000/
  model_spec.json
  params/                         # all Orbax files, including hidden metadata
  assets/sim_real_h30/norm_stats.json
```

No training state, dataset, simulator, or training configuration directory is
needed. RURI checks the schema, architecture, domain encoding, committed save,
required learned branches and the checkpoint normalization SHA256. Loading
then checks all parameter keys and shapes, without dropping extra parameters.
Plain Pi05 and plain train-RTC loaders reject `domain_proj` checkpoints.

The default is ten flow steps from the export. Like OpenPI's normal JAX serving
path, weights are loaded as bfloat16; exported EMA files are not modified.
The input pipeline is LiberoInputs → six-joint delta → shared quantile norm →
resize/tokenize/pad. State-token configuration remains exactly as trained.
Outputs invert normalization and add the current joint state back to the first
six dimensions; the gripper remains absolute. RTC previous actions go through
the same input transforms against the **new** observation's joint state.

After input transforms, the server adds `[0,1]` for real or `[1,0]` for sim,
and an all-true validity mask for the new H30 window. `DomainObservation`
preserves both through batching and model preprocessing. Neither field is
normalized or taken from an incoming client request. The domain projection is
added to every action token alongside the learned per-token flow-time projection.

## Environment and installation

The tested deployment layout is:

```
~/rutgers-robot-inference/          # repository, installed editable
~/ruri-deps/
  venv/                           # Python 3.11, independent of other services
  openpi/                         # matching OpenPI source snapshot
  openpi-cache/big_vision/paligemma_tokenizer.model
  checkpoints/10000/
```

The OpenPI snapshot is based on commit
`650c5b0283a49c42784fb5055a0507da2c6d347d`, including the local working-tree
files used during this audit. This inference wrapper imports no OpenPI training
registry or training data loader. It needs only OpenPI model/transform code and
the inference dependencies in `deploy/pi05_sim_real/requirements.txt`.

To recreate an environment with these source directories already present:

```bash
uv venv --python 3.11 "$HOME/ruri-deps/venv"
uv pip install --python "$HOME/ruri-deps/venv/bin/python" \
  --index https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
uv pip install --python "$HOME/ruri-deps/venv/bin/python" \
  -r "$HOME/rutgers-robot-inference/deploy/pi05_sim_real/requirements.txt"
uv pip install --python "$HOME/ruri-deps/venv/bin/python" \
  -e "$HOME/ruri-deps/openpi/packages/openpi-client"
uv pip install --python "$HOME/ruri-deps/venv/bin/python" --no-deps \
  -e "$HOME/ruri-deps/openpi" -e "$HOME/rutgers-robot-inference"
```

OpenPI imports PyTorch modules even for JAX inference; CPU wheels satisfy those
imports without adding a second GPU runtime. `--no-deps` on OpenPI deliberately
omits its unrelated training dependencies. This environment is for inference,
not for running the full OpenPI training suite.

Blackwell requires a compatible driver and CUDA stack. This recipe pins JAX
0.5.3 / Flax 0.10.2 / Orbax 0.11.13 with CUDA 12.9 compiler tools and cuDNN
9.10.2. Verify on the actual GPU rather than assuming that installing an arbitrary
newer JAX preserves OpenPI's NNX APIs. See the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) and
[NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/).

## Check and serve

From the repository root:

```bash
# No JAX/GPU initialization; metadata + normalization integrity only.
bash deploy/pi05_sim_real/serve.sh --check-only

# Loads weights, warms both paths, flips domain with fixed noise, and verifies
# absolute RTC prefixes after changing current state. Opens no robot hardware.
bash deploy/pi05_sim_real/serve.sh --prompt E1 --self-test

# Existing RURI clients can connect directly to this endpoint.
bash deploy/pi05_sim_real/serve.sh --prompt E1 \
  --bind 'tcp://*:5557' --advertise-host 172.16.90.199
```

The installed entry point is `ruri-serve-pi05-sim-real`; alternatively run
`python -m ruri.server.pi05_sim_real`. The launcher derives paths from its
location. Override `RURI_DEPLOY_ROOT`, `RURI_CHECKPOINT`, or `RURI_DOMAIN` as
needed. It defaults to on-demand GPU allocation with a 0.40 memory fraction for
the shared 5090 deployment. Adjust that fraction for the machine's deployment
layout. Warmup compiles both unconditioned and RTC paths before binding.

The old packaging declarations `ruri-server` and `ruri-client` pointed to
nonexistent `__main__` modules and were removed. The sim/real wrapper requires
no client protocol changes, including with the interactive and paired RTC clients.

## Source parity and tests

The inference model in `_sim_real_model.py` is a snapshot of the original
`tight_insertion/sim_real/model.py` and `pi05_train_rtc/model.py`, omitting
training-only loss/loaders. Training is explicitly unsupported in this copy.
Source files audited on 2026-10-11:

| Source | SHA256 |
| --- | --- |
| tight_insertion/sim_real/model.py | `07783946a89fee5010980f9aef028f6491d8d2ac02b293ff6969b8e05ac270ff` |
| pi05_train_rtc/model.py | `237a61504e01d1cdcabd79d126e374979e9fc47843ca4740a65b176e00d3e78d` |
| pi05_train_rtc/config.py | `982de969a4052a1e66a894ca5ca2b26ea7257fbac8b233b5cb4edfea2d85c4b2` |

The complete NNX parameter tree matches the training implementation in keys,
shapes and dtypes. The sampling method body has the identical Python AST after
excluding its docstring and signature. When upstream inference logic changes,
update this snapshot and repeat parity checks against the training source.

`tests/test_pi05_sim_real_wrapper.py` covers checkpoint failures, unchanged wire
keys, server-owned domain selection and RTC alignment. Optional OpenPI CPU tests
in `tests/test_pi05_sim_real_runtime.py` exercise the real preprocessing,
domain preservation and action rebasing without loading model weights.
Synthetic self-tests validate inference plumbing; they do not measure robot
insertion accuracy.

## hjs-alienware validation (2026-10-11)

Installed under `/home/hjs/rutgers-robot-inference` with the environment and
assets under `/home/hjs/ruri-deps`. The source checkout is on
`deploy/sim-real-20261011`. The integrated source includes upstream `main` and
`feature/piper-shared-reference` through `ac705ab`, alongside the local sim/real,
effort, torque and FastWAM wrappers. The canonical effort CLI option is
`--send-effort`; `--include-effort-observation` remains a compatible alias for
the original single-arm clients.

The RTX 5090 (driver 580.159.03) successfully loaded the 10000-step EMA export,
warmed both inference paths, and passed domain-flip / RTC checks. Model weights
are resident on the GPU; keeping NumPy weights on the host was measured to
cause repeated transfers and was corrected before deployment.

- Local merged full suite: 404 passed, 6 optional tests skipped.
- Remote merged full CPU suite: 403 passed, 7 skipped (optional dependencies).
- Six sequential requests through the existing RURI client over SSH forwarding:
  model time median **58.4 ms**, roundtrip median **84.6 ms** with two 480×640
  RGB images and ten denoising steps. GPU had other services running.
- Prefix checks at delays 1, 4 and 7: maximum absolute error **7.63e-6**.
- Actual 5090 policy with the upstream dual-arm simulated worker: 240 control
  steps over eight seconds, 53 requests and 37 RTC results. Gripper alignment,
  human takeover, recovery and resumed RTC all passed without writing recordings.
- The policy menu description exactly matches the inference endpoint metadata.
- Serving process GPU allocation after these checks: **8750 MiB**.

These are synthetic functional/latency checks, not an insertion-success test.
The running server is in remote tmux session `ruri-sim-real`; logs are at
`/home/hjs/ruri-deps/server.log`. It serves real-domain H30 actions, default
prompt `E1`, at port 5557; its menu listens on port 5550.

From the development host (`172.16.68.130`), SSH to `172.16.90.199:22` works,
but direct TCP connections to ports 5557/5550 time out. The remote service is
listening on all IPv4 interfaces and works via loopback. UFW is disabled;
other host/network filtering has not been ruled out. No firewall rules were
changed. Until direct access is available, establish a tunnel on the client:

```bash
ssh -N -L 15557:127.0.0.1:5557 hjs@172.16.90.199
# In another terminal, use --policy-endpoint tcp://127.0.0.1:15557
```

The tunnel changes only the endpoint configuration; client code stays the same.
To reproduce the network test after opening a tunnel:

```bash
python tests/smoke_pi05_sim_real_server.py \
  --policy-endpoint tcp://127.0.0.1:15557 --output /tmp/ruri-network-smoke.json
```

Validation logs and the installed package freeze are retained in
`/home/hjs/ruri-deps/`. The merge validation uses `merged-cpu-tests.log`,
`merged-self-test.log`, `merged-network-smoke.json`, `merged-dual-client-smoke.json`
and `merged-inference-environment.txt`. Original source patches, untracked files,
Git history and the prior server log are retained under
`backups/integration-20261011/`; the original remote edits also remain in Git stash.

For the full client suite, initialize `vendor/pyAgxArm` and install the `piper`
extra even on a machine with no arm attached. These tests use fake hardware.
When testing an isolated worktree with `PYTHONPATH`, use absolute paths: several
tests change directory before starting simulated worker subprocesses.
