"""
Serve the torque-aware training-time RTC Pi0.5 trained on tight_insertion_whole_v1.

The model (``/common/users/jh2400/pi05-torque``) conditions pi05's action expert
on a **history** of the seven measured efforts -- six joint torques and the
gripper force -- sampled 0.2 s apart over the last 1.8 s: ten frames at dataset
offsets ``-54, -48, ..., -6, 0`` at 30 fps. Everything else is
:class:`Pi05TrainRTCWrapper`: seven-value state, two RGB cameras, a 30-step
chunk of absolute targets, exact training-time RTC prefix conditioning.

Why the client sends a timestamped buffer
-----------------------------------------
The history is defined on a time grid, and no RURI scheduler reads the robot
at that grid: the blocking scheduler reads one observation per chunk and the
RTC scheduler one per request, on its own thread. So the client has to keep its
own effort buffer, at whatever rate its loop runs, and the server can only
place the ten frames correctly if it knows when each buffered row was taken.
It therefore takes the raw buffer with its timestamps and picks the rows
itself -- the one place that knows the offsets the checkpoint was trained on.

Selection is sample-and-hold, which is what the recorder did: at every 30 Hz
frame it latched the newest telemetry packet. For each offset the server takes
the newest row whose time is at or before ``t_newest + offset`` (with 1 ms of
tolerance, so an exactly 30 Hz buffer reproduces dataset indexing). An offset
older than the oldest row takes the oldest row -- which is exactly how the
training data looked for the first 1.8 s of every episode, since LeRobot clamps
history indices to the episode start. A client that clears its buffer when an
episode starts therefore matches training with no special case.

What the server checks, and what it cannot
------------------------------------------
Shapes, dtypes, finiteness, strictly increasing timestamps, and a coarse unit
bound (a joint torque of hundreds of N*m means the values are not in N*m) are
rejected loudly. The server cannot check that the newest buffer row and
``observation.state`` come from the same telemetry packet, that the effort
channels are in the recorder's order, or that the buffer was cleared at the
episode start. Each response carries ``torque.*`` diagnostics so a client can
log what the server actually used.

The model package is imported from ``torque_package``, never from a training
run's ``source/`` snapshot: the run that produced these weights pinned a
transform that rejected every RTC-conditioned inference call (fixed in
pi05-torque 79aff65, inference only). The loader probes for that.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ruri.server.wrappers.pi05.pi05 import (
    CONTEXT_CONSUMED_STEPS,
    CONTEXT_INFERENCE_DELAY,
    CONTEXT_PREV_CHUNK,
    OPENPI_BASE_IMAGE,
    OPENPI_PROMPT,
    OPENPI_STATE,
    OPENPI_WRIST_IMAGE,
    Pi05Wrapper,
    _require,
)
from ruri.server.wrappers.pi05.pi05_train_rtc import Pi05TrainRTCWrapper


logger = logging.getLogger(__name__)


TORQUE_PACKAGE = "/common/users/jh2400/pi05-torque"
TRAIN_RTC_PACKAGE = "/common/users/jh2400/pi05-rtc"
CONFIG_NAME = "pi05_whole_v1_torque_rtc_h30"
ASSET_ID = "tight_insertion_whole_v1_v21"

# RURI keys (layer 1), policy keys (layer 2), and what PiperTorqueInputs consumes (layer 3).
EFFORT_HISTORY = "observation.joint_effort_history"
EFFORT_TIMES = "observation.joint_effort_history_t"
OPENPI_EFFORT_HISTORY = "observation/joint_effort_history"
OPENPI_EFFORT_TIMES = "observation/joint_effort_history_t"
OPENPI_EFFORT = "observation/joint_effort"

EFFORT_DIM = 7
EFFORT_CHANNELS = (
    "joint1 torque [N*m]", "joint2 torque [N*m]", "joint3 torque [N*m]",
    "joint4 torque [N*m]", "joint5 torque [N*m]", "joint6 torque [N*m]",
    "gripper force [N]",
)
# Mirrors pi05_torque.spec.HISTORY_OFFSETS / FPS; checked against it at load time.
# Kept here so importing this module needs neither OpenPI nor the model package.
HISTORY_OFFSETS_S = tuple(k * 6 / 30 for k in range(-9, 1))
HISTORY_SPAN_S = -HISTORY_OFFSETS_S[0]
SELECTION_TOLERANCE_S = 1e-3
# The largest |effort| in the 291,756 training frames is 18.3 (joint2, N*m). A
# value past this bound is a unit or channel mistake, not a hard contact.
MAX_ABS_EFFORT = 200.0
MAX_HISTORY_ROWS = 10_000
# A slot whose chosen row is this much older than its target time means the
# buffer is sampled too sparsely to resolve the 0.2 s grid.
STALE_HOLD_MS = 100.0
# Normalized torque beyond this is clipped inside the model (pi05_torque.spec.TORQUE_CLIP).
TORQUE_CLIP = 5.0

PROMPTS = tuple(f"{column}{row}" for column in "ABCDE" for row in range(5))
WARMUP_PROMPT = "A0"
ACTION_HORIZON = 30
MAX_TRAINED_DELAY = 7

REQUIRED_PARAMS = ("tok_time_proj", "torque_mlp_in", "torque_mlp_out", "torque_out_proj")


def select_effort_history(history: Any, times: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Pick the ten trained history frames out of a client buffer.

    Returns the ``(10, 7)`` float32 history the model consumes, oldest first,
    and the diagnostics reported back to the client. Raises on anything that
    would otherwise feed the model a quietly wrong span of time or wrong units.
    """
    if not isinstance(history, np.ndarray):
        raise TypeError(
            f"{EFFORT_HISTORY!r} must be a np.ndarray of shape (N, {EFFORT_DIM}), "
            f"got {type(history).__name__}."
        )
    if history.ndim != 2 or history.shape[1] != EFFORT_DIM or history.shape[0] < 1:
        hint = ""
        if history.shape == (EFFORT_DIM,):
            hint = (
                " A single effort frame is the pi05_effort contract; this policy needs the "
                "buffered history, one row per telemetry sample, oldest first."
            )
        raise ValueError(
            f"{EFFORT_HISTORY!r} must have shape (N, {EFFORT_DIM}) with N >= 1, got "
            f"{history.shape}.{hint}"
        )
    if history.shape[0] > MAX_HISTORY_ROWS:
        raise ValueError(
            f"{EFFORT_HISTORY!r} has {history.shape[0]} rows; send only the last few "
            f"seconds (the policy reads {HISTORY_SPAN_S:.1f} s)."
        )
    if not np.issubdtype(history.dtype, np.floating):
        raise TypeError(f"{EFFORT_HISTORY!r} must be a float array, got {history.dtype}.")
    if not np.all(np.isfinite(history)):
        raise ValueError(f"{EFFORT_HISTORY!r} contains NaN or inf.")
    peak = float(np.max(np.abs(history)))
    if peak > MAX_ABS_EFFORT:
        raise ValueError(
            f"{EFFORT_HISTORY!r} contains |value| = {peak:.1f}. Expected the recorder's "
            "units -- joint torques in N*m and gripper force in N, as returned by "
            "recording_observation(packet)['observation.joint_effort'] -- whose training "
            f"range stays within +-20. Check units and channel order."
        )

    if not isinstance(times, np.ndarray):
        raise TypeError(
            f"{EFFORT_TIMES!r} must be a np.ndarray of shape (N,), got {type(times).__name__}."
        )
    if times.dtype != np.float64:
        raise TypeError(
            f"{EFFORT_TIMES!r} must be float64 seconds, got {times.dtype}. Integer timestamps "
            "are ambiguous (s or ns), and float32 cannot resolve milliseconds at monotonic-clock "
            "magnitudes; send e.g. np.asarray([p['monotonic_ns'] / 1e9 ...], dtype=np.float64)."
        )
    if times.shape != (history.shape[0],):
        raise ValueError(
            f"{EFFORT_TIMES!r} must have one entry per history row: expected shape "
            f"({history.shape[0]},), got {times.shape}."
        )
    times = times.astype(np.float64)
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{EFFORT_TIMES!r} contains NaN or inf.")
    if times.shape[0] > 1 and not np.all(np.diff(times) > 0):
        raise ValueError(
            f"{EFFORT_TIMES!r} must be strictly increasing (oldest row first). Drop a "
            "telemetry packet whose time is not newer than the last buffered row."
        )

    targets = times[-1] + np.asarray(HISTORY_OFFSETS_S)
    index = np.searchsorted(times, targets + SELECTION_TOLERANCE_S, side="right") - 1
    from_oldest = index < 0
    index = np.maximum(index, 0)
    hold_ms = (targets - times[index]) * 1000.0
    hold_ms = np.where(from_oldest, 0.0, np.maximum(hold_ms, 0.0))

    selected = np.ascontiguousarray(history[index], dtype=np.float32)
    info = {
        "torque.history_rows": int(history.shape[0]),
        "torque.history_span_s": float(times[-1] - times[0]),
        "torque.slots_from_oldest_row": int(np.sum(from_oldest)),
        "torque.max_hold_ms": round(float(np.max(hold_ms)), 3),
    }
    return selected, info


class Pi05TrainRTCTorqueWrapper(Pi05TrainRTCWrapper):
    """
    Torque-history Pi0.5 with training-time RTC.

    Example:
        >>> wrapper = Pi05TrainRTCTorqueWrapper(
        ...     "/common/users/jh2400/training_results/pi05_train_rtc/"
        ...     "260915_whole_v1_torque_rtc_h30_4gpu_bs128/checkpoints/"
        ...     "pi05_whole_v1_torque_rtc_h30/whole_v1_torque_rtc_h30_4gpu_bs128/10000"
        ... )
        >>> wrapper.infer({
        ...     "observation.state": state7,
        ...     "observation.images.top": top_rgb,
        ...     "observation.images.wrist": wrist_rgb,
        ...     "observation.joint_effort_history": effort_buffer,     # (N, 7)
        ...     "observation.joint_effort_history_t": effort_times,    # (N,) s
        ...     "prompt": "C3",
        ... })["action_chunk"].shape
        (30, 7)

    Args:
        torque_package:
            Directory containing ``pi05_torque``. Must be the package itself, not a
            training run's ``source/`` snapshot (see the module docstring).
        train_rtc_package:
            Directory containing ``pi05_train_rtc``, which ``pi05_torque`` subclasses.
        default_inference_delay, rtc_warmup:
            As for :class:`Pi05TrainRTCWrapper`.

        ``num_denoising_steps``, ``warmup`` and ``warmup_image_shape`` are
        :class:`Pi05Wrapper`'s. There is no default prompt: the prompt names the
        target hole, so each request must carry one.
    """

    INPUT_MAPPING = {
        **Pi05Wrapper.INPUT_MAPPING,
        EFFORT_HISTORY: OPENPI_EFFORT_HISTORY,
        EFFORT_TIMES: OPENPI_EFFORT_TIMES,
    }

    POLICY_METADATA = {
        "inputs": {
            OPENPI_STATE: {
                "type": "state",
                "required": True,
                "shape": [7],
                "dtype": "float32",
                "description": (
                    "Follower joint1..joint6 positions and gripper opening in the Piper "
                    "dataset convention, i.e. normalize_telemetry(packet), which equals "
                    "recording_observation(packet)['observation.state']. Read it from the "
                    "same telemetry packet as the newest row of "
                    "observation.joint_effort_history."
                ),
            },
            OPENPI_BASE_IMAGE: {
                "type": "image",
                "required": True,
                "shape": ["H", "W", 3],
                "dtype": "uint8",
                "channel_order": "RGB",
                "description": (
                    "Current head camera (D435) frame, trained as dataset key "
                    "observation.images.top. Any resolution; the server resizes."
                ),
            },
            OPENPI_WRIST_IMAGE: {
                "type": "image",
                "required": True,
                "shape": ["H", "W", 3],
                "dtype": "uint8",
                "channel_order": "RGB",
                "description": (
                    "Current wrist camera (D415) frame, trained as dataset key "
                    "observation.images.hand. Any resolution; the server resizes."
                ),
            },
            OPENPI_PROMPT: {
                "type": "string",
                "required": True,
                "allowed_values": list(PROMPTS),
                "description": (
                    "Target hole on the manipulation-net board: column letter A-E followed "
                    "by row digit 0-4, exactly as the training task strings, e.g. 'C3'. "
                    "Nothing else -- no sentence, no lowercase. There is no server default."
                ),
            },
            OPENPI_EFFORT_HISTORY: {
                "type": "effort_history",
                "required": True,
                "shape": ["N", 7],
                "dtype": "float32",
                "row_order": "oldest first, newest last",
                "channels": list(EFFORT_CHANNELS),
                "units": "raw measured values, signed: N*m for joints 1-6, N for the gripper",
                "source": (
                    "ruri.client.controllers.single_piper.normalization."
                    "recording_observation(packet)['observation.joint_effort'], one row per "
                    "follower telemetry packet"
                ),
                "min_span_s": HISTORY_SPAN_S,
                "description": (
                    "The client's rolling buffer of measured effort, sampled at >= 30 Hz "
                    "and covering at least the last 1.8 s (keep ~2 s). The newest row must "
                    "come from the same telemetry packet as observation.state. Clear the "
                    "buffer when an episode starts. Send it unmodified: no resampling, "
                    "normalization, filtering, abs() or padding -- the server selects the "
                    "rows it needs."
                ),
            },
            OPENPI_EFFORT_TIMES: {
                "type": "timestamps",
                "required": True,
                "shape": ["N"],
                "dtype": "float64",
                "units": "s",
                "description": (
                    "Acquisition time of each row of observation.joint_effort_history, "
                    "strictly increasing, one clock for all rows, e.g. "
                    "packet['monotonic_ns'] / 1e9. Only differences are used; the origin "
                    "does not matter. Integer (ns) arrays are rejected."
                ),
            },
            CONTEXT_PREV_CHUNK: {
                "type": "action_chunk",
                "optional": True,
                "shape": ["30 - consumed_steps", 7],
                "dtype": "float32",
                "description": (
                    "Training-time RTC: the previous response's action_chunk with its "
                    "consumed rows dropped, unmodified (absolute targets, as returned), so "
                    "its row 0 is the control step this request's row 0 covers. Omit on "
                    "the first request of an episode."
                ),
            },
            CONTEXT_CONSUMED_STEPS: {
                "type": "int",
                "optional": True,
                "description": (
                    "Rows of the previous chunk already executed when this request's "
                    "observation was taken; must equal 30 - len(prev_chunk_left_over)."
                ),
            },
            CONTEXT_INFERENCE_DELAY: {
                "type": "int",
                "optional": True,
                "range": [0, MAX_TRAINED_DELAY],
                "description": (
                    "Control steps (at 30 Hz) expected to elapse before this response can "
                    "be executed. These leading rows of prev_chunk_left_over are pinned. "
                    "Default 4 (~120 ms); values above 7 are clamped."
                ),
            },
        },
    }

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        torque_package: str | Path = TORQUE_PACKAGE,
        train_rtc_package: str | Path = TRAIN_RTC_PACKAGE,
        default_inference_delay: int = 4,
        rtc_warmup: bool = True,
        **kwargs: Any,
    ):
        if kwargs.get("state_dim", 7) != 7:
            raise ValueError("This policy takes seven state values and returns seven actions.")
        if kwargs.get("default_prompt") is not None:
            raise ValueError(
                "This policy has no default prompt: the prompt names the target hole "
                f"({PROMPTS[0]}..{PROMPTS[-1]}) and must come with every request."
            )
        if not 0 <= int(default_inference_delay) <= MAX_TRAINED_DELAY:
            raise ValueError(f"default_inference_delay must be in the trained range 0-{MAX_TRAINED_DELAY}.")
        self.torque_package = Path(torque_package).expanduser().resolve()
        self.train_rtc_package = Path(train_rtc_package).expanduser().resolve()
        self.default_inference_delay = int(default_inference_delay)
        self.rtc_warmup = rtc_warmup
        self._effort_q01: np.ndarray | None = None
        self._effort_q99: np.ndarray | None = None
        self._torque_info: dict[str, Any] = {}
        # Pi05TrainRTCWrapper.__init__ pins the row1 config name; this model has its own.
        Pi05Wrapper.__init__(self, checkpoint_path, CONFIG_NAME, **kwargs)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @property
    def max_inference_delay(self) -> int:
        return len(self.policy._model.delay_probs) - 1

    def _check_checkpoint(self) -> None:
        """Refuse anything but a committed torque + training-time RTC step directory."""
        root = self.checkpoint_path
        for required in ("params/_METADATA", "_CHECKPOINT_METADATA", f"assets/{ASSET_ID}/norm_stats.json"):
            if not (root / required).is_file():
                raise FileNotFoundError(
                    f"Checkpoint is missing {required}: {root}. Point at one training step "
                    "directory (e.g. .../10000), not at the run above it."
                )
        commit = json.loads((root / "_CHECKPOINT_METADATA").read_text())
        if not commit.get("commit_timestamp_nsecs"):
            raise ValueError(f"Checkpoint save has not committed; pick a completed step: {root}")
        metadata = json.loads((root / "params/_METADATA").read_text())
        paths = {
            tuple(k["key"] for k in leaf["key_metadata"])
            for leaf in metadata["tree_metadata"].values()
        }
        for layer in REQUIRED_PARAMS:
            for parameter in ("kernel", "bias"):
                if ("params", layer, parameter, "value") not in paths:
                    raise ValueError(
                        f"Not a torque training-time RTC checkpoint: missing {layer}/{parameter}. "
                        "A gripper-effort checkpoint goes to pi05_effort; a plain RTC one to "
                        "pi05_train_rtc."
                    )
        stats = json.loads((root / f"assets/{ASSET_ID}/norm_stats.json").read_text())["norm_stats"]
        if len(stats["state"]["q01"]) != 77 or len(stats["actions"]["q01"]) != 14:
            raise ValueError(
                "Checkpoint norm stats are not the torque layout (state 77 = 7 positions + "
                f"10x7 torque, actions 14 = 7 + 7): {root}"
            )

    def _import_torque_package(self):
        for directory in (self.torque_package, self.train_rtc_package):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        try:
            import pi05_torque
            from pi05_torque import spec
            from pi05_torque.config import make_train_config
        except ImportError as exc:
            raise ImportError(
                "Could not import pi05_torque / pi05_train_rtc. Run the server in the OpenPI "
                "environment and pass torque_package / train_rtc_package."
            ) from exc
        origin = Path(pi05_torque.__file__).resolve().parent.parent
        if origin != self.torque_package:
            raise ImportError(
                f"pi05_torque was already imported from {origin}, not {self.torque_package}. "
                "A training run's source/ snapshot carries a transform that rejects RTC "
                "inference; serve from the package."
            )
        expected = tuple(offset / spec.FPS for offset in spec.HISTORY_OFFSETS)
        if not np.allclose(expected, HISTORY_OFFSETS_S) or spec.TORQUE_DIM != EFFORT_DIM:
            raise RuntimeError(
                f"pi05_torque.spec history {expected} / dim {spec.TORQUE_DIM} disagrees with "
                f"this wrapper's {HISTORY_OFFSETS_S} / {EFFORT_DIM}; update both together."
            )
        return make_train_config

    def _load_policy(self):
        self._check_checkpoint()
        make_train_config = self._import_torque_package()
        try:
            import jax.numpy as jnp
            from openpi import transforms
            from openpi.models import model as _model
            from openpi.policies.policy import Policy
        except ImportError as exc:
            raise ImportError("Pi05TrainRTCTorqueWrapper requires OpenPI.") from exc

        # Every normalization read resolves against the checkpoint's own assets.
        config = make_train_config(CONFIG_NAME, "serve", self.checkpoint_path / "assets")
        if config.model.action_horizon != ACTION_HORIZON:
            raise RuntimeError(f"Expected action horizon {ACTION_HORIZON}, got {config.model.action_horizon}")
        data = config.data.create(config.assets_dirs, config.model)
        self._probe_inference_transform(data, config.model)

        logger.info("Loading torque training-time RTC policy from %s", self.checkpoint_path)
        params = _model.restore_params(self.checkpoint_path / "params", dtype=jnp.bfloat16)
        # remove_extra_params=False: an unexpected or missing parameter is an error, not a drop.
        model = config.model.load(params, remove_extra_params=False)

        norm_stats = data.norm_stats
        sample_kwargs = None if self.num_denoising_steps is None else {"num_steps": self.num_denoising_steps}
        policy = Policy(
            model,
            transforms=[
                *data.data_transforms.inputs,
                transforms.Normalize(norm_stats, use_quantiles=data.use_quantile_norm),
                *data.model_transforms.inputs,
            ],
            output_transforms=[
                *data.model_transforms.outputs,
                transforms.Unnormalize(norm_stats, use_quantiles=data.use_quantile_norm),
                *data.data_transforms.outputs,
            ],
            sample_kwargs=sample_kwargs,
            metadata=config.policy_metadata,
        )
        self._assert_rtc_capable(policy)
        self._assert_torque_trained(policy)
        self._effort_q01 = np.asarray(norm_stats["actions"].q01[7:14], dtype=np.float64)
        self._effort_q99 = np.asarray(norm_stats["actions"].q99[7:14], dtype=np.float64)
        self.train_config_name = config.name
        logger.info("Torque policy loaded (H=%d, trained delays 0-%d)",
                    model.action_horizon, len(model.delay_probs) - 1)
        return policy

    @staticmethod
    def _probe_inference_transform(data, model_config) -> None:
        """RTC inference passes 'actions' with a 10-frame history; the input transform must accept it."""
        probe = {
            "observation/state": np.zeros(7, np.float32),
            "observation/image": np.zeros((8, 8, 3), np.uint8),
            "observation/wrist_image": np.zeros((8, 8, 3), np.uint8),
            OPENPI_EFFORT: np.zeros((len(HISTORY_OFFSETS_S), EFFORT_DIM), np.float32),
            "actions": np.zeros((model_config.action_horizon, 7), np.float32),
            "prompt": WARMUP_PROMPT,
        }
        try:
            out = data.data_transforms.inputs[0](probe)
        except ValueError as exc:
            raise RuntimeError(
                "The loaded pi05_torque input transform rejects an RTC inference request "
                "(10 torque frames plus a previous chunk). It predates pi05-torque 79aff65; "
                "serve from the fixed package."
            ) from exc
        if np.asarray(out["actions"]).shape != (model_config.action_horizon, 7):
            raise RuntimeError("pi05_torque input transform altered the RTC prefix at inference.")

    @staticmethod
    def _assert_torque_trained(policy) -> None:
        """A zero torque output layer is the zero-init signature of an untrained adapter."""
        import flax.nnx as nnx

        state = nnx.state(policy._model, nnx.Param).to_pure_dict()
        for layer in ("torque_mlp_out", "torque_out_proj"):
            kernel = state.get(layer, {}).get("kernel")
            if kernel is None:
                raise RuntimeError(f"Loaded model has no {layer}; it is not the torque model.")
            norm = float(np.linalg.norm(np.asarray(kernel, dtype=np.float32)))
            if norm == 0.0:
                raise RuntimeError(f"{layer} is all zeros: this checkpoint's torque path was never trained.")
            logger.info("%s restored (||kernel||_F = %.4f)", layer, norm)

    def warmup(self) -> None:
        """Compile the unconditioned and the prefix-conditioned path with a full history."""
        buffer_rows = 60
        dummy = {
            "observation.state": np.zeros(7, dtype=np.float32),
            "observation.images.top": np.zeros(self.warmup_image_shape, dtype=np.uint8),
            "observation.images.wrist": np.zeros(self.warmup_image_shape, dtype=np.uint8),
            EFFORT_HISTORY: np.zeros((buffer_rows, EFFORT_DIM), dtype=np.float32),
            EFFORT_TIMES: np.arange(buffer_rows, dtype=np.float64) / 30.0,
            "prompt": WARMUP_PROMPT,
        }
        start = time.perf_counter()
        self.infer(dummy)
        logger.info("Torque warmup (unconditioned) finished in %.1f s", time.perf_counter() - start)
        if not self.rtc_warmup:
            return
        start = time.perf_counter()
        response = self.infer({
            **dummy,
            CONTEXT_PREV_CHUNK: np.zeros((self.output_chunk_size, 7), dtype=np.float32),
            CONTEXT_CONSUMED_STEPS: 0,
            CONTEXT_INFERENCE_DELAY: max(1, self.default_inference_delay),
        })
        if not response.get("rtc.applied"):
            raise RuntimeError(
                f"Torque RTC warmup did not exercise the conditioned path ({response.get('rtc.reason')!r})."
            )
        logger.info("Torque warmup (prefix-conditioned) finished in %.1f s", time.perf_counter() - start)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _resolve_prompt(self, inputs: dict[str, Any]) -> str:
        prompt = inputs.get(OPENPI_PROMPT)
        if not isinstance(prompt, str) or not prompt.strip():
            raise KeyError(
                f"Request has no 'prompt'. This policy needs the target hole, one of "
                f"{', '.join(PROMPTS)}, in every request."
            )
        prompt = prompt.strip()
        if prompt not in PROMPTS:
            raise ValueError(
                f"prompt {prompt!r} is not a trained task string. Send exactly one of "
                f"{', '.join(PROMPTS)} (uppercase column letter, then row digit)."
            )
        return prompt

    def _to_openpi_observation(self, inputs: dict[str, Any], prompt: str) -> dict[str, Any]:
        observation = super()._to_openpi_observation(inputs, prompt)
        state = observation[OPENPI_STATE]
        if state.shape != (7,) or not np.all(np.isfinite(state)):
            raise ValueError(
                f"'observation.state' must be seven finite values, got shape {state.shape}."
            )
        if OPENPI_EFFORT_HISTORY not in inputs and "observation.joint_effort" in inputs:
            raise KeyError(
                f"Request carries a single 'observation.joint_effort' frame. This policy needs "
                f"{EFFORT_HISTORY!r} (N, 7) and {EFFORT_TIMES!r} (N,) instead."
            )
        history = _require(inputs, OPENPI_EFFORT_HISTORY, "torque history", EFFORT_HISTORY)
        times = _require(inputs, OPENPI_EFFORT_TIMES, "torque history timestamps", EFFORT_TIMES)
        selected, info = select_effort_history(history, times)

        if self._effort_q01 is not None:
            normalized = (selected - self._effort_q01) / (self._effort_q99 - self._effort_q01 + 1e-6) * 2.0 - 1.0
            info["torque.clipped_values"] = int(np.sum(np.abs(normalized) > TORQUE_CLIP))
            if info["torque.clipped_values"]:
                logger.warning(
                    "%d torque values fall far outside the training range and are clipped; "
                    "newest row %s", info["torque.clipped_values"], np.round(selected[-1], 3).tolist(),
                )
        if info["torque.max_hold_ms"] > STALE_HOLD_MS:
            logger.warning(
                "torque history is sampled too sparsely: a selected row is %.0f ms older than "
                "its slot. Buffer every telemetry packet (>= 30 Hz).", info["torque.max_hold_ms"],
            )
        self._torque_info = info
        observation[OPENPI_EFFORT] = selected
        return observation

    def _infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self._torque_info = {}
        response = super()._infer(inputs)
        response.update(self._torque_info)
        return response

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    def ruri_metadata(self) -> dict[str, Any]:
        """Also state what an action means, which the base only gives the length of."""
        metadata = super().ruri_metadata()
        metadata["outputs"].update({
            "action_dim": 7,
            "action_chunk": {
                "shape": [self.output_chunk_size, 7],
                "dtype": "float32",
                "semantics": (
                    "absolute targets in the same normalized convention as observation.state "
                    "(joint1..joint6 positions, gripper opening), one row per 30 Hz control "
                    "step; row 0 is the step at which the request's observation was taken"
                ),
                "execute_from": "rtc.execute_from",
            },
            "response_keys": {
                "action_chunk": "float32 (30, 7)",
                "prompt": "str, the prompt used",
                "rtc.applied": "bool",
                "rtc.reason": "str or None, why prefix conditioning was not applied",
                "rtc.method": "'training-time'",
                "rtc.inference_delay": "int, pinned prefix length after clamping",
                "rtc.execute_from": "int, first row to execute (rows before it are the pinned prefix)",
                "torque.history_rows": "int, rows received",
                "torque.history_span_s": "float, newest minus oldest timestamp",
                "torque.slots_from_oldest_row": (
                    "int 0-9, history slots older than the buffer (filled with the oldest row); "
                    "should reach 0 once the episode is 1.8 s old"
                ),
                "torque.max_hold_ms": "float, staleness of the chosen rows; > 100 means the buffer is too sparse",
                "torque.clipped_values": "int, torque values far outside the training range; > 0 means check units",
                "timing.infer_ms": "float",
                "timing.wrapper_ms": "float",
            },
        })
        return metadata

    def optional_more_metadata(self) -> dict[str, Any]:
        metadata = super().optional_more_metadata()
        metadata.update({
            "name": "pi05_train_rtc_torque",
            "summary": (
                "Pi0.5 with training-time RTC, conditioned on a 1.8 s history of the seven "
                "measured efforts (6 joint torques + gripper force). Trained on "
                "tight_insertion_whole_v1 (920 episodes, 30 fps, prompts A0-E4)."
            ),
            "model_package": str(self.torque_package),
            "paper": "TA-VLA (arXiv 2509.07962), adapted; training-time RTC (arXiv 2512.05964)",
            "supports_rtc": True,
            "state_dim": 7,
            "action_dim": 7,
            "control_hz": 30,
            "prompts": list(PROMPTS),
            "torque": {
                "history_offsets_s": [round(x, 6) for x in HISTORY_OFFSETS_S],
                "history_offsets_frames_at_30hz": [round(x * 30) for x in HISTORY_OFFSETS_S],
                "selection": (
                    "sample-and-hold: for each offset, the newest row with "
                    "t <= t_newest + offset (+1 ms tolerance); offsets older than the oldest "
                    "row use the oldest row, as training did for the first 1.8 s of an episode"
                ),
                "min_span_s": HISTORY_SPAN_S,
                "recommended_buffer_s": 2.0,
                "recommended_sample_rate_hz": 30,
                "channels": list(EFFORT_CHANNELS),
                "max_abs_value_accepted": MAX_ABS_EFFORT,
                "training_q01": None if self._effort_q01 is None else np.round(self._effort_q01, 4).tolist(),
                "training_q99": None if self._effort_q99 is None else np.round(self._effort_q99, 4).tolist(),
            },
            "client_guide": [
                "1. Buffer: poll follower telemetry at a steady >= 30 Hz for as long as the "
                "policy runs, and for each NEW packet P append the row "
                "recording_observation(P)['observation.joint_effort'] (float32 (7,): joint1-6 "
                "torque in N*m, then gripper force in N; signed, raw) with the time "
                "P['monotonic_ns'] / 1e9 (float64). The receiver hands back the newest packet "
                "again when nothing new has arrived, so skip P when its 'sequence' equals the last "
                "buffered one. Keep the last ~2.0 s (a deque of 60-120 rows at 30 Hz).",
                "2. Where packets come from today: SinglePiperController reads them with "
                "self._telemetry.latest_engaged(config.telemetry_timeout_s) inside "
                "get_observation(); MITTelemetryReceiver keeps only the newest packet, so a "
                "history exists only if something polls it at >= 30 Hz, and the receiver is not "
                "thread-safe (RTCScheduler calls get_observation() from its inference thread). "
                "Do every telemetry read from one place -- e.g. one sampler thread that buffers "
                "and caches the latest packet -- and have observation readers use that cache. No "
                "public accessor returns the packet or its time yet; add one.",
                "3. Episode start: clear the buffer whenever a new episode or prompt starts "
                "(the same moment you drop the RTC previous chunk). The server fills history "
                "older than the buffer with the oldest row, which is what training saw.",
                "4. Each request: use ONE packet P for both observation.state "
                "(normalize_telemetry(P), equal to recording_observation(P)['observation.state']) "
                "and the newest buffer row, then send observation.joint_effort_history = the "
                "whole buffer as float32 (N, 7) oldest first, observation.joint_effort_history_t "
                "= its float64 (N,) times, the current frames from get_camera_observation() "
                "(HWC uint8 RGB), and prompt = the target hole ('A0'..'E4'). Do not pair the "
                "buffer with get_observation()'s state: that call reads its own, possibly newer, "
                "packet.",
                "5. Do not resample, normalize, filter, reorder, abs() or pad the effort history; "
                "the server picks the rows at t_newest - 1.8, -1.6, ..., -0.2, 0.0 s by "
                "sample-and-hold, exactly as the 30 Hz recorder latched them.",
                "6. Training-time RTC (optional, recommended with a background inference "
                "thread): add context.rtc.prev_chunk_left_over = previous action_chunk minus its "
                "consumed rows (unmodified), context.rtc.consumed_steps, and "
                "context.rtc.estimated_inference_delay_steps (0-7, 30 Hz steps). The existing "
                "ruri RTCScheduler already sends these.",
                "7. Response: action_chunk is float32 (30, 7) absolute targets in the "
                "observation.state convention, one row per 30 Hz step. Start executing at "
                "rtc.execute_from, never at row 0 when rtc.applied is true.",
                "8. Log torque.* from each response: slots_from_oldest_row should be 0 once the "
                "episode is 1.8 s old, max_hold_ms should stay below ~40 at 30 Hz, and "
                "clipped_values should be 0 (otherwise check units and channel order).",
            ],
            "example_request": {
                "observation.state": "float32 (7,)",
                "observation.images.top": "uint8 (480, 640, 3) RGB",
                "observation.images.wrist": "uint8 (480, 640, 3) RGB",
                "observation.joint_effort_history": "float32 (60, 7), oldest first",
                "observation.joint_effort_history_t": "float64 (60,), seconds, increasing",
                "prompt": "C3",
                "context.rtc.prev_chunk_left_over": "float32 (22, 7)  # optional",
                "context.rtc.consumed_steps": "8  # optional",
                "context.rtc.estimated_inference_delay_steps": "4  # optional",
            },
        })
        return metadata
