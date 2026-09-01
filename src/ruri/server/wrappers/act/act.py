"""
ACT (Action Chunking Transformer) policy wrapper, backed by LeRobot.

All of the machinery lives in
:class:`~ruri.server.wrappers.lerobot_base.LeRobotWrapper`. ACT needs only the
key mapping and its type declaration, because it is the simplest case LeRobot
serves: one observation step, no language conditioning, absolute joint targets.

What is ACT-specific
--------------------
**The wrist camera is called ``hand``.** The tight_insertion_E1 ACT checkpoint
declares ``observation.images.top`` and ``observation.images.hand``, where the
Pi0.5 side of this repo uses ``wrist``. :attr:`ACTWrapper.INPUT_MAPPING` absorbs
that difference so a client sends ``observation.images.wrist`` to either policy
and does not care which is loaded. If you point this wrapper at an ACT
checkpoint whose cameras are named differently, the base class's load-time check
will refuse to start rather than silently drop a camera.

**Chunks are long.** chunk_size is 100, which at this dataset's 30 fps is 3.3
seconds of motion -- ten times the Pi0.5 config's horizon. Combined with ~9 ms
inference, the scheduler has an enormous amount of slack here compared to Pi0.5,
where a whole chunk was 333 ms. Expect to use ``context.actions_per_chunk``, or
to re-plan long before the chunk runs out.

**Actions are absolute joint targets**, not deltas relative to the current
state. There is no delta transform anywhere in the LeRobot ACT pipeline, so the
returned chunk can go to the arm as-is.

**Inference is stateless.** LeRobot's ``ACTPolicy.select_action`` keeps an
internal action queue and, when configured for it, a temporal ensembler --
either would make the server carry hidden per-episode state. The base class
calls ``predict_action_chunk`` instead, which touches neither, so no reset
between episodes is needed and concurrent clients cannot corrupt each other.
Note that this also means ``temporal_ensemble_coeff`` is ignored; that
checkpoint sets it to ``None`` anyway, and ensembling belongs in the scheduler.

Runtime requirements
--------------------
Needs the LeRobot environment, not the OpenPI one. See the base class docstring.
"""

from __future__ import annotations

from typing import Any

from ruri.server.wrappers.lerobot_base import LeRobotWrapper


# ACT's own feature keys, i.e. what the checkpoint declares.
ACT_STATE = "observation.state"
ACT_TOP_IMAGE = "observation.images.top"
ACT_HAND_IMAGE = "observation.images.hand"


class ACTWrapper(LeRobotWrapper):
    """
    Serve a LeRobot ACT checkpoint through the RURI interface.

    Example:
        >>> wrapper = ACTWrapper(
        ...     "/common/users/jh2400/lerobot_outputs/tight_insertion_E1_act"
        ...     "/checkpoints/100000/pretrained_model",
        ... )
        >>> response = wrapper.infer({
        ...     "observation.state": np.zeros(7, dtype=np.float32),
        ...     "observation.images.top": np.zeros((480, 640, 3), dtype=np.uint8),
        ...     "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        ... })
        >>> response["action_chunk"].shape
        (100, 7)

    See :class:`~ruri.server.wrappers.lerobot_base.LeRobotWrapper` for the
    constructor arguments.
    """

    POLICY_TYPE = "act"

    # The one thing ACT needs beyond the base: `wrist` on the wire, `hand` in
    # the checkpoint. Keeping the RURI name stable across policies is the whole
    # point of this layer.
    INPUT_MAPPING = {
        "observation.state": ACT_STATE,
        "observation.images.top": ACT_TOP_IMAGE,
        "observation.images.wrist": ACT_HAND_IMAGE,
    }

    POLICY_METADATA = {
        "inputs": {
            ACT_STATE: {"type": "state"},
            ACT_TOP_IMAGE: {"type": "image"},
            ACT_HAND_IMAGE: {"type": "image"},
        },
    }

    def optional_more_metadata(self) -> dict[str, Any]:
        metadata = super().optional_more_metadata()
        # Worth publishing explicitly: a client that also talks to Pi0.5 gets
        # chunks an order of magnitude longer here, and they mean something
        # different (absolute targets, not deltas).
        metadata.update(
            {
                "name": "act",
                "action_space": "absolute_joint_targets",
                "chunk_size": self.config.chunk_size,
                "temporal_ensemble_coeff": self.config.temporal_ensemble_coeff,
            }
        )
        return metadata
