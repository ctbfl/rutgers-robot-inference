"""
Base interface for policy wrappers in Rutgers Robot Inference (RURI).

A PolicyWrapper adapts an arbitrary policy implementation to the RURI
inference stack. The underlying policy may use JAX, PyTorch, LeRobot,
OpenPI, or any other framework.

RURI uses flexible dictionaries as its inference interface:

    inputs: dict -> infer() -> response: dict

Key layers
----------
There are three layers, and keeping them apart is what lets a client
discover what a policy wants without knowing anything about the policy.

    1. RURI input key layer      flat, what the client sends
    2. Policy input key layer    flat, the policy's own key names
    3. Policy real input         whatever the backend actually consumes,
                                 nested or not

RURI owns 1 <-> 2, through INPUT_MAPPING. The wrapper author owns 2 <-> 3,
in whatever code their backend needs; RURI never sees layer 3.

Layer 1 and 2 are both flat, with dot-separated names expressing logical
namespaces, for example:

    observation.state
    observation.images.top
    observation.images.wrist
    prompt
    context.current_chunk

Two independent declarations
----------------------------
INPUT_MAPPING renames keys. POLICY_METADATA describes them. They are
deliberately separate: renaming and typing are different concerns, and a
wrapper whose policy already speaks RURI naming needs only the second.

    INPUT_MAPPING     RURI input key -> policy input key. Optional; the
                      default is identity. Must be injective, so that the
                      inverse is well defined (see ruri_metadata).

    POLICY_METADATA   Describes the policy input key layer, i.e. it is
                      keyed by layer-2 names. Named for the policy, not
                      for inputs, so that an output section can join it
                      later without a rename.

Values are forwarded transparently. They may be arrays, strings, nested
dictionaries, or any other data supported by the transport layer.
"""

from abc import ABC, abstractmethod
from typing import Any


# Global key in a describe() result under which a wrapper's own free-form
# metadata is namespaced, so that it can never collide with a global key.
POLICY_METADATA_KEY = "policy"


class PolicyWrapper(ABC):
    """
    Base class for all RURI policy wrappers.

    Concrete wrappers only need to:

        1. Declare POLICY_METADATA describing the policy's own input keys.
        2. Override INPUT_MAPPING when the policy's key names differ from
           the RURI convention.
        3. Implement _infer() with policy-specific inference logic.

    Model loading, preprocessing, inference, and postprocessing are left to
    the concrete wrapper.

    Policy wrappers should not manage robot hardware or inference timing.
    Those responsibilities belong to RobotSetupController and
    InferenceScheduler on the client side.
    """

    # RURI input key -> policy input key. Keys not listed here are forwarded
    # unchanged, in both directions. Must be injective, since ruri_metadata()
    # inverts it; __init_subclass__ enforces that.
    INPUT_MAPPING: dict[str, str] = {
        "observation.state": "observation.state",
        "observation.images.top": "observation.images.top",
        "observation.images.wrist": "observation.images.wrist",
        "prompt": "prompt",
    }

    # Describes the policy input key layer, so keyed by policy names, not
    # RURI ones -- ruri_metadata() translates:
    #
    #     POLICY_METADATA = {
    #         "inputs": {
    #             "observation/state": {"type": "state"},
    #             "observation/image": {"type": "image"},
    #             "prompt": {"type": "string"},
    #         },
    #     }
    POLICY_METADATA: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject a non-injective INPUT_MAPPING when the subclass is defined."""
        super().__init_subclass__(**kwargs)

        seen: dict[str, str] = {}
        collisions: list[str] = []
        for ruri_key, policy_key in cls.INPUT_MAPPING.items():
            if policy_key in seen:
                collisions.append(
                    f"{seen[policy_key]!r} and {ruri_key!r} both map to {policy_key!r}"
                )
            else:
                seen[policy_key] = ruri_key

        if collisions:
            raise TypeError(
                f"{cls.__name__}.INPUT_MAPPING must be injective so that its "
                "inverse is well defined: " + "; ".join(collisions)
            )

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Run one RURI inference request.

        INPUT_MAPPING is applied only to the top-level keys of `inputs`.
        Values are forwarded unchanged, including nested dictionaries.

        Example:

            {
                "observation.state": ...,
                "observation.images.top": ...,
                "prompt": "pick up the red block",
            }

        may be mapped by a concrete wrapper to:

            {
                "observation/state": ...,
                "observation/image": ...,
                "prompt": "pick up the red block",
            }

        before being passed to `_infer()`.

        Args:
            inputs:
                Flexible inference request from the RURI client, in RURI
                input key naming.

        Returns:
            Flexible inference response produced by the concrete policy
            wrapper.
        """
        mapped_inputs = {
            self.INPUT_MAPPING.get(key, key): value
            for key, value in inputs.items()
        }

        return self._infer(mapped_inputs)

    @abstractmethod
    def _infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Run policy-specific inference on already-mapped inputs.

        `inputs` is in policy input key naming (layer 2). Translating from
        there to whatever the backend actually consumes (layer 3) is this
        method's job.
        """
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """
        Everything a client needs in order to talk to this policy.

        This is what a ``metadata`` request returns. RURI defines the
        top-level keys, and they mean the same thing for every policy:

            inputs      Standard, always present, safe to rely on.

            policy      Whatever this wrapper registered through
                        optional_more_metadata(). Free-form, and omitted
                        when the wrapper registers nothing.

        Subclasses fill the ``policy`` slot by overriding
        optional_more_metadata() rather than this method, so that the
        top-level structure stays identical across policies.
        """
        description = self.ruri_metadata()

        extra = self.optional_more_metadata()
        if extra:
            description[POLICY_METADATA_KEY] = extra

        return description

    def optional_more_metadata(self) -> dict[str, Any]:
        """
        Extra, wrapper-specific facts to publish under the ``policy`` key:
        which checkpoint is loaded, sampling settings, build info.
        """
        return {}

    @classmethod
    def ruri_metadata(cls) -> dict[str, Any]:
        """
        POLICY_METADATA restated in RURI input keys, i.e. what a client
        needs to know in order to name what it sends.

        The input section goes through the inverse of INPUT_MAPPING; a key
        the mapping does not cover keeps its name. Other sections pass
        through untouched.
        """
        inverse = {policy: ruri for ruri, policy in cls.INPUT_MAPPING.items()}

        metadata = dict(cls.POLICY_METADATA)
        policy_inputs = metadata.get("inputs")
        if isinstance(policy_inputs, dict):
            metadata["inputs"] = {
                inverse.get(key, key): value for key, value in policy_inputs.items()
            }

        return metadata
