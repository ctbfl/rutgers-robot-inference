"""Machine registrations for hardware whose Linux device names may change."""

from __future__ import annotations

from dataclasses import dataclass


#: The two physical arms on this rig. This is the label everything above the
#: CAN layer should use: an arm's travel, its calibration file and any log line
#: about it are all keyed on this, so a reader can tell which arm in the room a
#: piece of code is about without tracing a SocketCAN name back to a cable.
ARM_NAMES = ("left_arm", "right_arm")


@dataclass(frozen=True, slots=True)
class ArmHardwareRegistration:
    """Bind one USB-CAN adapter serial to one physical arm.

    Two separate things meet here and should not be confused: ``name`` is which
    arm in the room, and ``can_hardware_id`` is only how we recognise it. The
    adapter serial exists to answer "which cable is this arm on today"; it is
    not the arm's identity.
    """

    can_hardware_id: str
    side: str
    role: str

    @property
    def name(self) -> str:
        """``left_arm`` or ``right_arm`` -- the arm, not the interface."""
        return f"{self.side}_arm"


# These IDs use the USB adapter's own serial, not its transient SocketCAN name
# or USB port. Moving an adapter can therefore rename can0/can1 without changing
# which physical arm this controller considers left/main or right/secondary.
REGISTERED_ARMS = (
    ArmHardwareRegistration(
        can_hardware_id="usb:1d50:606f:0042002F4759530820353131",
        side="left",
        role="main",
    ),
    ArmHardwareRegistration(
        can_hardware_id="usb:1d50:606f:002B00464759530920353131",
        side="right",
        role="secondary",
    ),
)


def find_arm_registration(side: str, role: str) -> ArmHardwareRegistration:
    matches = [item for item in REGISTERED_ARMS if item.side == side and item.role == role]
    if len(matches) != 1:
        raise RuntimeError(
            f"No unique Single Piper hardware registration for side={side!r} role={role!r}"
        )
    return matches[0]


def find_arm_by_hardware_id(can_hardware_id: str) -> ArmHardwareRegistration:
    matches = [item for item in REGISTERED_ARMS if item.can_hardware_id == can_hardware_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"CAN hardware ID {can_hardware_id!r} is not registered to one Single Piper arm"
        )
    return matches[0]


def find_arm_by_name(name: str) -> ArmHardwareRegistration:
    """Look up an arm by ``left_arm`` / ``right_arm``."""
    matches = [item for item in REGISTERED_ARMS if item.name == name]
    if len(matches) != 1:
        raise RuntimeError(
            f"No unique Single Piper hardware registration named {name!r}; "
            f"known arms are {sorted(a.name for a in REGISTERED_ARMS)}"
        )
    return matches[0]
