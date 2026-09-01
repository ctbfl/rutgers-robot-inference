"""Machine registrations for hardware whose Linux device names may change."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArmHardwareRegistration:
    """Bind one USB-CAN adapter serial to its physical arm role."""

    can_hardware_id: str
    side: str
    role: str


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
