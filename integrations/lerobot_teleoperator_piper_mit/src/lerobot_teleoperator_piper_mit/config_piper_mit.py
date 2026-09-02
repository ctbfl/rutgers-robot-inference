from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig

from ruri.client.controllers.single_piper.mit_io import parse_local_udp


@TeleoperatorConfig.register_subclass("piper_mit")
@dataclass(kw_only=True)
class PiperMITConfig(TeleoperatorConfig):
    telemetry_address: str = "udp://127.0.0.1:6670"
    telemetry_timeout_s: float = 0.25

    def __post_init__(self) -> None:
        parse_local_udp(self.telemetry_address)
        if not 0.05 <= self.telemetry_timeout_s <= 2.0:
            raise ValueError("telemetry_timeout_s must be between 0.05 and 2.0")
