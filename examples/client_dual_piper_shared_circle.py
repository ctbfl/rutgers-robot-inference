#!/usr/bin/env python
"""Both arms share a circle goal; drag the right arm to lead the left.

--execute enables/homes both registered arms, then runs until Ctrl-C.
Default: offline audit with both arms' bounds and gravity calibrations.
Left uses the original single-Piper MIT gains and calibrated gravity feed-forward.
--simulate --simulate-intervention --seconds 9 exercises the complete lifecycle
without opening CAN or cameras. Right-only entry remains available separately.
"""

from client_single_piper_shared_circle import main


if __name__ == "__main__":
    raise SystemExit(main(dual=True))
