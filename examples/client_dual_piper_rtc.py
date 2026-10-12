#!/usr/bin/env python
"""Interactive Pi0.5 RTC with synchronized Piper arms and right-arm intervention.

Enter a task to enable/home both arms and start RTC. Press q to pause and enter
a new task, home, or switch PORT. Esc/Ctrl-C holds and exits. The left arm
provides state/top/wrist observations; closing the right gripper by 1cm requests
local takeover, after matching the left gripper. It reopens on policy resumption.
The local gripper range is 0-100 mm; policy gripper values remain 0-100.
No dataset, intervention, video or effort files are recorded.
"""

from client_single_piper_assisted_rtc_interactive import main


if __name__ == "__main__":
    main()
