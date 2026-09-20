"""Interactive paired-Piper RTC demo, without episode or dataset machinery."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

from ruri.client._args import get_arg
from ruri.client.utils import inference_client
from ruri.client.utils.prompt_terminal import PromptTerminal

from ._actions import clip_target
from .assisted import AssistedScheduler
from .rtc_interactive import InteractiveRTCScheduler, _validate_switch


logger = logging.getLogger(__name__)
PAUSED = (
    "Both arms paused. Enter a task to resume, home, or switch <port>. "
    "Esc or Ctrl-C quits."
)


def policy_version(policy):
    metadata = policy.metadata
    return str(
        metadata.get("policy_version")
        or metadata.get("policy", {}).get("checkpoint_path")
        or policy.endpoint
    )


class InteractiveAssistedRTC:
    """Small operator loop around AssistedScheduler and its existing worker."""

    def run(self, *, controller, policy, args, terminal=None):
        robot = controller(args)
        owned_terminal = None
        scheduler = None
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="assisted-switch"
        )
        connections = []
        active_policy = policy
        switching = None
        resume_after_switch = True
        homing = False
        home_actions = None
        home_epoch = home_revision = None
        paused = False
        stats = {"steps": 0, "corrections": 0, "switches": 0, "homes": 0}

        def connect_candidate(endpoint):
            candidate = inference_client.connect(
                {
                    "policy_endpoint": endpoint,
                    "policy_timeout_s": get_arg(args, "policy_timeout_s", 5.0),
                    "policy_ready_timeout_s": 5.0,
                    "policy_ready_retry_s": 0.1,
                }
            )
            try:
                _validate_switch(active_policy, candidate)
            except Exception:
                candidate.close()
                raise
            connections.append(candidate)
            return candidate

        try:
            if terminal is None:
                owned_terminal = terminal = PromptTerminal()
                terminal.open()
            prompt = InteractiveRTCScheduler._first_prompt(terminal)
            if prompt is None:
                return stats
            hz = float(get_arg(args, "control_hz", 30.0))
            scheduler = AssistedScheduler(
                robot,
                policy,
                policy_version=policy_version(policy),
                prompt=prompt,
                control_hz=hz,
                execution_horizon=get_arg(args, "execution_horizon", 5),
                rtc_latency_window=get_arg(args, "rtc_latency_window", 10),
                rtc_initial_delay_steps=get_arg(args, "rtc_initial_delay_steps", 4),
                inference_timeout_s=get_arg(args, "policy_timeout_s", 5.0),
            )
            robot.start()
            logger.info(
                "Paired Piper: right/secondary leader, left/main follower; "
                "shared reference, 100 Hz MIT. Drag the right arm to correct; "
                "policy inference continues. q pauses. No dataset recording."
            )
            scheduler.resume()
            previous_intervention = (False, 0)
            next_tick = time.monotonic()
            while True:
                while (event := terminal.poll_event(0.0)) is not None:
                    if event.kind == "exit":
                        return stats
                    if switching is not None:
                        if event.kind == "pause":
                            resume_after_switch = False
                        continue
                    if event.kind == "pause":
                        scheduler.pause()
                        paused, homing, home_actions = True, False, None
                        terminal.request_prompt(PAUSED)
                    elif event.kind == "prompt" and paused and not homing:
                        words = event.text.split()
                        if words and words[0].casefold() == "switch":
                            try:
                                if len(words) != 2:
                                    raise ValueError(
                                        "Use switch <port>, e.g. switch 5558"
                                    )
                                endpoint = inference_client.endpoint_with_port(
                                    active_policy.endpoint, words[1]
                                )
                            except ValueError as error:
                                terminal.request_prompt(str(error))
                                continue
                            switching = executor.submit(connect_candidate, endpoint)
                            resume_after_switch = True
                            logger.info("Switching to %s; both arms hold", endpoint)
                        elif event.text.strip().casefold() == "home":
                            homing = True
                            home_actions = None
                            logger.info("Paired home at up to 1 rad/s; q pauses")
                        else:
                            prompt = event.text
                            scheduler.replace_policy(
                                active_policy,
                                policy_version(active_policy),
                                prompt=prompt,
                            )
                            scheduler.resume()
                            paused = False

                if switching is not None and switching.done():
                    try:
                        candidate = switching.result()
                        scheduler.replace_policy(candidate, policy_version(candidate))
                    except Exception as error:
                        logger.warning("Port switch failed: %s", error)
                        terminal.request_prompt(
                            f"Switch failed; still on {active_policy.endpoint}. {PAUSED}"
                        )
                    else:
                        previous = active_policy
                        active_policy = candidate
                        if previous is not policy:
                            executor.submit(previous.close)
                        stats["switches"] += 1
                        logger.info("Using %s, task=%r", active_policy.endpoint, prompt)
                        if resume_after_switch:
                            scheduler.resume()
                            paused = False
                        else:
                            terminal.request_prompt(PAUSED)
                    switching = None

                # This is an in-memory observation snapshot. No recorder,
                # episode, uploader or workflow package participates here.
                sample = robot.get_sample()
                if "observation.joint_effort" not in active_policy.metadata.get(
                    "inputs", {}
                ):
                    sample["observation"].pop("observation.joint_effort", None)
                scheduler.step(sample)
                state = robot.status()
                intervention = (state["intervention_active"], state["intervention_id"])
                if intervention != previous_intervention:
                    if intervention[0]:
                        stats["corrections"] += 1
                        logger.info(
                            "Right leader controls the follower; inference continues in background"
                        )
                    else:
                        logger.info(
                            "Correction released; continuing policy from current state"
                        )
                    previous_intervention = intervention

                if homing:
                    if home_actions is None:
                        if state["mode"] != "HOLD":
                            raise RuntimeError("Home requires paused shared control")
                        state = robot.request_mode("POLICY")
                        home_epoch = state["epoch"]
                        home_revision = state["reference_revision"]
                        home_actions = iter(robot.iter_home_actions(control_hz=hz))
                    if home_actions is not None:
                        if state["epoch"] != home_epoch:
                            raise RuntimeError("Control authority changed during home")
                        if state["reference_revision"] != home_revision:
                            scheduler.pause()
                            homing, home_actions = False, None
                            terminal.request_prompt(
                                "Human correction paused home. " + PAUSED
                            )
                        else:
                            try:
                                target, _ = clip_target(robot, next(home_actions))
                            except StopIteration:
                                scheduler.pause()
                                homing, home_actions = False, None
                                stats["homes"] += 1
                                terminal.request_prompt(
                                    "Both arms reached home. " + PAUSED
                                )
                            else:
                                robot.send_action(
                                    target,
                                    reference_revision=home_revision,
                                    observation_ns=time.monotonic_ns(),
                                )
                stats["steps"] += 1
                next_tick += 1.0 / hz
                time.sleep(max(0.0, next_tick - time.monotonic()))
                next_tick = max(next_tick, time.monotonic())
        finally:
            try:
                if scheduler is not None:
                    with suppress(RuntimeError, TimeoutError, OSError):
                        scheduler.close()
                robot.stop()
            finally:
                try:
                    executor.shutdown(wait=True)
                    for connection in connections:
                        connection.close()
                finally:
                    if owned_terminal is not None:
                        owned_terminal.close()


__all__ = ["InteractiveAssistedRTC"]
