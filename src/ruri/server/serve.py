# 用途：加载某个policy wrapper， 然后启动zmq服务进行serve. 

"""
RURI inference server.

Usage:
    1. Load a policy wrapper.
    2. Start a ZeroMQ REP server.
    3. Dispatch each request on its "type" field.
    4. Return the result to the client.

Request types:

    {"type": "infer", "observation.state": ..., "observation.images.top": ...}
        Normal inference. Everything except "type" is forwarded to
        policy.infer() as RURI input keys.

    {"type": "metadata"}
        Publish what this policy expects and what it is serving, via
        policy.describe(). A client calls this to discover the input
        contract instead of hard-coding it.

Press Ctrl+C to stop the server.
"""

import argparse
import logging

import zmq

from ruri.common.zmq import recv, send
from ruri.server.wrappers.policy_wrapper import PolicyWrapper


# Request envelope. The client tags every request with its kind; everything
# else in the dict is payload.
REQUEST_TYPE_KEY = "type"
TYPE_INFER = "infer"
TYPE_METADATA = "metadata"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RURI policy inference server."
    )

    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Policy wrapper to load, e.g. pi05.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the policy checkpoint.",
    )

    parser.add_argument(
        "--bind",
        type=str,
        default="tcp://*:5555",
        help="ZeroMQ bind address.",
    )

    return parser.parse_args()


def build_wrapper(args: argparse.Namespace) -> PolicyWrapper:
    """
    Build the requested policy wrapper.

    Policy-specific dependencies are intentionally imported lazily so that
    RURI itself does not require every policy backend to be installed.
    """

    if args.policy == "pi05":
        from ruri.server.wrappers.pi05.pi05 import Pi05Wrapper

        return Pi05Wrapper(
            checkpoint=args.checkpoint,
        )

    raise ValueError(
        f"Unknown policy wrapper: {args.policy}"
    )


def handle_request(
    policy: PolicyWrapper,
    request: dict,
) -> dict:
    """
    Dispatch one request on its "type" field.

        {"type": "infer", <ruri input keys...>}  -> policy.infer(inputs)
        {"type": "metadata"}                     -> policy.describe()

    "type" is an envelope field, not policy input, so it is stripped before
    the rest of the request reaches the wrapper. It defaults to "infer".
    """

    request_type = request.get(REQUEST_TYPE_KEY, TYPE_INFER)

    if request_type == TYPE_METADATA:
        return policy.describe()

    if request_type == TYPE_INFER:
        inputs = {k: v for k, v in request.items() if k != REQUEST_TYPE_KEY}
        return policy.infer(inputs)

    return {
        "error": (
            f"unknown request type {request_type!r}; "
            f"expected one of {TYPE_INFER!r}, {TYPE_METADATA!r}"
        )
    }


def serve(
    policy: PolicyWrapper,
    bind_address: str,
) -> None:
    """
    Serve a PolicyWrapper over ZeroMQ.

    The server follows a simple REP loop:

        receive request
            ↓
        dispatch on request["type"]
            ↓
        send response
    """

    context = zmq.Context()
    socket = context.socket(zmq.REP)

    try:
        socket.bind(bind_address)

        print(f"[RURI] Inference server ready: {bind_address}")

        while True:
            # A REP socket must answer every request it receives, or it locks
            # into a "must send" state and the next recv fails outright. So
            # every failure turns into a reply rather than escaping the loop.
            try:
                request = recv(socket)
                response = handle_request(policy, request)
            except Exception as exc:
                logging.exception("Request failed")
                response = {"error": f"{type(exc).__name__}: {exc}"}

            send(socket, response)

    except KeyboardInterrupt:
        print("\n[RURI] Stopping inference server...")

    finally:
        socket.close(linger=0)
        context.term()


def main() -> None:
    """
    Load a policy wrapper and expose it through ZeroMQ.
    """

    args = parse_args()

    print(f"[RURI] Loading policy: {args.policy}")

    policy = build_wrapper(args)

    print("[RURI] Policy ready.")

    serve(
        policy=policy,
        bind_address=args.bind,
    )


if __name__ == "__main__":
    main()