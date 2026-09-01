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

Announcing itself
-----------------
When started with a name, the server writes an entry into the RURI runtime
directory once it is able to serve, and starts the menu process if nothing
holds its port yet. Both are best-effort: a server that cannot announce itself
still serves. See :mod:`ruri.server.registry` for why it never removes that
entry itself.

Press Ctrl+C to stop the server.
"""

import argparse
import logging

import zmq

from ruri.common.zmq import recv, send
from ruri.server import registry
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
    server_info: dict | None = None,
) -> dict:
    """
    Dispatch one request on its "type" field.

        {"type": "infer", <ruri input keys...>}  -> policy.infer(inputs)
        {"type": "metadata"}                     -> policy.describe()

    "type" is an envelope field, not policy input, so it is stripped before
    the rest of the request reaches the wrapper. It defaults to "infer".

    `server_info` carries facts about this process rather than about the
    policy -- the name it is registered under, the address it is bound to. It
    joins the metadata response as its own section rather than going under
    "policy", which is the wrapper's slot: a wrapper has no idea it is being
    served over a socket, and should not have to.
    """

    request_type = request.get(REQUEST_TYPE_KEY, TYPE_INFER)

    if request_type == TYPE_METADATA:
        description = policy.describe()
        if server_info:
            description["server"] = dict(server_info)
        return description

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
    name: str | None = None,
    advertise_host: str | None = None,
) -> None:
    """
    Serve a PolicyWrapper over ZeroMQ.

    The server follows a simple REP loop:

        receive request
            ↓
        dispatch on request["type"]
            ↓
        send response

    Args:
        name:
            Register under this name so the menu can list this server. It
            identifies the *instance*, not the wrapper class -- two
            Pi05Wrapper servers on two checkpoints both call themselves
            "pi05", which is useless for picking one. Omit to run
            unannounced.
        advertise_host:
            Hostname to publish in place of the bind wildcard. Defaults to
            this machine's own; pass it when that is not what the robot can
            resolve.
    """

    context = zmq.Context()
    socket = context.socket(zmq.REP)

    server_info: dict | None = None

    try:
        socket.bind(bind_address)

        # Only now can this process answer anything: loading the checkpoint
        # took ~22 s for Pi0.5, and a client that had found it in the menu
        # during that window would just have timed out.
        if name:
            endpoint = registry.advertised_endpoint(bind_address, advertise_host)
            server_info = {"name": name, "endpoint": endpoint}
            registry.register(name, endpoint, policy.describe())

        print(f"[RURI] Inference server ready: {bind_address}")
        if name:
            print(f"[RURI] Registered as {name!r} ({server_info['endpoint']})")

        while True:
            # A REP socket must answer every request it receives, or it locks
            # into a "must send" state and the next recv fails outright. So
            # every failure turns into a reply rather than escaping the loop.
            try:
                request = recv(socket)
                response = handle_request(policy, request, server_info)
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