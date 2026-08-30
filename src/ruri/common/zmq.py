"""
MessagePack codec for RURI's ZeroMQ transport.

RURI passes flat dictionaries whose values are mostly numpy arrays -- camera
frames, robot state, action chunks. MessagePack has no native array type, so
this module adds one, and provides the socket helpers the server and client
use instead of pyzmq's ``send_pyobj`` / ``recv_pyobj``.

MessagePack rather than pickle because ``recv_pyobj`` unpickles whatever
arrives, letting anyone who can reach the port execute arbitrary code.

Wire format
-----------
A numpy array travels as a single-key map carrying its raw C-order buffer::

    {"__ndarray__": {"dtype": "|u1", "shape": [480, 640, 3], "data": b"..."}}

``dtype`` is numpy's typestr, so it carries byte order and survives a move
between machines.

A value the codec does not understand becomes a marker rather than an error,
so that ``pack`` cannot fail -- it is called at the point where a REP socket
has already committed to replying::

    {"__unencodable__": {"type": "PosixPath", "repr": "PosixPath('/tmp/x')"}}

Both marker keys are reserved and must not appear in a payload of your own.
"""

from __future__ import annotations

import logging
from typing import Any

import msgpack
import numpy as np
import zmq


logger = logging.getLogger(__name__)


NDARRAY_KEY = "__ndarray__"
UNENCODABLE_KEY = "__unencodable__"


def _default(obj: Any) -> Any:
    """Encode a value MessagePack does not handle natively."""
    if isinstance(obj, np.ndarray):
        # tobytes() is always C-order, so views and transposes need no flag.
        return {
            NDARRAY_KEY: {
                "dtype": obj.dtype.str,
                "shape": list(obj.shape),
                "data": obj.tobytes(),
            }
        }

    # np.float32, np.int64: scalars that leak out of numpy math.
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)

    logger.warning(
        "Dropping unencodable value of type %s from a RURI message", type(obj).__name__
    )
    return {UNENCODABLE_KEY: {"type": type(obj).__name__, "repr": repr(obj)}}


def _object_hook(obj: dict) -> Any:
    """Rebuild a numpy array from its wire form; pass everything else through."""
    spec = obj.get(NDARRAY_KEY)
    if spec is None:
        return obj

    array = np.frombuffer(spec["data"], dtype=np.dtype(spec["dtype"]))
    # frombuffer aliases the immutable payload, so the result is read-only.
    # Copy so downstream code can write in place.
    return array.reshape(spec["shape"]).copy()


def pack(obj: Any) -> bytes:
    """Serialize a RURI message. Never raises on unsupported types."""
    return msgpack.packb(obj, default=_default, use_bin_type=True)


def unpack(data: bytes) -> Any:
    """Deserialize a RURI message. Raises on a malformed payload."""
    return msgpack.unpackb(data, object_hook=_object_hook, raw=False)


def send(socket: zmq.Socket, obj: Any) -> None:
    """Send one RURI message."""
    socket.send(pack(obj))


def recv(socket: zmq.Socket) -> Any:
    """Receive one RURI message."""
    return unpack(socket.recv())
