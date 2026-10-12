"""Read single keys and whole prompt lines from the controlling terminal.

The interactive RTC client must keep commanding the arm while a person types:
once commanded, the MIT worker aborts the run when no command arrives for its
command timeout. So stdin is read on a background thread in cbreak mode (no
line buffering and no echo, while Ctrl-C still raises SIGINT), and the control
loop only polls for finished events.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import termios
import threading
import tty
from collections import deque
from dataclasses import dataclass
from typing import Any, TextIO


logger = logging.getLogger(__name__)

ESC = b"\x1b"
# An arrow or function key is ESC with more bytes arriving right behind it; a
# lone Esc press has nothing behind it.
ESCAPE_SEQUENCE_WAIT_S = 0.05
PROMPT_LABEL = "prompt> "


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    kind: str  # "pause", "exit" or "prompt"
    text: str | None = None


class _RedrawingStream:
    """Log stream that keeps a half-typed prompt line intact below the logs."""

    def __init__(self, stream: TextIO, terminal: "PromptTerminal"):
        self.stream = stream
        self._terminal = terminal

    def write(self, text: str) -> int:
        self._terminal._write_log(self.stream, text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()


class PromptTerminal:
    """Turn ``q``, ``Esc`` and typed prompt lines into polled events.

    Outside prompt entry, ``q`` posts ``pause`` and every other key is ignored.
    After ``request_prompt`` the keys edit one line: Enter posts ``prompt`` with
    the text, or asks again when the line is empty. ``Esc`` posts ``exit`` in
    both modes.
    """

    def __init__(self, *, input_fd: int | None = None, output_fd: int | None = None):
        self._input_fd = sys.stdin.fileno() if input_fd is None else input_fd
        self._output_fd = sys.stdout.fileno() if output_fd is None else output_fd
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._events: deque[TerminalEvent] = deque()
        self._line_mode = False
        self._buffer = bytearray()
        self._saved_mode: list[Any] | None = None
        self._handlers: list[tuple[logging.StreamHandler, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        if not os.isatty(self._input_fd):
            raise RuntimeError(
                "The interactive client reads q, Esc and prompts from a terminal, "
                "but stdin is not a TTY"
            )
        self._saved_mode = termios.tcgetattr(self._input_fd)
        tty.setcbreak(self._input_fd)
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                original = handler.setStream(_RedrawingStream(handler.stream, self))
                self._handlers.append((handler, original))
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="ruri-prompt-terminal",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            if self._line_mode:
                self._line_mode = False
                self._write(b"\n")
        for handler, original in self._handlers:
            handler.setStream(original)
        self._handlers.clear()
        if self._saved_mode is not None:
            termios.tcsetattr(self._input_fd, termios.TCSADRAIN, self._saved_mode)
            self._saved_mode = None

    def __enter__(self) -> "PromptTerminal":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def request_prompt(self, message: str) -> None:
        """Print ``message`` and start editing a new prompt line."""
        with self._lock:
            # A q pressed twice must not pause again the moment this is answered.
            self._events = deque(
                event for event in self._events if event.kind != "pause"
            )
            self._buffer.clear()
            self._line_mode = True
            self._write(f"\n{message}\n{PROMPT_LABEL}".encode())

    def poll_event(self, timeout_s: float = 0.0) -> TerminalEvent | None:
        """Return the oldest event, waiting at most ``timeout_s`` for one."""
        with self._changed:
            if not self._events and timeout_s > 0:
                self._changed.wait(timeout_s)
            return self._events.popleft() if self._events else None

    def _read_loop(self) -> None:
        fd = self._input_fd
        while not self._stop.is_set():
            try:
                if not select.select([fd], [], [], 0.1)[0]:
                    continue
                data = os.read(fd, 1)
                if data == ESC and select.select([fd], [], [], ESCAPE_SEQUENCE_WAIT_S)[0]:
                    os.read(fd, 64)  # arrow or function key
                    continue
            except OSError:
                # Without this reader neither q nor Esc works any more, so end
                # the run rather than keep driving the arm deaf to the operator.
                logger.exception("Terminal input failed; quitting")
                data = b""
            if not data:
                with self._lock:
                    self._post(TerminalEvent("exit"))
                return
            self._feed(data)

    def _feed(self, data: bytes) -> None:
        with self._lock:
            if data == ESC:
                self._post(TerminalEvent("exit"))
                return
            if not self._line_mode:
                if data in (b"q", b"Q"):
                    self._post(TerminalEvent("pause"))
                return
            if data in (b"\r", b"\n"):
                text = self._buffer.decode("utf-8", errors="replace").strip()
                self._buffer.clear()
                if not text:
                    self._write(f"\n{PROMPT_LABEL}".encode())
                    return
                self._line_mode = False
                self._write(b"\n")
                self._post(TerminalEvent("prompt", text))
            elif data in (b"\x7f", b"\x08"):
                text = self._buffer.decode("utf-8", errors="ignore")
                if text:
                    self._buffer[:] = text[:-1].encode("utf-8")
                    self._write(b"\b \b")
            elif data[0] >= 0x20:
                self._buffer += data
                self._write(data)

    def _post(self, event: TerminalEvent) -> None:
        self._events.append(event)
        self._changed.notify_all()

    def _write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            view = view[os.write(self._output_fd, view):]

    def _write_log(self, stream: TextIO, text: str) -> None:
        with self._lock:
            if not self._line_mode:
                stream.write(text)
                return
            self._write(b"\r\x1b[K")
            stream.write(text)
            stream.flush()
            if text.endswith("\n"):
                self._write(PROMPT_LABEL.encode() + bytes(self._buffer))


__all__ = ["PromptTerminal", "TerminalEvent"]
