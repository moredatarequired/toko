"""Run a script attached to a real terminal, since rich renders differently without one."""

import os
import select
import struct
import subprocess
import sys
import time

import pytest

try:
    import fcntl
    import pty
    import termios

    HAS_PTY = True
except ImportError:  # pty, termios and fcntl are POSIX-only
    HAS_PTY = False

PTY_SKIP_REASON = "pty, termios and fcntl are POSIX-only"

_PTY_READ_TIMEOUT = 60.0


def run_under_pty(
    script: str,
    env: dict[str, str],
    *,
    winsize: tuple[int, int] = (50, 200),
    pipe_stdout: bool = False,
) -> str:
    """Return what the child wrote: the pipe with `pipe_stdout`, the terminal otherwise.

    With `pipe_stdout` the terminal is left on stdin and stderr, which is how a shell
    hands over a command whose output is redirected.
    """
    primary, secondary = pty.openpty()
    rows, columns = winsize
    fcntl.ioctl(secondary, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, script],
        stdin=secondary if pipe_stdout else subprocess.DEVNULL,
        stdout=subprocess.PIPE if pipe_stdout else secondary,
        stderr=secondary,
        env=env,
    )
    os.close(secondary)
    # The terminal is drained either way, since a child blocked on a full pty would
    # otherwise never exit.
    wanted: int = process.stdout.fileno() if process.stdout is not None else primary
    open_sources: list[int] = [primary] if wanted == primary else [primary, wanted]
    chunks = []
    deadline = time.monotonic() + _PTY_READ_TIMEOUT
    timed_out = False
    try:
        # Not `while open_sources`: ty then narrows the list and misinfers select's
        # typevars, which fails `just typecheck`.
        while len(open_sources) > 0:
            # A child hanging with the pty still open never reaches EIO, and an untimed
            # read would then hang the whole suite instead of failing this one test.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            for source in select.select(open_sources, [], [], remaining)[0]:
                try:
                    data = os.read(source, 65536)
                except OSError:  # the terminal reports EIO once the child is gone
                    data = b""
                if not data:
                    open_sources.remove(source)
                elif source == wanted:
                    chunks.append(data)
    finally:
        os.close(primary)
        if process.stdout is not None:
            process.stdout.close()
        if timed_out:
            process.kill()
        process.wait(timeout=10)
    if timed_out:
        pytest.fail(f"child did not exit within {_PTY_READ_TIMEOUT}s")
    return b"".join(chunks).decode(errors="replace")
