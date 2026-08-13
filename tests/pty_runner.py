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


def run_under_pty(script: str, env: dict[str, str]) -> str:
    primary, secondary = pty.openpty()
    fcntl.ioctl(secondary, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 200, 0, 0))
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, script],
        stdin=subprocess.DEVNULL,
        stdout=secondary,
        stderr=secondary,
        env=env,
    )
    os.close(secondary)
    chunks = []
    deadline = time.monotonic() + _PTY_READ_TIMEOUT
    timed_out = False
    try:
        while True:
            # A child hanging with the pty still open never reaches EIO, and an untimed
            # read would then hang the whole suite instead of failing this one test.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if not select.select([primary], [], [], remaining)[0]:
                continue
            try:
                data = os.read(primary, 65536)
            except OSError:  # the terminal reports EIO once the child is gone
                break
            if not data:
                break
            chunks.append(data)
    finally:
        os.close(primary)
        if timed_out:
            process.kill()
        process.wait(timeout=10)
    if timed_out:
        pytest.fail(f"child did not exit within {_PTY_READ_TIMEOUT}s")
    return b"".join(chunks).decode(errors="replace")
