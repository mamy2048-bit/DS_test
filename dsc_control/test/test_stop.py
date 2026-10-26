import os
import signal

import pytest

from dsc_control.nodes.stop import raise_on_sigterm, send_stop


def test_send_stop_repeats():
    calls = []
    assert send_stop(lambda: calls.append(1), count=5, interval_s=0.0) == 5
    assert len(calls) == 5


def test_send_stop_stops_when_context_is_gone():
    calls = []

    def publish():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("context invalid")
    assert send_stop(publish, count=5, interval_s=0.0) == 1
    assert len(calls) == 2


def test_sigterm_becomes_keyboard_interrupt():
    old = signal.getsignal(signal.SIGTERM)
    try:
        raise_on_sigterm()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)  # the handler runs before the next bytecode
    finally:
        signal.signal(signal.SIGTERM, old)
