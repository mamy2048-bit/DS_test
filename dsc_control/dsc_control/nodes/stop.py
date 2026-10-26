"""Shutdown helpers for the nodes that can command the robot.

m2_driver_node keeps executing the last /cmd_vel it received, so a node that stops publishing must
leave a zero behind, and that zero must actually get out of the process:
  - SIGTERM (kill, ros2 launch escalating after SIGINT) would end the process without running
    `finally`; raise_on_sigterm() turns it into KeyboardInterrupt like Ctrl-C.
  - A single publish right before exit can be dropped; send_stop() repeats it with short pauses.
Nodes must call rclpy.init(signal_handler_options=SignalHandlerOptions.NO) so the context is still
valid in `finally`.
"""
import signal
import time
from typing import Callable


def raise_on_sigterm() -> None:
    def _raise(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _raise)


def send_stop(publish_zero: Callable[[], None], count: int = 5, interval_s: float = 0.02) -> int:
    """Call publish_zero `count` times, `interval_s` apart. Returns how many publishes succeeded."""
    sent = 0
    for i in range(count):
        try:
            publish_zero()
        except Exception:  # context already gone: nothing more can be sent
            break
        sent += 1
        if i < count - 1:
            time.sleep(interval_s)
    return sent
