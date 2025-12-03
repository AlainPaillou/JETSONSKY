"""
Timing State - Performance and timing state.
"""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime


@dataclass
class TimingState:
    """
    Holds performance timing state.

    This includes FPS tracking, execution timing, and timestamp info.
    """
    # FPS tracking
    curFPS: float = 0.0
    fpsQueue: List[float] = field(default_factory=list)

    # Processing time tracking
    curTT: float = 0.0
    TTQueue: List[float] = field(default_factory=list)

    # Total timing
    total_start: float = 0.0
    total_stop: float = 0.0
    total_time: float = 1.0

    # Execution test timing
    time_exec_test: float = 0.0
    timer1: float = 0.0
    val_deltat: int = 0

    # Date/time stamp for images
    Date_hour_image: str = field(
        default_factory=lambda: datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')
    )

    def update_fps(self, new_fps: float, max_queue_size: int = 10):
        """Update FPS tracking with new value."""
        self.fpsQueue.append(new_fps)
        if len(self.fpsQueue) > max_queue_size:
            self.fpsQueue.pop(0)
        if self.fpsQueue:
            self.curFPS = sum(self.fpsQueue) / len(self.fpsQueue)

    def update_tt(self, new_tt: float, max_queue_size: int = 10):
        """Update processing time tracking with new value."""
        self.TTQueue.append(new_tt)
        if len(self.TTQueue) > max_queue_size:
            self.TTQueue.pop(0)
        if self.TTQueue:
            self.curTT = sum(self.TTQueue) / len(self.TTQueue)

    def update_timestamp(self):
        """Update the current timestamp."""
        self.Date_hour_image = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')
