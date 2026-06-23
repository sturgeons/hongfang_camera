"""
AprilTag 车辆轨迹追踪与进出库判定。

判定规则（图像坐标系，Y 轴向下）：
  - 标签从画面上方移动到下方 → 入库 (in)
  - 标签从画面下方移动到上方 → 出库 (out)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median


@dataclass
class TrackState:
    tag_id: int
    y_samples: list[float] = field(default_factory=list)
    x_samples: list[float] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def center_x(self) -> float:
        return self.x_samples[-1] if self.x_samples else 0.0

    @property
    def center_y(self) -> float:
        return self.y_samples[-1] if self.y_samples else 0.0


@dataclass
class PassageEvent:
    tag_id: int
    op: str  # "in" | "out"
    displacement: float
    duration: float
    timestamp: float = field(default_factory=time.time)


class TagTracker:
    """基于轨迹采样的进出库判定器。"""

    def __init__(
        self,
        frame_height: int = 720,
        min_track_frames: int = 2,
        min_displacement_ratio: float = 0.02,
        disappear_timeout: float = 0.25,
        max_track_duration: float = 15.0,
        cooldown: float = 2.0,
        sample_window: int = 2,
        smooth_window: int = 2,
    ):
        self.frame_height = frame_height
        self.min_track_frames = min_track_frames
        self.min_displacement_ratio = min_displacement_ratio
        self.min_displacement_px = frame_height * min_displacement_ratio
        self.disappear_timeout = disappear_timeout
        self.max_track_duration = max_track_duration
        self.cooldown = cooldown
        self.sample_window = sample_window
        self.smooth_window = smooth_window

        self._active: dict[int, TrackState] = {}
        self._cooldown_until: dict[int, float] = {}
        self.recent_events: deque[PassageEvent] = deque(maxlen=50)
        self.rejected_short = 0
        self.rejected_static = 0

    def set_frame_height(self, height: int) -> None:
        self.frame_height = height
        self.min_displacement_px = height * self.min_displacement_ratio

    def update(self, detections: list[tuple[int, float, float]]) -> list[PassageEvent]:
        """
        输入当前帧检测到的标签列表 [(tag_id, center_x, center_y), ...]。
        返回本帧新完成的进出库事件。
        """
        now = time.time()
        seen_ids: set[int] = set()
        new_events: list[PassageEvent] = []

        for tag_id, cx, cy in detections:
            seen_ids.add(tag_id)

            if tag_id in self._cooldown_until and now < self._cooldown_until[tag_id]:
                continue

            if tag_id not in self._active:
                self._active[tag_id] = TrackState(tag_id=tag_id)

            track = self._active[tag_id]
            track.y_samples.append(self._smooth_value(track.y_samples, cy))
            track.x_samples.append(self._smooth_value(track.x_samples, cx))
            track.last_seen = now

        for tag_id in list(self._active.keys()):
            if tag_id in seen_ids:
                track = self._active[tag_id]
                if now - track.first_seen > self.max_track_duration:
                    event = self._finalize(tag_id)
                    if event:
                        new_events.append(event)
                continue

            track = self._active[tag_id]
            if now - track.last_seen >= self.disappear_timeout:
                event = self._finalize(tag_id)
                if event:
                    new_events.append(event)

        for event in new_events:
            self.recent_events.appendleft(event)

        return new_events

    def active_tracks(self) -> dict[int, TrackState]:
        return dict(self._active)

    def _smooth_value(self, samples: list[float], value: float) -> float:
        if not samples or self.smooth_window <= 1:
            return value
        window = samples[-(self.smooth_window - 1):] + [value]
        return float(median(window))

    def _finalize(self, tag_id: int) -> PassageEvent | None:
        track = self._active.pop(tag_id, None)
        if track is None:
            return None

        if len(track.y_samples) < self.min_track_frames:
            self.rejected_short += 1
            return None

        window = min(self.sample_window, len(track.y_samples) // 2)
        if window < 1:
            window = 1

        start_y = sum(track.y_samples[:window]) / window
        end_y = sum(track.y_samples[-window:]) / window
        displacement = end_y - start_y

        if abs(displacement) < self.min_displacement_px:
            self.rejected_static += 1
            return None

        op = "in" if displacement > 0 else "out"
        duration = track.last_seen - track.first_seen

        self._cooldown_until[tag_id] = time.time() + self.cooldown

        return PassageEvent(
            tag_id=tag_id,
            op=op,
            displacement=displacement,
            duration=duration,
        )
