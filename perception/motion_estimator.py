from __future__ import annotations

import math
from collections import deque
from typing import Any

from perception.types import Detection


class MotionEstimator:
    """
    Estimate image-plane movement from tracked bounding-box centres.

    The result describes movement in the image, not necessarily physical
    vehicle movement. Drone or camera motion can make stationary vehicles
    appear to move.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.history_length = int(config.get("history_length", 5))
        self.minimum_history = int(config.get("minimum_history", 3))
        self.moving_threshold = float(
            config.get(
                "moving_threshold_px_per_frame",
                3.0,
            )
        )
        self.stationary_threshold = float(
            config.get(
                "stationary_threshold_px_per_frame",
                1.5,
            )
        )

        if self.history_length < 2:
            raise ValueError("history_length must be at least 2.")

        if not 2 <= self.minimum_history <= self.history_length:
            raise ValueError("minimum_history must be between 2 and history_length.")

        if self.stationary_threshold < 0:
            raise ValueError("stationary_threshold must not be negative.")

        if self.moving_threshold <= self.stationary_threshold:
            raise ValueError(
                "moving_threshold must be greater than stationary_threshold."
            )

        self._history: dict[
            str,
            deque[tuple[int, tuple[int, int]]],
        ] = {}

        self._states: dict[str, str] = {}

    def estimate(
        self,
        detection: Detection,
        frame_index: int,
    ) -> str:
        """
        Estimate movement for one tracked vehicle.

        Args:
            detection: Current tracked detection.
            frame_index: Sequential processed-frame index.

        Returns:
            moving, stationary, unknown or disabled.
        """
        if not self.enabled:
            return "disabled"

        track_id = detection.track_id

        if track_id is None:
            return "unknown"

        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.history_length)

        history = self._history[track_id]

        if history and history[-1][0] == frame_index:
            history[-1] = (
                frame_index,
                detection.centre,
            )
        else:
            history.append(
                (
                    frame_index,
                    detection.centre,
                )
            )

        if len(history) < self.minimum_history:
            return self._states.get(
                track_id,
                "unknown",
            )

        first_frame, first_centre = history[0]
        last_frame, last_centre = history[-1]

        frame_difference = last_frame - first_frame

        if frame_difference <= 0:
            return self._states.get(
                track_id,
                "unknown",
            )

        displacement_x = last_centre[0] - first_centre[0]
        displacement_y = last_centre[1] - first_centre[1]

        displacement_pixels = math.hypot(
            displacement_x,
            displacement_y,
        )

        speed_px_per_frame = displacement_pixels / frame_difference

        previous_state = self._states.get(
            track_id,
            "unknown",
        )

        if previous_state == "moving":
            if speed_px_per_frame <= self.stationary_threshold:
                state = "stationary"
            else:
                state = "moving"
        elif speed_px_per_frame >= self.moving_threshold:
            state = "moving"
        else:
            state = "stationary"

        self._states[track_id] = state

        return state

    def reset(self) -> None:
        """Clear all movement histories and states."""
        self._history.clear()
        self._states.clear()

    def remove_track(self, track_id: str) -> None:
        """Remove stored information for one expired track."""
        self._history.pop(track_id, None)
        self._states.pop(track_id, None)
