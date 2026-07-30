from __future__ import annotations

from collections import Counter, deque
from typing import Any

import cv2
import numpy as np

from perception.types import Detection


class ColourEstimator:
    """
    Estimate a tracked vehicle's dominant body colour using HSV analysis.

    The estimator:
    - analyses the central portion of the vehicle bounding box;
    - avoids including excessive road and background pixels;
    - uses median HSV values for robustness;
    - smooths estimates separately for each persistent track ID.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.history_length = int(config.get("history_length", 12))
        self.minimum_box_side = int(config.get("minimum_box_side", 20))
        self.central_crop_ratio = float(config.get("central_crop_ratio", 0.20))
        self.resize_width = int(config.get("resize_width", 48))
        self.resize_height = int(config.get("resize_height", 48))

        if self.history_length < 1:
            raise ValueError("history_length must be at least 1.")

        if self.minimum_box_side < 1:
            raise ValueError("minimum_box_side must be at least 1.")

        if not 0.0 <= self.central_crop_ratio < 0.5:
            raise ValueError(
                "central_crop_ratio must be greater than or equal to 0.0 "
                "and less than 0.5."
            )

        if self.resize_width < 1 or self.resize_height < 1:
            raise ValueError("Resize dimensions must be positive integers.")

        self._history: dict[str, deque[str]] = {}

    def estimate(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> str:
        """
        Estimate and temporally smooth one vehicle's colour.

        Args:
            frame: Raw BGR frame before drawing boxes or labels.
            detection: A tracked vehicle detection.

        Returns:
            A coarse colour label.
        """
        if not self.enabled:
            return "disabled"

        if frame is None or frame.size == 0:
            return self._smooth_colour(
                detection.track_id,
                "unknown",
            )

        crop = self._extract_central_crop(
            frame,
            detection.bbox,
        )

        if crop is None:
            return self._smooth_colour(
                detection.track_id,
                "unknown",
            )

        colour = self._classify_crop(crop)

        return self._smooth_colour(
            detection.track_id,
            colour,
        )

    def reset(self) -> None:
        """Clear all stored per-track colour histories."""
        self._history.clear()

    def remove_track(self, track_id: str) -> None:
        """Remove stored history for an expired track."""
        self._history.pop(track_id, None)

    def _extract_central_crop(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        """
        Extract the central region of a bounding box.

        Removing the outer region reduces contamination from roads,
        shadows, neighbouring objects and inaccurate box boundaries.
        """
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width))
        y2 = max(0, min(y2, frame_height))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width < self.minimum_box_side or box_height < self.minimum_box_side:
            return None

        margin_x = round(box_width * self.central_crop_ratio)
        margin_y = round(box_height * self.central_crop_ratio)

        crop_x1 = x1 + margin_x
        crop_y1 = y1 + margin_y
        crop_x2 = x2 - margin_x
        crop_y2 = y2 - margin_y

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        crop = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        if crop.size == 0:
            return None

        return crop

    def _classify_crop(
        self,
        crop: np.ndarray,
    ) -> str:
        """Classify a cropped vehicle into a coarse colour category."""
        resized_crop = cv2.resize(
            crop,
            (
                self.resize_width,
                self.resize_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        hsv_image = cv2.cvtColor(
            resized_crop,
            cv2.COLOR_BGR2HSV,
        )

        hue_channel, saturation_channel, value_channel = cv2.split(hsv_image)

        # Prefer pixels that are not extreme shadows or highlights.
        valid_pixels = (value_channel > 25) & (value_channel < 245)

        minimum_valid_pixels = max(
            20,
            round(value_channel.size * 0.05),
        )

        if np.count_nonzero(valid_pixels) < minimum_valid_pixels:
            valid_pixels = np.ones_like(
                value_channel,
                dtype=bool,
            )

        median_hue = float(np.median(hue_channel[valid_pixels]))
        median_saturation = float(np.median(saturation_channel[valid_pixels]))
        median_value = float(np.median(value_channel[valid_pixels]))

        return self._hsv_to_colour(
            hue=median_hue,
            saturation=median_saturation,
            value=median_value,
        )

    @staticmethod
    def _hsv_to_colour(
        hue: float,
        saturation: float,
        value: float,
    ) -> str:
        """
        Convert OpenCV HSV values into a coarse colour label.

        OpenCV represents hue in the range 0-179.
        """
        # Low-saturation colours.
        if saturation < 38:
            if value < 58:
                return "black"

            if value < 118:
                return "grey"

            if value < 185:
                return "silver"

            return "white"

        # Chromatic colours.
        if hue < 8 or hue >= 170:
            if value < 90:
                return "dark red"

            return "red"

        if hue < 20:
            if value < 125:
                return "brown"

            return "orange"

        if hue < 35:
            return "yellow"

        if hue < 85:
            if value < 90:
                return "dark green"

            return "green"

        if hue < 132:
            if value < 90:
                return "dark blue"

            return "blue"

        if hue < 165:
            return "purple"

        return "red"

    def _smooth_colour(
        self,
        track_id: str | None,
        colour: str,
    ) -> str:
        """
        Smooth colour estimates using a per-track majority vote.

        Unknown values are not stored. When the current estimate is
        unavailable, the previously stable colour may still be returned.
        """
        if track_id is None:
            return colour

        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.history_length)

        history = self._history[track_id]

        if colour not in {
            "unknown",
            "disabled",
        }:
            history.append(colour)

        if not history:
            return colour

        return Counter(history).most_common(1)[0][0]
