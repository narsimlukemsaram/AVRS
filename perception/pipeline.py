from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from perception.types import Detection

BOX_COLOURS: dict[str, tuple[int, int, int]] = {
    "car": (0, 200, 255),
    "van": (200, 0, 200),
    "motorcycle": (255, 100, 0),
    "bus": (0, 255, 100),
    "truck": (60, 60, 255),
    "lorry": (60, 60, 255),
}


class VideoSourceProtocol(Protocol):
    """Interface required from a video-source implementation."""

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame."""

    def change_source(self, source: str | int) -> None:
        """Change the active video source."""

    def release(self) -> None:
        """Release the active video source."""


class DetectorProtocol(Protocol):
    """Interface required from a vehicle detector."""

    def predict(
        self,
        frame: np.ndarray,
    ) -> list[Detection]:
        """Detect vehicles in one frame."""


class TrackerProtocol(Protocol):
    """Interface required from a vehicle tracker."""

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
    ) -> list[Detection]:
        """Update persistent vehicle tracks."""

    def reset(self) -> None:
        """Clear all active tracks."""


class ColourEstimatorProtocol(Protocol):
    """Interface required from a colour estimator."""

    def estimate(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> str:
        """Estimate one tracked vehicle's colour."""

    def reset(self) -> None:
        """Clear all colour histories."""


class MotionEstimatorProtocol(Protocol):
    """Interface required from a motion estimator."""

    def estimate(
        self,
        detection: Detection,
        frame_index: int,
    ) -> str:
        """Estimate one tracked vehicle's image-plane movement."""

    def reset(self) -> None:
        """Clear all motion histories."""


@dataclass
class PipelineResult:
    """
    Results produced for one successfully processed frame.

    raw_frame:
        Resized frame without overlays. This remains suitable for later
        perception or control processing.

    annotated_frame:
        Frame containing bounding boxes, labels and count information.
    """

    raw_frame: np.ndarray
    annotated_frame: np.ndarray
    detections: list[Detection]
    counts: dict[str, int]
    frame_index: int
    fps: float
    processing_time_ms: float


class PerceptionPipeline:
    """
    Coordinate video acquisition and Stage 1 vehicle perception.

    Processing order:

        frame acquisition
        -> resizing
        -> vehicle detection
        -> persistent tracking
        -> colour estimation
        -> image-plane motion estimation
        -> annotation and counting
    """

    def __init__(
        self,
        source: VideoSourceProtocol,
        detector: DetectorProtocol,
        tracker: TrackerProtocol,
        colour_estimator: ColourEstimatorProtocol,
        motion_estimator: MotionEstimatorProtocol,
        output_width: int = 720,
        output_height: int = 405,
        fps_smoothing_factor: float = 0.10,
        draw_summary: bool = True,
    ) -> None:
        if output_width < 1 or output_height < 1:
            raise ValueError("Output width and height must be positive integers.")

        if not 0.0 < fps_smoothing_factor <= 1.0:
            raise ValueError(
                "fps_smoothing_factor must be greater than 0.0 "
                "and less than or equal to 1.0."
            )

        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.colour_estimator = colour_estimator
        self.motion_estimator = motion_estimator

        self.output_width = output_width
        self.output_height = output_height
        self.fps_smoothing_factor = fps_smoothing_factor
        self.draw_summary_enabled = draw_summary

        self.frame_index = 0
        self.smoothed_fps = 0.0
        self._previous_frame_time: float | None = None

    def process_next_frame(self) -> PipelineResult | None:
        """
        Read and process one frame.

        Returns:
            PipelineResult when a frame is available.
            None when the source is temporarily unavailable.
        """
        success, input_frame = self.source.read()

        if not success or input_frame is None:
            return None

        processing_start = time.perf_counter()

        raw_frame = cv2.resize(
            input_frame,
            (
                self.output_width,
                self.output_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        self.frame_index += 1

        raw_detections = self.detector.predict(raw_frame)

        tracked_detections = self.tracker.update(
            raw_detections,
            frame_index=self.frame_index,
        )

        # Attribute estimation must use the raw frame before overlays.
        for detection in tracked_detections:
            detection.colour = self.colour_estimator.estimate(
                raw_frame,
                detection,
            )

            detection.movement = self.motion_estimator.estimate(
                detection,
                frame_index=self.frame_index,
            )

        counts = dict(Counter(detection.label for detection in tracked_detections))

        annotated_frame = raw_frame.copy()

        for detection in tracked_detections:
            self._draw_detection(
                annotated_frame,
                detection,
            )

        processing_end = time.perf_counter()
        processing_time = processing_end - processing_start

        fps = self._update_fps(current_time=processing_end)

        if self.draw_summary_enabled:
            self._draw_summary(
                annotated_frame,
                tracked_detections,
                counts,
                fps,
            )

        return PipelineResult(
            raw_frame=raw_frame,
            annotated_frame=annotated_frame,
            detections=tracked_detections,
            counts=counts,
            frame_index=self.frame_index,
            fps=fps,
            processing_time_ms=processing_time * 1000.0,
        )

    def change_source(
        self,
        source: str | int,
    ) -> None:
        """
        Change the video source and reset temporal perception state.

        Track IDs, colour histories and motion histories are reset because
        they are not valid across unrelated videos or cameras.
        """
        self.source.change_source(source)
        self.reset()

    def reset(self) -> None:
        """Reset frame counters and all temporal perception modules."""
        self.frame_index = 0
        self.smoothed_fps = 0.0
        self._previous_frame_time = None

        self.tracker.reset()
        self.colour_estimator.reset()
        self.motion_estimator.reset()

    def close(self) -> None:
        """Release the video source."""
        self.source.release()

    def _update_fps(
        self,
        current_time: float,
    ) -> float:
        """Update the exponentially smoothed processing rate."""
        if self._previous_frame_time is None:
            self._previous_frame_time = current_time
            return 0.0

        elapsed = max(
            current_time - self._previous_frame_time,
            1e-6,
        )

        self._previous_frame_time = current_time
        instantaneous_fps = 1.0 / elapsed

        if self.smoothed_fps == 0.0:
            self.smoothed_fps = instantaneous_fps
        else:
            alpha = self.fps_smoothing_factor

            self.smoothed_fps = (
                1.0 - alpha
            ) * self.smoothed_fps + alpha * instantaneous_fps

        return self.smoothed_fps

    @staticmethod
    def _draw_detection(
        frame: np.ndarray,
        detection: Detection,
    ) -> None:
        """Draw one tracked vehicle and its attributes."""
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        box_colour = BOX_COLOURS.get(
            detection.label,
            (255, 255, 255),
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_colour,
            2,
        )

        track_id = detection.track_id or "UNTRACKED"

        label = (
            f"{track_id} "
            f"{detection.label} "
            f"{detection.confidence:.2f} "
            f"{detection.colour} "
            f"{detection.movement}"
        )

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            1,
        )

        text_width, text_height = text_size

        label_bottom = max(
            text_height + baseline + 4,
            y1,
        )

        label_top = max(
            0,
            label_bottom - text_height - baseline - 6,
        )

        label_right = min(
            frame_width - 1,
            x1 + text_width + 8,
        )

        cv2.rectangle(
            frame,
            (x1, label_top),
            (label_right, label_bottom),
            box_colour,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (
                x1 + 4,
                label_bottom - baseline - 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.drawMarker(
            frame,
            detection.centre,
            box_colour,
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=1,
        )

    @staticmethod
    def _draw_summary(
        frame: np.ndarray,
        detections: list[Detection],
        counts: dict[str, int],
        fps: float,
    ) -> None:
        """Draw active vehicle counts and processing speed."""
        truck_count = counts.get("truck", 0) + counts.get("lorry", 0)

        summary_lines = [
            f"Total vehicles: {len(detections)}",
            f"Cars: {counts.get('car', 0)}",
            f"Vans: {counts.get('van', 0)}",
            f"Motorcycles: {counts.get('motorcycle', 0)}",
            f"Lorries/Trucks: {truck_count}",
            f"Buses: {counts.get('bus', 0)}",
            f"FPS: {fps:.1f}",
        ]

        panel_width = 235
        panel_height = 28 * len(summary_lines) + 12

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (8, 8),
            (
                8 + panel_width,
                8 + panel_height,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.65,
            frame,
            0.35,
            0,
            frame,
        )

        for index, text in enumerate(summary_lines):
            y_position = 32 + index * 28

            cv2.putText(
                frame,
                text,
                (18, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
