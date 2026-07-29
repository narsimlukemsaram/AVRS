from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import ClassVar

from perception.types import Detection


def calculate_iou(
    first_box: tuple[int, int, int, int],
    second_box: tuple[int, int, int, int],
) -> float:
    """
    Calculate intersection over union between two bounding boxes.

    Bounding-box format:
        x1, y1, x2, y2
    """
    ax1, ay1, ax2, ay2 = first_box
    bx1, by1, bx2, by2 = second_box

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1,
    )
    intersection_height = max(
        0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = intersection_width * intersection_height

    if intersection_area == 0:
        return 0.0

    first_area = max(0, ax2 - ax1) * max(
        0,
        ay2 - ay1,
    )
    second_area = max(0, bx2 - bx1) * max(
        0,
        by2 - by1,
    )

    union_area = first_area + second_area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


class VehicleTracker:
    """
    Lightweight IoU-based vehicle tracker.

    This is suitable as a modular baseline. It can later be replaced by
    ByteTrack or BoT-SORT without changing the detector interface.
    """

    ID_PREFIXES: ClassVar[dict[str, str]] = {
        "car": "CAR",
        "motorcycle": "MOTO",
        "bus": "BUS",
        "truck": "TRK",
        "lorry": "TRK",
        "van": "VAN",
    }

    def __init__(
        self,
        iou_threshold: float = 0.20,
        maximum_lost_frames: int = 15,
        minimum_hits: int = 2,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.maximum_lost_frames = maximum_lost_frames
        self.minimum_hits = minimum_hits

        self._tracks: dict[str, Detection] = {}
        self._class_counters: defaultdict[str, int] = defaultdict(int)

    def reset(self) -> None:
        """Remove all tracks and restart identifier counters."""
        self._tracks.clear()
        self._class_counters.clear()

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
    ) -> list[Detection]:
        """
        Associate current detections with existing tracks.

        Args:
            detections: Current-frame YOLO detections.
            frame_index: Sequential frame number.

        Returns:
            Visible and sufficiently mature tracks.
        """
        available_track_ids = set(self._tracks)
        updated_tracks: dict[str, Detection] = {}

        detections_by_confidence = sorted(
            detections,
            key=lambda detection: detection.confidence,
            reverse=True,
        )

        for detection in detections_by_confidence:
            best_track_id: str | None = None
            best_iou = self.iou_threshold

            for track_id in available_track_ids:
                previous_track = self._tracks[track_id]

                # Prevent matching vehicles from different classes.
                if previous_track.label != detection.label:
                    continue

                iou = calculate_iou(
                    previous_track.bbox,
                    detection.bbox,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is None:
                track_id = self._create_track_id(detection.label)

                updated_detection = replace(
                    detection,
                    track_id=track_id,
                    last_seen_frame=frame_index,
                    hits=1,
                    lost_frames=0,
                )

                updated_tracks[track_id] = updated_detection
                continue

            previous_track = self._tracks[best_track_id]

            updated_detection = replace(
                detection,
                track_id=best_track_id,
                colour=previous_track.colour,
                movement=previous_track.movement,
                last_seen_frame=frame_index,
                hits=previous_track.hits + 1,
                lost_frames=0,
            )

            updated_tracks[best_track_id] = updated_detection
            available_track_ids.remove(best_track_id)

        # Retain unmatched tracks temporarily for short occlusions.
        for track_id in available_track_ids:
            previous_track = self._tracks[track_id]

            lost_track = replace(
                previous_track,
                lost_frames=previous_track.lost_frames + 1,
            )

            if lost_track.lost_frames <= self.maximum_lost_frames:
                updated_tracks[track_id] = lost_track

        self._tracks = updated_tracks

        return [
            track
            for track in self._tracks.values()
            if (track.hits >= self.minimum_hits and track.lost_frames == 0)
        ]

    def _create_track_id(self, label: str) -> str:
        """Create a per-class identifier such as CAR01."""
        self._class_counters[label] += 1

        prefix = self.ID_PREFIXES.get(
            label,
            label[:3].upper(),
        )

        number = self._class_counters[label]

        return f"{prefix}{number:02d}"
