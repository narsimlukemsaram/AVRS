from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Detection:
    """
    Represents one detected or tracked vehicle.

    Image coordinates follow the OpenCV convention:
        x increases from left to right.
        y increases from top to bottom.
    """

    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    centre: tuple[int, int]

    track_id: str | None = None
    colour: str = "unknown"
    movement: str = "unknown"

    last_seen_frame: int = 0
    hits: int = 1
    lost_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the detection into a JSON-compatible dictionary."""
        result = asdict(self)
        result["bbox"] = list(self.bbox)
        result["centre"] = list(self.centre)
        return result
