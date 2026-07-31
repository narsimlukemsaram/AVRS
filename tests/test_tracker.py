from __future__ import annotations

from perception.tracker import VehicleTracker, calculate_iou
from perception.types import Detection


def create_car_detection(x_position: int) -> Detection:
    """Create a synthetic car detection for tracker testing."""
    return Detection(
        class_id=2,
        label="car",
        confidence=0.90,
        bbox=(
            x_position,
            20,
            x_position + 100,
            80,
        ),
        centre=(
            x_position + 50,
            50,
        ),
    )


def test_iou_for_identical_boxes() -> None:
    box = (10, 20, 100, 80)

    assert calculate_iou(box, box) == 1.0


def test_iou_for_separate_boxes() -> None:
    first_box = (0, 0, 50, 50)
    second_box = (100, 100, 150, 150)

    assert calculate_iou(first_box, second_box) == 0.0


def test_tracker_preserves_identifier() -> None:
    tracker = VehicleTracker(
        iou_threshold=0.20,
        maximum_lost_frames=15,
        minimum_hits=1,
    )

    first_tracks = tracker.update(
        [create_car_detection(10)],
        frame_index=1,
    )

    second_tracks = tracker.update(
        [create_car_detection(15)],
        frame_index=2,
    )

    assert len(first_tracks) == 1
    assert len(second_tracks) == 1
    assert first_tracks[0].track_id == second_tracks[0].track_id


def test_tracker_creates_new_identifier() -> None:
    tracker = VehicleTracker(
        iou_threshold=0.20,
        maximum_lost_frames=15,
        minimum_hits=1,
    )

    first_tracks = tracker.update(
        [create_car_detection(10)],
        frame_index=1,
    )

    second_tracks = tracker.update(
        [
            create_car_detection(15),
            create_car_detection(300),
        ],
        frame_index=2,
    )

    track_ids = {track.track_id for track in second_tracks}

    assert first_tracks[0].track_id in track_ids
    assert len(track_ids) == 2


def test_tracker_requires_minimum_hits() -> None:
    tracker = VehicleTracker(
        iou_threshold=0.20,
        maximum_lost_frames=15,
        minimum_hits=2,
    )

    first_tracks = tracker.update(
        [create_car_detection(10)],
        frame_index=1,
    )

    second_tracks = tracker.update(
        [create_car_detection(15)],
        frame_index=2,
    )

    assert first_tracks == []
    assert len(second_tracks) == 1
    assert second_tracks[0].track_id == "CAR01"


def test_tracker_removes_expired_track() -> None:
    tracker = VehicleTracker(
        iou_threshold=0.20,
        maximum_lost_frames=2,
        minimum_hits=1,
    )

    initial_tracks = tracker.update(
        [create_car_detection(10)],
        frame_index=1,
    )

    tracker.update([], frame_index=2)
    tracker.update([], frame_index=3)
    tracker.update([], frame_index=4)

    new_tracks = tracker.update(
        [create_car_detection(10)],
        frame_index=5,
    )

    assert initial_tracks[0].track_id == "CAR01"
    assert new_tracks[0].track_id == "CAR02"
