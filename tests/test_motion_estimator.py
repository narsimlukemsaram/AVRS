from __future__ import annotations

from perception.motion_estimator import MotionEstimator
from perception.types import Detection


def create_estimator() -> MotionEstimator:
    return MotionEstimator(
        {
            "enabled": True,
            "history_length": 5,
            "minimum_history": 3,
            "moving_threshold_px_per_frame": 3.0,
            "stationary_threshold_px_per_frame": 1.5,
        }
    )


def create_detection(
    centre: tuple[int, int],
    track_id: str = "CAR01",
) -> Detection:
    x_centre, y_centre = centre

    return Detection(
        class_id=2,
        label="car",
        confidence=0.90,
        bbox=(
            x_centre - 20,
            y_centre - 10,
            x_centre + 20,
            y_centre + 10,
        ),
        centre=centre,
        track_id=track_id,
    )


def test_insufficient_history_returns_unknown() -> None:
    estimator = create_estimator()

    state = estimator.estimate(
        create_detection((100, 100)),
        frame_index=1,
    )

    assert state == "unknown"


def test_stationary_vehicle() -> None:
    estimator = create_estimator()

    estimator.estimate(
        create_detection((100, 100)),
        frame_index=1,
    )
    estimator.estimate(
        create_detection((101, 100)),
        frame_index=2,
    )

    state = estimator.estimate(
        create_detection((101, 101)),
        frame_index=3,
    )

    assert state == "stationary"


def test_moving_vehicle() -> None:
    estimator = create_estimator()

    estimator.estimate(
        create_detection((100, 100)),
        frame_index=1,
    )
    estimator.estimate(
        create_detection((110, 100)),
        frame_index=2,
    )

    state = estimator.estimate(
        create_detection((120, 100)),
        frame_index=3,
    )

    assert state == "moving"


def test_histories_are_track_specific() -> None:
    estimator = create_estimator()

    estimator.estimate(
        create_detection(
            (100, 100),
            track_id="CAR01",
        ),
        frame_index=1,
    )
    estimator.estimate(
        create_detection(
            (110, 100),
            track_id="CAR01",
        ),
        frame_index=2,
    )

    moving_state = estimator.estimate(
        create_detection(
            (120, 100),
            track_id="CAR01",
        ),
        frame_index=3,
    )

    estimator.estimate(
        create_detection(
            (200, 200),
            track_id="CAR02",
        ),
        frame_index=1,
    )
    estimator.estimate(
        create_detection(
            (200, 200),
            track_id="CAR02",
        ),
        frame_index=2,
    )

    stationary_state = estimator.estimate(
        create_detection(
            (201, 200),
            track_id="CAR02",
        ),
        frame_index=3,
    )

    assert moving_state == "moving"
    assert stationary_state == "stationary"


def test_disabled_estimator() -> None:
    estimator = MotionEstimator(
        {
            "enabled": False,
            "history_length": 5,
            "minimum_history": 3,
            "moving_threshold_px_per_frame": 3.0,
            "stationary_threshold_px_per_frame": 1.5,
        }
    )

    state = estimator.estimate(
        create_detection((100, 100)),
        frame_index=1,
    )

    assert state == "disabled"
