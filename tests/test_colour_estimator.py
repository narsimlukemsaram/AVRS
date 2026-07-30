import cv2
import numpy as np

from perception.colour_estimator import ColourEstimator
from perception.types import Detection


def create_estimator(
    history_length: int = 5,
    minimum_box_side: int = 10,
) -> ColourEstimator:
    return ColourEstimator(
        {
            "enabled": True,
            "history_length": history_length,
            "minimum_box_side": minimum_box_side,
            "central_crop_ratio": 0.20,
            "resize_width": 48,
            "resize_height": 48,
        }
    )


def create_detection(
    track_id: str = "CAR01",
    bbox: tuple[int, int, int, int] = (
        10,
        10,
        90,
        90,
    ),
) -> Detection:
    return Detection(
        class_id=2,
        label="car",
        confidence=0.90,
        bbox=bbox,
        centre=(
            (bbox[0] + bbox[2]) // 2,
            (bbox[1] + bbox[3]) // 2,
        ),
        track_id=track_id,
    )


def create_colour_frame(
    bgr_colour: tuple[int, int, int],
) -> np.ndarray:
    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )
    frame[:, :] = bgr_colour

    return frame


def test_white_vehicle_colour() -> None:
    estimator = create_estimator()
    frame = create_colour_frame((240, 240, 240))

    colour = estimator.estimate(
        frame,
        create_detection(),
    )

    assert colour == "white"


def test_black_vehicle_colour() -> None:
    estimator = create_estimator()
    frame = create_colour_frame((20, 20, 20))

    colour = estimator.estimate(
        frame,
        create_detection(),
    )

    assert colour == "black"


def test_red_vehicle_colour() -> None:
    estimator = create_estimator()
    frame = create_colour_frame((0, 0, 220))

    colour = estimator.estimate(
        frame,
        create_detection(),
    )

    assert colour == "red"


def test_blue_vehicle_colour() -> None:
    estimator = create_estimator()
    frame = create_colour_frame((220, 0, 0))

    colour = estimator.estimate(
        frame,
        create_detection(),
    )

    assert colour == "blue"


def test_small_detection_returns_unknown() -> None:
    estimator = create_estimator(minimum_box_side=20)

    frame = create_colour_frame((0, 0, 220))

    detection = create_detection(bbox=(10, 10, 18, 18))

    colour = estimator.estimate(
        frame,
        detection,
    )

    assert colour == "unknown"


def test_colour_is_smoothed_per_track() -> None:
    estimator = create_estimator(history_length=5)

    detection = create_detection(track_id="CAR01")

    red_frame = create_colour_frame((0, 0, 220))
    blue_frame = create_colour_frame((220, 0, 0))

    estimator.estimate(red_frame, detection)
    estimator.estimate(red_frame, detection)
    estimator.estimate(red_frame, detection)

    colour = estimator.estimate(
        blue_frame,
        detection,
    )

    assert colour == "red"


def test_colour_histories_are_track_specific() -> None:
    estimator = create_estimator()

    red_detection = create_detection(track_id="CAR01")
    blue_detection = create_detection(track_id="CAR02")

    red_frame = create_colour_frame((0, 0, 220))
    blue_frame = create_colour_frame((220, 0, 0))

    red_colour = estimator.estimate(
        red_frame,
        red_detection,
    )
    blue_colour = estimator.estimate(
        blue_frame,
        blue_detection,
    )

    assert red_colour == "red"
    assert blue_colour == "blue"


def test_central_crop_reduces_background_effect() -> None:
    estimator = create_estimator()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    # Green road/background area.
    frame[:, :] = (0, 180, 0)

    # Red vehicle body in the central region.
    cv2.rectangle(
        frame,
        (26, 26),
        (74, 74),
        (0, 0, 220),
        -1,
    )

    detection = create_detection(bbox=(10, 10, 90, 90))

    colour = estimator.estimate(
        frame,
        detection,
    )

    assert colour == "red"
