import numpy as np

from perception.pipeline import PerceptionPipeline
from perception.types import Detection


class FakeVideoSource:
    def __init__(
        self,
        frame_available: bool = True,
    ) -> None:
        self.frame_available = frame_available
        self.source: str | int = "sample.mp4"
        self.released = False

    def read(
        self,
    ) -> tuple[bool, np.ndarray | None]:
        if not self.frame_available:
            return False, None

        frame = np.zeros(
            (200, 300, 3),
            dtype=np.uint8,
        )

        return True, frame

    def change_source(
        self,
        source: str | int,
    ) -> None:
        self.source = source

    def release(self) -> None:
        self.released = True


class FakeDetector:
    def predict(
        self,
        frame: np.ndarray,
    ) -> list[Detection]:
        del frame

        return [
            Detection(
                class_id=2,
                label="car",
                confidence=0.91,
                bbox=(50, 40, 150, 100),
                centre=(100, 70),
            )
        ]


class FakeTracker:
    def __init__(self) -> None:
        self.reset_called = False

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
    ) -> list[Detection]:
        del frame_index

        for detection in detections:
            detection.track_id = "CAR01"

        return detections

    def reset(self) -> None:
        self.reset_called = True


class FakeColourEstimator:
    def __init__(self) -> None:
        self.reset_called = False

    def estimate(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> str:
        del frame
        del detection

        return "red"

    def reset(self) -> None:
        self.reset_called = True


class FakeMotionEstimator:
    def __init__(self) -> None:
        self.reset_called = False

    def estimate(
        self,
        detection: Detection,
        frame_index: int,
    ) -> str:
        del detection
        del frame_index

        return "moving"

    def reset(self) -> None:
        self.reset_called = True


def create_pipeline(
    frame_available: bool = True,
) -> tuple[
    PerceptionPipeline,
    FakeVideoSource,
    FakeTracker,
    FakeColourEstimator,
    FakeMotionEstimator,
]:
    source = FakeVideoSource(frame_available=frame_available)
    tracker = FakeTracker()
    colour_estimator = FakeColourEstimator()
    motion_estimator = FakeMotionEstimator()

    pipeline = PerceptionPipeline(
        source=source,
        detector=FakeDetector(),
        tracker=tracker,
        colour_estimator=colour_estimator,
        motion_estimator=motion_estimator,
        output_width=720,
        output_height=405,
    )

    return (
        pipeline,
        source,
        tracker,
        colour_estimator,
        motion_estimator,
    )


def test_pipeline_processes_and_enriches_detection() -> None:
    pipeline, _, _, _, _ = create_pipeline()

    result = pipeline.process_next_frame()

    assert result is not None
    assert result.frame_index == 1
    assert result.raw_frame.shape == (405, 720, 3)
    assert result.annotated_frame.shape == (405, 720, 3)

    assert len(result.detections) == 1
    assert result.counts == {"car": 1}

    detection = result.detections[0]

    assert detection.track_id == "CAR01"
    assert detection.colour == "red"
    assert detection.movement == "moving"


def test_pipeline_keeps_raw_frame_unannotated() -> None:
    pipeline, _, _, _, _ = create_pipeline()

    result = pipeline.process_next_frame()

    assert result is not None

    assert np.array_equal(
        result.raw_frame,
        np.zeros(
            (405, 720, 3),
            dtype=np.uint8,
        ),
    )

    assert not np.array_equal(
        result.raw_frame,
        result.annotated_frame,
    )


def test_pipeline_returns_none_when_source_is_unavailable() -> None:
    pipeline, _, _, _, _ = create_pipeline(frame_available=False)

    result = pipeline.process_next_frame()

    assert result is None
    assert pipeline.frame_index == 0


def test_pipeline_change_source_resets_temporal_modules() -> None:
    (
        pipeline,
        source,
        tracker,
        colour_estimator,
        motion_estimator,
    ) = create_pipeline()

    pipeline.process_next_frame()
    pipeline.change_source("rtsp://camera/stream")

    assert source.source == "rtsp://camera/stream"
    assert pipeline.frame_index == 0
    assert pipeline.smoothed_fps == 0.0

    assert tracker.reset_called
    assert colour_estimator.reset_called
    assert motion_estimator.reset_called


def test_pipeline_close_releases_video_source() -> None:
    pipeline, source, _, _, _ = create_pipeline()

    pipeline.close()

    assert source.released
