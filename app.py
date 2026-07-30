from __future__ import annotations

import time

import cv2

from camera.video_source import VideoSource
from config_loader import load_config
from perception.colour_estimator import ColourEstimator
from perception.detector import VehicleDetector
from perception.motion_estimator import MotionEstimator
from perception.pipeline import PerceptionPipeline
from perception.tracker import VehicleTracker


def main() -> None:
    """Run the modular AI vehicle-recognition application."""
    config = load_config()

    app_config = config["app"]
    camera_config = config["camera"]
    detector_config = config["detector"]
    tracker_config = config["tracker"]
    colour_config = config["attributes"]["colour"]
    motion_config = config["attributes"]["motion"]

    source = VideoSource(
        source=camera_config["source"],
        loop_local_file=bool(
            camera_config.get(
                "loop_local_file",
                True,
            )
        ),
        reconnect_delay_s=float(
            camera_config.get(
                "reconnect_delay_s",
                2.0,
            )
        ),
    )

    detector = VehicleDetector(detector_config)

    tracker = VehicleTracker(
        iou_threshold=float(
            tracker_config.get(
                "iou_threshold",
                0.20,
            )
        ),
        maximum_lost_frames=int(
            tracker_config.get(
                "maximum_lost_frames",
                15,
            )
        ),
        minimum_hits=int(
            tracker_config.get(
                "minimum_hits",
                2,
            )
        ),
    )

    colour_estimator = ColourEstimator(colour_config)

    motion_estimator = MotionEstimator(motion_config)

    pipeline = PerceptionPipeline(
        source=source,
        detector=detector,
        tracker=tracker,
        colour_estimator=colour_estimator,
        motion_estimator=motion_estimator,
        output_width=int(
            camera_config.get(
                "output_width",
                720,
            )
        ),
        output_height=int(
            camera_config.get(
                "output_height",
                405,
            )
        ),
        fps_smoothing_factor=0.10,
        draw_summary=True,
    )

    window_name = str(
        app_config.get(
            "window_name",
            "AI Vehicle Recognition",
        )
    )

    display_enabled = bool(
        app_config.get(
            "display_enabled",
            True,
        )
    )

    print(f"[Application] Source: {camera_config['source']}")
    print("[Application] Press Q or Esc to close the application.")

    try:
        while True:
            result = pipeline.process_next_frame()

            if result is None:
                print(
                    "[Camera] Waiting for video source...",
                    end="\r",
                )

                if display_enabled:
                    key = cv2.waitKey(20) & 0xFF

                    if key in (
                        27,
                        ord("q"),
                    ):
                        break

                time.sleep(0.02)
                continue

            if display_enabled:
                cv2.imshow(
                    window_name,
                    result.annotated_frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    27,
                    ord("q"),
                ):
                    break

    except KeyboardInterrupt:
        print("\n[Application] Interrupted by user.")

    finally:
        pipeline.close()
        cv2.destroyAllWindows()

        print("\n[Application] Closed safely.")


if __name__ == "__main__":
    main()
