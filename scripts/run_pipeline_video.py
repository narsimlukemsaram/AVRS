from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# Allow this script to import modules from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.video_source import VideoSource
from config_loader import load_config
from perception.colour_estimator import ColourEstimator
from perception.detector import VehicleDetector
from perception.motion_estimator import MotionEstimator
from perception.pipeline import PerceptionPipeline
from perception.tracker import VehicleTracker


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete vehicle-perception pipeline on a local "
            "video and save an annotated MP4."
        )
    )

    parser.add_argument(
        "--config",
        default="config-jetson.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--output",
        default="runs/jetson_pipeline/short_test_annotated.mp4",
        help="Path for the annotated output video.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process. Use 0 for the complete video.",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the annotated frames while processing.",
    )

    return parser.parse_args()


def probe_video_fps(video_path: str) -> float:
    """Return a suitable frame rate for the output video."""
    capture = cv2.VideoCapture(video_path)

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()

    if fps < 1.0 or fps > 120.0:
        return 15.0

    return fps


def main() -> int:
    """Run the complete perception pipeline."""
    args = parse_arguments()
    config = load_config(args.config)

    camera_config = config["camera"]
    detector_config = config["detector"]
    tracker_config = config["tracker"]
    colour_config = config["attributes"]["colour"]
    motion_config = config["attributes"]["motion"]

    source_path = str(camera_config["source"])
    input_path = Path(source_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input video was not found: {input_path.resolve()}"
        )

    model_path = Path(str(detector_config["model_path"]))

    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model was not found: {model_path.resolve()}"
        )

    source = VideoSource(
        source=source_path,
        loop_local_file=False,
        reconnect_delay_s=float(
            camera_config.get("reconnect_delay_s", 2.0)
        ),
    )

    detector = VehicleDetector(detector_config)

    tracker = VehicleTracker(
        iou_threshold=float(
            tracker_config.get("iou_threshold", 0.20)
        ),
        maximum_lost_frames=int(
            tracker_config.get("maximum_lost_frames", 15)
        ),
        minimum_hits=int(
            tracker_config.get("minimum_hits", 2)
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
            camera_config.get("output_width", 640)
        ),
        output_height=int(
            camera_config.get("output_height", 360)
        ),
        fps_smoothing_factor=0.10,
        draw_summary=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_fps = probe_video_fps(source_path)

    writer = None
    processed_frames = 0
    unavailable_reads = 0
    total_detections = 0
    processing_times = []

    print(f"[Pipeline] Input: {input_path}")
    print(f"[Pipeline] Model: {model_path}")
    print(f"[Pipeline] Output: {output_path}")
    print(f"[Pipeline] Output FPS: {output_fps:.2f}")

    try:
        while True:
            result = pipeline.process_next_frame()

            if result is None:
                if processed_frames > 0:
                    # End of the local video.
                    break

                unavailable_reads += 1

                if unavailable_reads >= 50:
                    raise RuntimeError(
                        "The input video could not be opened or decoded."
                    )

                time.sleep(0.05)
                continue

            unavailable_reads = 0

            if writer is None:
                frame_height, frame_width = (
                    result.annotated_frame.shape[:2]
                )

                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (frame_width, frame_height),
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "OpenCV could not create the output MP4 file."
                    )

            writer.write(result.annotated_frame)

            if args.display:
                cv2.imshow(
                    "AI Vehicle Recognition - Jetson Nano",
                    result.annotated_frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    print(
                        "\n[Pipeline] Live display closed by user."
                    )
                    break

            processed_frames += 1
            total_detections += len(result.detections)
            processing_times.append(result.processing_time_ms)

            if processed_frames % 25 == 0:
                print(
                    f"[Pipeline] Frames: {processed_frames}, "
                    f"FPS: {result.fps:.2f}, "
                    f"Processing: {result.processing_time_ms:.1f} ms, "
                    f"Counts: {result.counts}"
                )

            if (
                args.max_frames > 0
                and processed_frames >= args.max_frames
            ):
                break

    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user.")

    finally:
        pipeline.close()

        if writer is not None:
            writer.release()

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    if processed_frames == 0:
        print("[Pipeline] No frames were processed.")
        return 1

    average_processing_ms = (
        sum(processing_times) / len(processing_times)
    )

    print("[Pipeline] Completed successfully.")
    print(f"[Pipeline] Processed frames: {processed_frames}")
    print(f"[Pipeline] Total tracked detections: {total_detections}")
    print(
        "[Pipeline] Average processing time: "
        f"{average_processing_ms:.1f} ms/frame"
    )
    print(f"[Pipeline] Saved output: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
