from __future__ import annotations

import time
from collections import Counter

import cv2

from camera.video_source import VideoSource
from config_loader import load_config
from perception.detector import VehicleDetector
from perception.types import Detection

BOX_COLOURS: dict[str, tuple[int, int, int]] = {
    "car": (0, 200, 255),
    "motorcycle": (255, 100, 0),
    "bus": (0, 255, 100),
    "truck": (60, 60, 255),
}


def draw_detection(
    frame,
    detection: Detection,
) -> None:
    """Draw one vehicle bounding box and label."""
    x1, y1, x2, y2 = detection.bbox

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

    label = f"{detection.label} {detection.confidence:.2f}"

    text_size, baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )

    text_width, text_height = text_size

    label_top = max(0, y1 - text_height - baseline - 8)

    cv2.rectangle(
        frame,
        (x1, label_top),
        (x1 + text_width + 8, y1),
        box_colour,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 4, max(text_height + 2, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def draw_summary(
    frame,
    detections: list[Detection],
    fps: float,
) -> None:
    """Draw vehicle counts and processing speed."""
    counts = Counter(detection.label for detection in detections)

    summary_lines = [
        f"Total vehicles: {len(detections)}",
        f"Cars: {counts.get('car', 0)}",
        f"Motorcycles: {counts.get('motorcycle', 0)}",
        f"Buses: {counts.get('bus', 0)}",
        f"Trucks: {counts.get('truck', 0)}",
        f"FPS: {fps:.1f}",
    ]

    panel_width = 210
    panel_height = 28 * len(summary_lines) + 12

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (8, 8),
        (8 + panel_width, 8 + panel_height),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
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


def main() -> None:
    config = load_config()

    app_config = config["app"]
    camera_config = config["camera"]
    detector_config = config["detector"]

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

    output_width = int(
        camera_config.get(
            "output_width",
            720,
        )
    )

    output_height = int(
        camera_config.get(
            "output_height",
            405,
        )
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

    print(f"[Application] Source: {source.source}")
    print("[Application] Press Q or Esc to close the application")

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            success, frame = source.read()

            if not success or frame is None:
                print(
                    "[Camera] Waiting for video source...",
                    end="\r",
                )

                if display_enabled:
                    key = cv2.waitKey(20) & 0xFF

                    if key in (27, ord("q")):
                        break

                time.sleep(0.02)
                continue

            frame = cv2.resize(
                frame,
                (output_width, output_height),
                interpolation=cv2.INTER_LINEAR,
            )

            detections = detector.predict(frame)

            for detection in detections:
                draw_detection(
                    frame,
                    detection,
                )

            current_time = time.perf_counter()
            elapsed_time = max(
                current_time - previous_time,
                1e-6,
            )
            previous_time = current_time

            current_fps = 1.0 / elapsed_time

            if smoothed_fps == 0.0:
                smoothed_fps = current_fps
            else:
                smoothed_fps = 0.90 * smoothed_fps + 0.10 * current_fps

            draw_summary(
                frame,
                detections,
                smoothed_fps,
            )

            if display_enabled:
                cv2.imshow(
                    window_name,
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    break

    except KeyboardInterrupt:
        print("\n[Application] Interrupted by user")

    finally:
        source.release()
        cv2.destroyAllWindows()
        print("\n[Application] Closed safely")


if __name__ == "__main__":
    main()
