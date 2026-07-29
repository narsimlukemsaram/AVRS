from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics import YOLO

from perception.types import Detection


class VehicleDetector:
    """
    YOLO vehicle detector for Windows development.

    The initial model uses COCO classes:
        2: car
        3: motorcycle
        5: bus
        7: truck

    A custom aerial vehicle model can later replace this backend.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.model_path = str(config.get("model_path", "yolov8n.pt"))

        self.image_size = int(config.get("image_size", 640))

        self.confidence_threshold = float(config.get("confidence", 0.35))

        self.iou_threshold = float(config.get("iou", 0.45))

        self.device = str(config.get("device", "auto"))

        self.vehicle_class_ids = {
            int(class_id)
            for class_id in config.get(
                "vehicle_class_ids",
                [2, 3, 5, 7],
            )
        }

        print(f"[Detector] Loading model: {self.model_path}")

        self.model = YOLO(self.model_path)

        print("[Detector] Model loaded successfully")

    def predict(
        self,
        frame: np.ndarray,
    ) -> list[Detection]:
        """
        Run vehicle detection on one BGR OpenCV frame.

        Args:
            frame: OpenCV BGR image.

        Returns:
            List of vehicle detections.
        """
        if frame is None or frame.size == 0:
            return []

        prediction_arguments: dict[str, Any] = {
            "source": frame,
            "imgsz": self.image_size,
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "classes": sorted(self.vehicle_class_ids),
            "verbose": False,
        }

        if self.device.lower() != "auto":
            prediction_arguments["device"] = self.device

        results = self.model.predict(**prediction_arguments)

        if not results:
            return []

        result = results[0]
        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())

            if class_id not in self.vehicle_class_ids:
                continue

            confidence = float(box.conf.item())

            coordinates = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [round(value) for value in coordinates]

            if x2 <= x1 or y2 <= y1:
                continue

            label = self._get_class_name(
                result.names,
                class_id,
            )

            detections.append(
                Detection(
                    class_id=class_id,
                    label=label,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    centre=(
                        (x1 + x2) // 2,
                        (y1 + y2) // 2,
                    ),
                )
            )

        return detections

    @staticmethod
    def _get_class_name(
        class_names: dict[int, str] | list[str],
        class_id: int,
    ) -> str:
        """Retrieve a class label from Ultralytics result metadata."""
        if isinstance(class_names, dict):
            return str(
                class_names.get(
                    class_id,
                    f"class_{class_id}",
                )
            )

        if 0 <= class_id < len(class_names):
            return str(class_names[class_id])

        return f"class_{class_id}"
