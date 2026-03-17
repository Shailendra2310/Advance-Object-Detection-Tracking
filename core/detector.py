"""
Core detector – wraps YOLOv8 for both Traffic and Surveillance modes.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("detector")


# COCO class indices relevant to each mode
VEHICLE_CLASS_IDS  = {2, 3, 5, 7}   # car, motorcycle, bus, truck
PERSON_CLASS_ID    = 0


class Detection:
    """Simple container for a single detection result."""
    def __init__(self, bbox, class_id, class_name, confidence):
        self.bbox       = bbox          # [x1, y1, x2, y2]
        self.class_id   = class_id
        self.class_name = class_name
        self.confidence = confidence

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self):
        return self.width * self.height


class Detector:
    """YOLOv8-based object detector."""

    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        model_path = f"models/weights/{self.settings.model_size}.pt"

        logger.info(f"Loading YOLO model: {self.settings.model_size} on {self.settings.device}")
        # Ultralytics auto-downloads if not present
        self.model = YOLO(model_path if self._model_exists(model_path)
                          else self.settings.model_size)
        self.model.to(self.settings.device)
        self.class_names = self.model.names
        logger.info("Model loaded successfully.")

    def _model_exists(self, path):
        import os
        return os.path.exists(path)

    def detect(self, frame: np.ndarray,
               filter_classes: set = None) -> list[Detection]:
        """
        Run detection on a single frame.

        Args:
            frame:          BGR image (numpy array)
            filter_classes: set of COCO class IDs to keep (None = keep all)

        Returns:
            List of Detection objects
        """
        results = self.model(
            frame,
            conf=self.settings.confidence_threshold,
            iou=self.settings.iou_threshold,
            imgsz=self.settings.input_size,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if filter_classes and cls_id not in filter_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            name = self.class_names[cls_id]
            detections.append(Detection([x1, y1, x2, y2], cls_id, name, conf))

        return detections

    def detect_vehicles(self, frame: np.ndarray) -> list[Detection]:
        return self.detect(frame, filter_classes=VEHICLE_CLASS_IDS)

    def detect_persons(self, frame: np.ndarray) -> list[Detection]:
        return self.detect(frame, filter_classes={PERSON_CLASS_ID})
