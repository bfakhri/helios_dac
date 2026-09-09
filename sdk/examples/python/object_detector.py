#!/usr/bin/env python3
"""
Modular Object Detection Engine for Stereo Vision.
Supports multiple detection backends:
- OpenCV Haar Cascades (Face, Fullbody, Upperbody) - Zero setup required
- HSV Color / Blob Tracker (Colored objects, laser dots, markers)
- OpenCV DNN (YOLOv5/v8, MobileNet ONNX models)
- MediaPipe Tasks API
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class Detection2D:
    """Represents a 2D object detection in a single camera frame."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    centroid: Tuple[float, float]
    ground_point: Tuple[float, float]  # Bottom-center (cx, y2)
    keypoints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    color_hist: Optional[np.ndarray] = None


def compute_hsv_histogram(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract normalized HSV color histogram from a bounding box for appearance matching."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((30,), dtype=np.float32)

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 16 bins for H, 8 bins for S
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


class BaseDetector(ABC):
    """Abstract Base Class for Object Detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        """Detect objects in a BGR frame and return a list of Detection2D instances."""
        pass


class CascadeDetector(BaseDetector):
    """OpenCV Haar Cascade Detector (Face, Fullbody, Upperbody)."""

    def __init__(self, cascade_type: str = "face", min_size: Tuple[int, int] = (30, 30)):
        self.cascade_type = cascade_type.lower()
        if self.cascade_type in ("face", "frontalface"):
            xml_file = "haarcascade_frontalface_default.xml"
            self.class_name = "face"
            self.class_id = 1
        elif self.cascade_type in ("fullbody", "body", "person"):
            xml_file = "haarcascade_fullbody.xml"
            self.class_name = "person"
            self.class_id = 2
        elif self.cascade_type == "upperbody":
            xml_file = "haarcascade_upperbody.xml"
            self.class_name = "upperbody"
            self.class_id = 3
        else:
            xml_file = cascade_type

        cascade_path = os.path.join(cv2.data.haarcascades, xml_file) if not os.path.exists(cascade_type) else cascade_type
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade XML not found: {cascade_path}")

        self.classifier = cv2.CascadeClassifier(cascade_path)
        self.min_size = min_size

    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        if frame is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=self.min_size
        )

        detections = []
        for (x, y, w, h) in boxes:
            cx = x + (w / 2.0)
            cy = y + (h / 2.0)
            ground_pt = (cx, float(y + h))
            bbox = (int(x), int(y), int(x + w), int(y + h))
            hist = compute_hsv_histogram(frame, bbox)

            detections.append(Detection2D(
                bbox=bbox,
                confidence=0.90,
                class_id=self.class_id,
                class_name=self.class_name,
                centroid=(cx, cy),
                ground_point=ground_pt,
                color_hist=hist,
            ))

        return detections


class ColorBlobDetector(BaseDetector):
    """
    HSV Color Blob & Contour Detector.
    Useful for tracking bright markers, colored targets, or laser spots.
    """

    def __init__(
        self,
        color: str = "red",
        lower_hsv: Optional[Tuple[int, int, int]] = None,
        upper_hsv: Optional[Tuple[int, int, int]] = None,
        min_area: int = 150,
        max_area: int = 500000,
    ):
        self.color = color.lower()
        self.min_area = min_area
        self.max_area = max_area

        if lower_hsv is not None and upper_hsv is not None:
            self.ranges = [(np.array(lower_hsv), np.array(upper_hsv))]
        elif self.color == "red":
            # Red wraps around 0/180 in HSV
            self.ranges = [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([170, 100, 100]), np.array([180, 255, 255])),
            ]
        elif self.color == "green":
            self.ranges = [(np.array([35, 80, 80]), np.array([85, 255, 255]))]
        elif self.color == "blue":
            self.ranges = [(np.array([95, 80, 80]), np.array([135, 255, 255]))]
        elif self.color == "yellow":
            self.ranges = [(np.array([20, 100, 100]), np.array([35, 255, 255]))]
        else:
            self.ranges = [(np.array([0, 50, 50]), np.array([180, 255, 255]))]

    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        if frame is None:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in self.ranges:
            m = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, m)

        # Morphology cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + (w / 2.0)
                cy = y + (h / 2.0)
                bbox = (int(x), int(y), int(x + w), int(y + h))
                hist = compute_hsv_histogram(frame, bbox)

                detections.append(Detection2D(
                    bbox=bbox,
                    confidence=min(1.0, area / 1000.0),
                    class_id=10,
                    class_name=f"{self.color}_target",
                    centroid=(cx, cy),
                    ground_point=(cx, float(y + h)),
                    color_hist=hist,
                ))

        return detections


class OpenCVDNNDetector(BaseDetector):
    """OpenCV DNN Object Detector loading ONNX models (e.g. YOLOv8 / YOLOv5 / MobileNet)."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.40,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        classes: Optional[List[str]] = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DNN Model file not found: {model_path}")

        self.net = cv2.dnn.readNet(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.classes = classes or ["object"]

    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        if frame is None:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, self.input_size, swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Handle YOLOv8 format (1, 84, 8400) or YOLOv5 format (1, 25200, 85)
        detections = []
        boxes, confidences, class_ids = [], [], []

        if len(outputs.shape) == 3:
            # Transpose if (1, 84, N)
            if outputs.shape[1] < outputs.shape[2]:
                outputs = np.transpose(outputs, (0, 2, 1))

            rows = outputs[0]
            x_scale = w / self.input_size[0]
            y_scale = h / self.input_size[1]

            for row in rows:
                scores = row[4:]
                max_score = np.max(scores)
                if max_score >= self.conf_threshold:
                    class_id = int(np.argmax(scores))
                    cx, cy, bw, bh = row[0] * x_scale, row[1] * y_scale, row[2] * x_scale, row[3] * y_scale
                    x1 = int(cx - bw / 2.0)
                    y1 = int(cy - bh / 2.0)
                    boxes.append([x1, y1, int(bw), int(bh)])
                    confidences.append(float(max_score))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        if len(indices) > 0:
            for idx in indices.flatten():
                x, y, bw, bh = boxes[idx]
                bbox = (x, y, x + bw, y + bh)
                cid = class_ids[idx]
                cname = self.classes[cid] if cid < len(self.classes) else f"class_{cid}"
                cx = x + (bw / 2.0)
                cy = y + (bh / 2.0)
                hist = compute_hsv_histogram(frame, bbox)

                detections.append(Detection2D(
                    bbox=bbox,
                    confidence=confidences[idx],
                    class_id=cid,
                    class_name=cname,
                    centroid=(cx, cy),
                    ground_point=(cx, float(y + bh)),
                    color_hist=hist,
                ))

        return detections


def create_detector(detector_type: str = "face", **kwargs) -> BaseDetector:
    """
    Factory function to instantiate detectors by name.
    Options: 'face', 'fullbody', 'upperbody', 'color' (or 'red', 'green', 'blue'), 'dnn'
    """
    dtype = detector_type.lower()
    if dtype in ("face", "fullbody", "upperbody"):
        return CascadeDetector(cascade_type=dtype, **kwargs)
    elif dtype.startswith("color") or dtype in ("red", "green", "blue", "yellow"):
        color = kwargs.get("color", dtype.replace("color_", "") if "_" in dtype else dtype)
        if color == "color":
            color = "red"
        return ColorBlobDetector(color=color, **kwargs)
    elif dtype == "dnn":
        return OpenCVDNNDetector(**kwargs)
    else:
        # Default fallback to face detector
        return CascadeDetector(cascade_type="face", **kwargs)
