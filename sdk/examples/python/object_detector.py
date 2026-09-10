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


def letterbox_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize image preserving aspect ratio with padding (letterboxing).
    Returns:
        (letterboxed_image, scale_factor, (pad_x, pad_y))
    """
    ih, iw = image.shape[:2]
    tw, th = target_size
    scale = min(tw / iw, th / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_x = (tw - nw) / 2.0
    pad_y = (th - nh) / 2.0
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))

    letterboxed = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return letterboxed, scale, (pad_x, pad_y)


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

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=self.min_size
        )

        detections = []
        for (x, y, bw, bh) in boxes:
            x1 = max(0, min(w - 1, int(x)))
            y1 = max(0, min(h - 1, int(y)))
            x2 = max(x1 + 1, min(w, int(x + bw)))
            y2 = max(y1 + 1, min(h, int(y + bh)))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            ground_pt = (cx, float(y2))
            bbox = (x1, y1, x2, y2)
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

        img_h, img_w = frame.shape[:2]
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
                x, y, bw, bh = cv2.boundingRect(cnt)
                x1 = max(0, min(img_w - 1, int(x)))
                y1 = max(0, min(img_h - 1, int(y)))
                x2 = max(x1 + 1, min(img_w, int(x + bw)))
                y2 = max(y1 + 1, min(img_h, int(y + bh)))
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                ground_pt = (cx, float(y2))
                bbox = (x1, y1, x2, y2)
                hist = compute_hsv_histogram(frame, bbox)

                detections.append(Detection2D(
                    bbox=bbox,
                    confidence=min(1.0, area / 1000.0),
                    class_id=10,
                    class_name=f"{self.color}_target",
                    centroid=(cx, cy),
                    ground_point=ground_pt,
                    color_hist=hist,
                ))

        return detections


class OpenCVDNNDetector(BaseDetector):
    """
    OpenCV DNN Object Detector loading ONNX models (e.g. YOLOv8 / YOLOv5 / MobileNet / SSD).
    Supports aspect-ratio preserving letterbox preprocessing and precise coordinate transformation.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        classes: Optional[List[str]] = None,
        use_letterbox: bool = True,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DNN Model file not found: {model_path}")

        self.net = cv2.dnn.readNet(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.classes = classes or ["object"]
        self.use_letterbox = use_letterbox

    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        if frame is None:
            return []

        h, w = frame.shape[:2]

        if self.use_letterbox:
            input_img, scale, (pad_x, pad_y) = letterbox_image(frame, self.input_size)
            blob = cv2.dnn.blobFromImage(
                input_img, 1 / 255.0, self.input_size, swapRB=True, crop=False
            )
        else:
            scale = 1.0
            pad_x, pad_y = 0.0, 0.0
            blob = cv2.dnn.blobFromImage(
                frame, 1 / 255.0, self.input_size, swapRB=True, crop=False
            )

        self.net.setInput(blob)
        outputs = self.net.forward()

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        detections: List[Detection2D] = []
        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        # 1. SSD format: 4D tensor (1, 1, N, 7) or (1, N, 7)
        if len(outputs.shape) == 4 and outputs.shape[3] == 7:
            detections_data = outputs[0, 0]
            for row in detections_data:
                conf = float(row[2])
                if conf >= self.conf_threshold:
                    cid = int(row[1])
                    x1_m = row[3] * self.input_size[0]
                    y1_m = row[4] * self.input_size[1]
                    x2_m = row[5] * self.input_size[0]
                    y2_m = row[6] * self.input_size[1]

                    if self.use_letterbox:
                        x1 = int(round((x1_m - pad_x) / scale))
                        y1 = int(round((y1_m - pad_y) / scale))
                        x2 = int(round((x2_m - pad_x) / scale))
                        y2 = int(round((y2_m - pad_y) / scale))
                    else:
                        x1 = int(round(row[3] * w))
                        y1 = int(round(row[4] * h))
                        x2 = int(round(row[5] * w))
                        y2 = int(round(row[6] * h))

                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(x1 + 1, min(w, x2))
                    y2 = max(y1 + 1, min(h, y2))
                    bw = x2 - x1
                    bh = y2 - y1

                    boxes.append([x1, y1, bw, bh])
                    confidences.append(conf)
                    class_ids.append(cid)

        # 2. YOLO format: 3D tensor (1, C, N) or (1, N, C)
        elif len(outputs.shape) == 3:
            if outputs.shape[1] < outputs.shape[2]:
                outputs = np.transpose(outputs, (0, 2, 1))

            rows = outputs[0]
            num_channels = rows.shape[1]

            # Detect normalized coordinates [0..1] vs absolute model pixel coordinates [0..640]
            is_normalized = bool(np.max(rows[:, :4]) <= 1.5)

            # Check if tensor has YOLOv5 objectness score (channel 4) vs direct class probabilities
            has_objectness = (num_channels > 5 and num_channels != 84 and num_channels != len(self.classes) + 4)

            for row in rows:
                if has_objectness:
                    obj_conf = float(row[4])
                    if obj_conf < self.conf_threshold:
                        continue
                    class_scores = row[5:]
                    if len(class_scores) == 0:
                        class_id = 0
                        conf = obj_conf
                    else:
                        class_id = int(np.argmax(class_scores))
                        conf = float(obj_conf * class_scores[class_id])
                else:
                    class_scores = row[4:]
                    class_id = int(np.argmax(class_scores))
                    conf = float(class_scores[class_id])

                if conf >= self.conf_threshold:
                    cx_raw, cy_raw, bw_raw, bh_raw = float(row[0]), float(row[1]), float(row[2]), float(row[3])

                    if is_normalized:
                        cx_m = cx_raw * self.input_size[0]
                        cy_m = cy_raw * self.input_size[1]
                        bw_m = bw_raw * self.input_size[0]
                        bh_m = bh_raw * self.input_size[1]
                    else:
                        cx_m, cy_m, bw_m, bh_m = cx_raw, cy_raw, bw_raw, bh_raw

                    if self.use_letterbox:
                        cx = (cx_m - pad_x) / scale
                        cy = (cy_m - pad_y) / scale
                        bw = bw_m / scale
                        bh = bh_m / scale
                    else:
                        cx = cx_m * (w / self.input_size[0])
                        cy = cy_m * (h / self.input_size[1])
                        bw = bw_m * (w / self.input_size[0])
                        bh = bh_m * (h / self.input_size[1])

                    x1 = int(round(cx - bw / 2.0))
                    y1 = int(round(cy - bh / 2.0))
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    bw = max(1, min(w - x1, int(round(bw))))
                    bh = max(1, min(h - y1, int(round(bh))))

                    boxes.append([x1, y1, bw, bh])
                    confidences.append(conf)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        if len(indices) > 0:
            for idx in (indices.flatten() if hasattr(indices, "flatten") else [indices[0]]):
                x, y, bw, bh = boxes[idx]
                x1 = max(0, min(w - 1, int(x)))
                y1 = max(0, min(h - 1, int(y)))
                x2 = max(x1 + 1, min(w, int(x + bw)))
                y2 = max(y1 + 1, min(h, int(y + bh)))
                bbox = (x1, y1, x2, y2)

                cid = class_ids[idx]
                cname = self.classes[cid] if cid < len(self.classes) else f"class_{cid}"
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                ground_pt = (cx, float(y2))
                hist = compute_hsv_histogram(frame, bbox)

                detections.append(Detection2D(
                    bbox=bbox,
                    confidence=confidences[idx],
                    class_id=cid,
                    class_name=cname,
                    centroid=(cx, cy),
                    ground_point=ground_pt,
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
