#!/usr/bin/env python3
"""
Coral TPU Pose Estimation Engine.
Leverages Google Coral Edge TPU USB Accelerator to run off-the-shelf pose models
(MoveNet Lightning & MoveNet Thunder) with sub-10ms inference latencies.
"""

from dataclasses import dataclass, field
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import urllib.request
import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    try:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter
        load_delegate = tflite.experimental.load_delegate
    except ImportError:
        Interpreter = None
        load_delegate = None

# 17 COCO Keypoints definition
KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Standard skeleton connections: (joint_1_name, joint_2_name, body_part_group)
SKELETON_EDGES = [
    # Head / Face (Yellow / Orange)
    ("nose", "left_eye", "head"),
    ("nose", "right_eye", "head"),
    ("left_eye", "left_ear", "head"),
    ("right_eye", "right_ear", "head"),
    # Torso (Yellow / White)
    ("left_shoulder", "right_shoulder", "torso"),
    ("left_shoulder", "left_hip", "torso"),
    ("right_shoulder", "right_hip", "torso"),
    ("left_hip", "right_hip", "torso"),
    # Left Arm (Green / Cyan)
    ("left_shoulder", "left_elbow", "left_arm"),
    ("left_elbow", "left_wrist", "left_arm"),
    # Right Arm (Sky Blue / Purple)
    ("right_shoulder", "right_elbow", "right_arm"),
    ("right_elbow", "right_wrist", "right_arm"),
    # Left Leg (Lime Green)
    ("left_hip", "left_knee", "left_leg"),
    ("left_knee", "left_ankle", "left_leg"),
    # Right Leg (Magenta / Pink)
    ("right_hip", "right_knee", "right_leg"),
    ("right_knee", "right_ankle", "right_leg"),
]

# Color palettes (BGR format) for limbs and keypoint groups
BODY_PART_COLORS = {
    "head": (0, 215, 255),       # Gold / Yellow
    "torso": (255, 255, 0),      # Cyan / Yellow
    "left_arm": (0, 255, 128),   # Spring Green
    "right_arm": (255, 128, 0),  # Royal Blue / Cyan
    "left_leg": (50, 255, 50),   # Bright Green
    "right_leg": (255, 0, 255),  # Magenta
}

MODEL_URLS = {
    "movenet_lightning_tpu": (
        "https://github.com/google-coral/test_data/raw/master/movenet_single_pose_lightning_ptq_edgetpu.tflite",
        "movenet_single_pose_lightning_ptq_edgetpu.tflite",
        (192, 192),
    ),
    "movenet_thunder_tpu": (
        "https://github.com/google-coral/test_data/raw/master/movenet_single_pose_thunder_ptq_edgetpu.tflite",
        "movenet_single_pose_thunder_ptq_edgetpu.tflite",
        (256, 256),
    ),
    "movenet_lightning_cpu": (
        "https://github.com/google-coral/test_data/raw/master/movenet_single_pose_lightning_ptq.tflite",
        "movenet_single_pose_lightning_ptq.tflite",
        (192, 192),
    ),
    "ssd_mobilenet_tpu": (
        "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
        "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
        (300, 300),
    ),
}


@dataclass
class Keypoint:
    """Represents a single 2D joint detection."""
    name: str
    id: int
    x: float  # pixel x
    y: float  # pixel y
    score: float  # confidence [0.0, 1.0]

    @property
    def point(self) -> Tuple[int, int]:
        return int(round(self.x)), int(round(self.y))


@dataclass
class PoseEstimate:
    """Represents full 17-point pose estimation for a person."""
    keypoints: Dict[str, Keypoint] = field(default_factory=dict)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    score: float = 0.0
    joint_angles: Dict[str, float] = field(default_factory=dict)
    centroid: Tuple[float, float] = (0.0, 0.0)
    inference_time_ms: float = 0.0

    def get_keypoint(self, name: str, min_conf: float = 0.2) -> Optional[Keypoint]:
        kp = self.keypoints.get(name)
        if kp and kp.score >= min_conf:
            return kp
        return None


def calculate_angle(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float]
) -> float:
    """
    Calculate the interior angle (in degrees) between three 2D points with vertex at point b (a-b-c).
    Returns angle in [0, 180] degrees.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])

    if mag_ba * mag_bc < 1e-6:
        return 0.0

    cosine_angle = max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine_angle))


def compute_all_joint_angles(keypoints: Dict[str, Keypoint], min_conf: float = 0.25) -> Dict[str, float]:
    """Calculate key biomechanical joint angles (elbows, knees, shoulders, hips)."""
    angles = {}

    def valid(name: str) -> Optional[Tuple[float, float]]:
        kp = keypoints.get(name)
        if kp and kp.score >= min_conf:
            return (kp.x, kp.y)
        return None

    # Left Elbow: shoulder - elbow - wrist
    ls, le, lw = valid("left_shoulder"), valid("left_elbow"), valid("left_wrist")
    if ls and le and lw:
        angles["left_elbow"] = calculate_angle(ls, le, lw)

    # Right Elbow: shoulder - elbow - wrist
    rs, re, rw = valid("right_shoulder"), valid("right_elbow"), valid("right_wrist")
    if rs and re and rw:
        angles["right_elbow"] = calculate_angle(rs, re, rw)

    # Left Knee: hip - knee - ankle
    lh, lk, la = valid("left_hip"), valid("left_knee"), valid("left_ankle")
    if lh and lk and la:
        angles["left_knee"] = calculate_angle(lh, lk, la)

    # Right Knee: hip - knee - ankle
    rh, rk, ra = valid("right_hip"), valid("right_knee"), valid("right_ankle")
    if rh and rk and ra:
        angles["right_knee"] = calculate_angle(rh, rk, ra)

    # Left Shoulder: hip - shoulder - elbow
    if lh and ls and le:
        angles["left_shoulder"] = calculate_angle(lh, ls, le)

    # Right Shoulder: hip - shoulder - elbow
    if rh and rs and re:
        angles["right_shoulder"] = calculate_angle(rh, rs, re)

    return angles


def find_edgetpu_lib(custom_path: Optional[str] = None) -> Optional[str]:
    """Search for the libedgetpu shared library across standard and local locations."""
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "models", "libedgetpu.so.1"),
        os.path.join(base_dir, "models", "libedgetpu_extracted", "usr", "lib", "x86_64-linux-gnu", "libedgetpu.so.1.0"),
        "/usr/lib/x86_64-linux-gnu/libedgetpu.so.1.0",
        "/usr/lib/x86_64-linux-gnu/libedgetpu.so.1",
        "/usr/lib/libedgetpu.so.1",
        "libedgetpu.so.1",
    ]

    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def ensure_model_file(model_key: str, models_dir: Optional[str] = None) -> str:
    """Download model file if not already present locally."""
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    if model_key not in MODEL_URLS:
        raise ValueError(f"Unknown model key: {model_key}. Options: {list(MODEL_URLS.keys())}")

    url, filename, _ = MODEL_URLS[model_key]
    dest_path = os.path.join(models_dir, filename)

    if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1000:
        print(f"[CoralPoseEngine] Downloading {model_key} from {url}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"[CoralPoseEngine] Saved to {dest_path} ({os.path.getsize(dest_path)} bytes)")
        except Exception as e:
            raise RuntimeError(f"Failed to download model {model_key}: {e}")

    return dest_path


def letterbox_for_movenet(
    image: np.ndarray,
    target_size: Tuple[int, int]
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Letterbox BGR image into square input tensor preserving aspect ratio.
    Returns:
        letterboxed_image (RGB), scale_factor, (pad_x, pad_y)
    """
    ih, iw = image.shape[:2]
    tw, th = target_size
    scale = min(tw / iw, th / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (tw - nw) / 2.0
    pad_y = (th - nh) / 2.0

    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))

    # Pad with black border
    letterboxed_bgr = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    # MoveNet expects RGB uint8 image
    letterboxed_rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    return letterboxed_rgb, scale, (pad_x, pad_y)


class KeypointSmoother:
    """Exponential Moving Average (EMA) smoother to reduce frame-to-frame keypoint jitter."""

    def __init__(self, alpha: float = 0.65):
        self.alpha = alpha
        self.prev_kps: Dict[str, Tuple[float, float]] = {}

    def smooth(self, pose: PoseEstimate) -> PoseEstimate:
        smoothed_kps = {}
        for name, kp in pose.keypoints.items():
            if kp.score < 0.2:
                self.prev_kps.pop(name, None)
                smoothed_kps[name] = kp
                continue

            if name in self.prev_kps:
                prev_x, prev_y = self.prev_kps[name]
                new_x = self.alpha * kp.x + (1.0 - self.alpha) * prev_x
                new_y = self.alpha * kp.y + (1.0 - self.alpha) * prev_y
            else:
                new_x, new_y = kp.x, kp.y

            self.prev_kps[name] = (new_x, new_y)
            smoothed_kps[name] = Keypoint(
                name=name,
                id=kp.id,
                x=new_x,
                y=new_y,
                score=kp.score,
            )

        pose.keypoints = smoothed_kps
        pose.joint_angles = compute_all_joint_angles(smoothed_kps)
        return pose

    def reset(self) -> None:
        self.prev_kps.clear()


class CoralPoseDetector:
    """
    Core Pose Detector running MoveNet on Google Coral TPU USB.
    Supports dynamic switching between Lightning (high FPS) and Thunder (high accuracy).
    """

    def __init__(
        self,
        model_type: str = "lightning",
        use_tpu: bool = True,
        edgetpu_lib: Optional[str] = None,
        conf_threshold: float = 0.25,
        smooth: bool = True,
    ):
        self.model_type = model_type.lower()
        self.use_tpu = use_tpu
        self.conf_threshold = conf_threshold
        self.edgetpu_lib_path = find_edgetpu_lib(edgetpu_lib) if use_tpu else None
        self.smoother = KeypointSmoother() if smooth else None

        self.interpreter: Optional[Interpreter] = None
        self.is_tpu_active = False
        self._init_interpreter()

    def _init_interpreter(self) -> None:
        model_key = f"movenet_{self.model_type}_{'tpu' if self.use_tpu else 'cpu'}"
        if self.use_tpu and not self.edgetpu_lib_path:
            print("[CoralPoseDetector] Warning: libedgetpu library not found, falling back to CPU.")
            self.use_tpu = False
            model_key = f"movenet_{self.model_type}_cpu"

        model_path = ensure_model_file(model_key)
        self.target_size = MODEL_URLS[model_key][2]

        if self.use_tpu:
            try:
                delegate = load_delegate(self.edgetpu_lib_path)
                self.interpreter = Interpreter(
                    model_path=model_path,
                    experimental_delegates=[delegate]
                )
                self.interpreter.allocate_tensors()
                self.is_tpu_active = True
                print(f"[CoralPoseDetector] Initialized MoveNet ({self.model_type}) on Google Coral TPU USB!")
            except Exception as e:
                print(f"[CoralPoseDetector] Coral TPU load failed ({e}). Falling back to CPU.")
                self.use_tpu = False
                self.is_tpu_active = False
                cpu_key = f"movenet_{self.model_type}_cpu"
                cpu_model = ensure_model_file(cpu_key)
                self.interpreter = Interpreter(model_path=cpu_model)
                self.interpreter.allocate_tensors()
                print(f"[CoralPoseDetector] Initialized MoveNet ({self.model_type}) on CPU.")
        else:
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.is_tpu_active = False
            print(f"[CoralPoseDetector] Initialized MoveNet ({self.model_type}) on CPU.")

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]
        self.input_shape = self.input_details[0]["shape"]
        self.input_dtype = self.input_details[0]["dtype"]

    def set_model(self, model_type: str) -> None:
        """Switch between 'lightning' and 'thunder' models."""
        if model_type.lower() != self.model_type:
            self.model_type = model_type.lower()
            self._init_interpreter()
            if self.smoother:
                self.smoother.reset()

    def toggle_tpu(self) -> bool:
        """Toggle TPU acceleration on or off."""
        self.use_tpu = not self.use_tpu
        self._init_interpreter()
        return self.is_tpu_active

    def estimate_pose(
        self,
        frame: np.ndarray,
        crop_box: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[PoseEstimate]:
        """
        Estimate pose on the given frame (or a sub-region defined by crop_box).
        Args:
            frame: Full BGR video frame
            crop_box: Optional (x1, y1, x2, y2) region of interest
        Returns:
            PoseEstimate instance or None if input invalid
        """
        if frame is None or frame.size == 0:
            return None

        fh, fw = frame.shape[:2]
        if crop_box is not None:
            cx1, cy1, cx2, cy2 = crop_box
            cx1, cy1 = max(0, cx1), max(0, cy1)
            cx2, cy2 = min(fw, cx2), min(fh, cy2)
            if cx2 <= cx1 or cy2 <= cy1:
                return None
            input_crop = frame[cy1:cy2, cx1:cx2]
            offset_x, offset_y = cx1, cy1
        else:
            input_crop = frame
            offset_x, offset_y = 0, 0

        # Preprocessing: letterbox preserving aspect ratio
        letterboxed, scale, (pad_x, pad_y) = letterbox_for_movenet(input_crop, self.target_size)

        # Set input tensor (1, H, W, 3)
        input_data = np.expand_dims(letterboxed, axis=0).astype(self.input_dtype)
        self.interpreter.set_tensor(self.input_index, input_data)

        # Execute inference on Coral TPU / CPU
        t0 = time.perf_counter()
        self.interpreter.invoke()
        t1 = time.perf_counter()
        inf_time_ms = (t1 - t0) * 1000.0

        # Output shape: (1, 1, 17, 3) -> [y, x, score]
        outputs = self.interpreter.get_tensor(self.output_index)
        kps_raw = outputs[0, 0]  # shape: (17, 3)

        keypoints: Dict[str, Keypoint] = {}
        valid_x: List[float] = []
        valid_y: List[float] = []
        conf_sum = 0.0

        tw, th = self.target_size

        for idx, (y_norm, x_norm, score) in enumerate(kps_raw):
            name = KEYPOINT_NAMES[idx]
            conf = float(score)
            conf_sum += conf

            # Un-normalize from letterbox coordinates to crop space
            x_m = float(x_norm) * tw
            y_m = float(y_norm) * th

            # Remove padding and undo scaling
            x_crop = (x_m - pad_x) / scale
            y_crop = (y_m - pad_y) / scale

            # Map back to full frame pixel coordinates
            full_x = float(max(0, min(fw - 1, x_crop + offset_x)))
            full_y = float(max(0, min(fh - 1, y_crop + offset_y)))

            keypoints[name] = Keypoint(
                name=name,
                id=idx,
                x=full_x,
                y=full_y,
                score=conf,
            )

            if conf >= self.conf_threshold:
                valid_x.append(full_x)
                valid_y.append(full_y)

        avg_conf = conf_sum / len(KEYPOINT_NAMES)

        # Compute bounding box from keypoints if sufficient confidence
        if len(valid_x) >= 3:
            min_x, max_x = min(valid_x), max(valid_x)
            min_y, max_y = min(valid_y), max(valid_y)
            # Add small padding margin around person bbox
            pad_w = (max_x - min_x) * 0.15
            pad_h = (max_y - min_y) * 0.15
            bbox = (
                int(max(0, min_x - pad_w)),
                int(max(0, min_y - pad_h)),
                int(min(fw - 1, max_x + pad_w)),
                int(min(fh - 1, max_y + pad_h)),
            )
            centroid = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        elif crop_box is not None:
            bbox = crop_box
            centroid = ((crop_box[0] + crop_box[2]) / 2.0, (crop_box[1] + crop_box[3]) / 2.0)
        else:
            bbox = (0, 0, fw, fh)
            centroid = (fw / 2.0, fh / 2.0)

        joint_angles = compute_all_joint_angles(keypoints, min_conf=self.conf_threshold)

        pose = PoseEstimate(
            keypoints=keypoints,
            bbox=bbox,
            score=avg_conf,
            joint_angles=joint_angles,
            centroid=centroid,
            inference_time_ms=inf_time_ms,
        )

        if self.smoother and crop_box is None:
            pose = self.smoother.smooth(pose)

        return pose


class MultiPersonCoralPoseEngine:
    """
    Multi-Person Pose Pipeline:
    1. Uses Coral SSD-MobileNet to detect person bounding boxes in real-time (~6ms on TPU).
    2. Runs MoveNet on each person crop (~5ms per person on TPU).
    3. Supports fallback to single-pose mode for maximum framerate.
    """

    def __init__(
        self,
        model_type: str = "lightning",
        use_tpu: bool = True,
        multi_person: bool = True,
        conf_threshold: float = 0.25,
        max_persons: int = 4,
    ):
        self.multi_person = multi_person
        self.max_persons = max_persons
        self.conf_threshold = conf_threshold
        self.pose_detector = CoralPoseDetector(
            model_type=model_type,
            use_tpu=use_tpu,
            conf_threshold=conf_threshold,
            smooth=True,
        )

        self.person_detector_interpreter: Optional[Interpreter] = None
        self.person_detector_active = False
        if multi_person and self.pose_detector.is_tpu_active:
            self._init_person_detector()

    def _init_person_detector(self) -> None:
        try:
            model_path = ensure_model_file("ssd_mobilenet_tpu")
            edgetpu_lib = self.pose_detector.edgetpu_lib_path
            if edgetpu_lib:
                delegate = load_delegate(edgetpu_lib)
                self.person_detector_interpreter = Interpreter(
                    model_path=model_path,
                    experimental_delegates=[delegate]
                )
                self.person_detector_interpreter.allocate_tensors()
                self.person_detector_active = True
                print("[MultiPersonCoralPoseEngine] Person detector initialized on Coral TPU!")
        except Exception as e:
            print(f"[MultiPersonCoralPoseEngine] Person detector init error ({e}). Multi-person fallback.")
            self.person_detector_active = False

    def detect_persons(self, frame: np.ndarray, person_conf: float = 0.4) -> List[Tuple[int, int, int, int]]:
        """Detect person bounding boxes using Coral SSD MobileNet."""
        if not self.person_detector_active or self.person_detector_interpreter is None:
            return []

        fh, fw = frame.shape[:2]
        # SSD MobileNet expects 300x300 RGB
        resized = cv2.resize(frame, (300, 300))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb, axis=0)

        in_idx = self.person_detector_interpreter.get_input_details()[0]["index"]
        self.person_detector_interpreter.set_tensor(in_idx, input_data)
        self.person_detector_interpreter.invoke()

        out_details = self.person_detector_interpreter.get_output_details()
        boxes = self.person_detector_interpreter.get_tensor(out_details[0]["index"])[0]
        classes = self.person_detector_interpreter.get_tensor(out_details[1]["index"])[0]
        scores = self.person_detector_interpreter.get_tensor(out_details[2]["index"])[0]

        person_boxes = []
        for i in range(len(scores)):
            if scores[i] >= person_conf and int(classes[i]) == 0:  # Class 0 is person in COCO
                ymin, xmin, ymax, xmax = boxes[i]
                x1 = int(max(0, xmin * fw))
                y1 = int(max(0, ymin * fh))
                x2 = int(min(fw - 1, xmax * fw))
                y2 = int(min(fh - 1, ymax * fh))
                if (x2 - x1) > 20 and (y2 - y1) > 30:
                    person_boxes.append((x1, y1, x2, y2))
                    if len(person_boxes) >= self.max_persons:
                        break

        return person_boxes

    def process_frame(self, frame: np.ndarray) -> List[PoseEstimate]:
        """
        Process frame and return list of PoseEstimates.
        In multi-person mode: detects people and runs MoveNet per person.
        In single-person mode: runs MoveNet directly on the full frame.
        """
        if not self.multi_person or not self.person_detector_active:
            single_pose = self.pose_detector.estimate_pose(frame)
            return [single_pose] if single_pose is not None else []

        # Multi-person pipeline
        person_boxes = self.detect_persons(frame)
        if not person_boxes:
            # Fallback to full frame single pose if no person boxes detected
            single_pose = self.pose_detector.estimate_pose(frame)
            return [single_pose] if single_pose is not None else []

        poses = []
        for box in person_boxes:
            pose = self.pose_detector.estimate_pose(frame, crop_box=box)
            if pose is not None:
                poses.append(pose)

        return poses
