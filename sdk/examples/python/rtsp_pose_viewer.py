#!/usr/bin/env python3
"""
RTSP / Webcam Pose & Detection Viewer — Google Coral TPU + OpenCV DNN.

Connects to a live RTSP video stream *or* a local webcam and runs either:
  • Real-time human pose estimation (MoveNet Lightning/Thunder on Coral TPU), or
  • Object detection only (YOLO/ONNX via OpenCV DNN, or Haar Cascades — no TPU needed).

Usage:
    # RTSP stream — pose estimation (default)
    conda run -n Laser python rtsp_pose_viewer.py --url rtsp://192.168.1.100/stream1
    conda run -n Laser python rtsp_pose_viewer.py --url rtsp://192.168.1.100/stream1 --model thunder
    conda run -n Laser python rtsp_pose_viewer.py --url rtsp://192.168.1.100/stream1 --multi

    # Local webcam — pose estimation
    conda run -n Laser python rtsp_pose_viewer.py --webcam
    conda run -n Laser python rtsp_pose_viewer.py --webcam 2          # device /dev/video2

    # Detection-only mode — Haar cascade (zero setup)
    conda run -n Laser python rtsp_pose_viewer.py --webcam --detect face
    conda run -n Laser python rtsp_pose_viewer.py --webcam --detect fullbody

    # Detection-only mode — YOLO / ONNX DNN model
    conda run -n Laser python rtsp_pose_viewer.py --webcam --detect dnn \\
        --detect-model yolov8n.onnx --detect-classes coco.txt
    conda run -n Laser python rtsp_pose_viewer.py --url rtsp://... --detect dnn \\
        --detect-model yolov8n.onnx --detect-classes coco.txt --detect-conf 0.4

Keyboard Controls:
    q / ESC   : Quit
    m         : Switch pose model (Lightning ↔ Thunder)  [pose mode only]
    t         : Toggle TPU / CPU inference                [pose mode only]
    s         : Toggle skeleton overlay                   [pose mode only]
    b         : Toggle bounding boxes
    a         : Toggle joint angle labels                 [pose mode only]
    k         : Toggle keypoint dot labels                [pose mode only]
    h         : Toggle help / HUD overlay
    c         : Save snapshot to disk
    r         : Reset keypoint smoother                   [pose mode only]
    f         : Toggle fullscreen
"""

import argparse
import datetime
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─── Ensure local modules are importable ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coral_pose_engine import (
    BODY_PART_COLORS,
    KEYPOINT_NAMES,
    SKELETON_EDGES,
    CoralPoseDetector,
    MultiPersonCoralPoseEngine,
    PoseEstimate,
    Keypoint,
)
from object_detector import (
    Detection2D,
    BaseDetector,
    CascadeDetector,
    OpenCVDNNDetector,
    create_detector,
)

# ─── Low-latency FFMPEG capture ───────────────────────────────────────────────
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
)

# ─── Per-person track colors (deterministic by index) ─────────────────────────
PERSON_COLORS = [
    (0, 255, 128),    # Spring green
    (0, 165, 255),    # Orange
    (255, 0, 255),    # Magenta
    (0, 215, 255),    # Gold
    (255, 64, 64),    # Coral red
    (64, 200, 255),   # Sky blue
]


def person_color(idx: int) -> Tuple[int, int, int]:
    return PERSON_COLORS[idx % len(PERSON_COLORS)]


# ──────────────────────────────────────────────────────────────────────────────
# RTSP Stream Reader
# ──────────────────────────────────────────────────────────────────────────────
class RTSPStreamReader:
    """Threaded RTSP reader with automatic reconnect and low-latency buffering."""

    def __init__(self, src: str, name: str = "Stream", reconnect_delay: float = 2.0):
        self.src = src
        self.name = name
        self.reconnect_delay = reconnect_delay

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._grabbed: bool = False
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.frame_count: int = 0
        self.drop_count: int = 0
        self._last_frame_time: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> "RTSPStreamReader":
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"RTSPReader-{self.name}")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if not self._grabbed or self._frame is None:
                return False, None
            return True, self._frame.copy()

    @property
    def is_connected(self) -> bool:
        return self._grabbed and (time.perf_counter() - self._last_frame_time) < 3.0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            grabbed, _ = cap.read()
            if grabbed:
                print(f"[{self.name}] Connected to {self.src}")
                return cap
        cap.release()
        return None

    def _run(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        while self._running:
            if cap is None or not cap.isOpened():
                cap = self._open_capture()
                if cap is None:
                    print(f"[{self.name}] Could not connect, retrying in {self.reconnect_delay}s...")
                    time.sleep(self.reconnect_delay)
                    continue

            grabbed, frame = cap.read()
            if not grabbed or frame is None:
                print(f"[{self.name}] Stream lost, reconnecting...")
                cap.release()
                cap = None
                with self._lock:
                    self._grabbed = False
                time.sleep(self.reconnect_delay)
                continue

            with self._lock:
                self._frame = frame
                self._grabbed = True
                self._last_frame_time = time.perf_counter()
            self.frame_count += 1

        if cap and cap.isOpened():
            cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# Webcam Reader
# ──────────────────────────────────────────────────────────────────────────────
class WebcamReader:
    """Thin threaded wrapper around a local webcam device.

    Exposes the same ``start`` / ``stop`` / ``read`` interface as
    :class:`RTSPStreamReader` so the rest of the pipeline is identical.
    """

    def __init__(self, device: int = 0, name: str = "Webcam"):
        self.device = device
        self.name = name
        self.label = f"webcam:{device}"   # used where stream_url is displayed

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._grabbed: bool = False
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.frame_count: int = 0
        self._last_frame_time: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> "WebcamReader":
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"WebcamReader-{self.name}")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if not self._grabbed or self._frame is None:
                return False, None
            return True, self._frame.copy()

    @property
    def is_connected(self) -> bool:
        return self._grabbed and (time.perf_counter() - self._last_frame_time) < 3.0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            print(f"[{self.name}] ERROR: Cannot open webcam device {self.device}")
            return
        print(f"[{self.name}] Opened /dev/video{self.device} "
              f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
              f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps)")

        while self._running:
            grabbed, frame = cap.read()
            if not grabbed or frame is None:
                # Transient failure — just keep trying
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._grabbed = True
                self._last_frame_time = time.perf_counter()
            self.frame_count += 1

        cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# Detection-only Rendering
# ──────────────────────────────────────────────────────────────────────────────

# Palette: one BGR color per class_id (mod len)
_DET_COLORS = [
    (0, 255, 128),   # spring green
    (0, 165, 255),   # orange
    (255, 0, 255),   # magenta
    (0, 215, 255),   # gold
    (255, 64, 64),   # coral red
    (64, 200, 255),  # sky blue
    (180, 255, 0),   # lime
    (255, 128, 0),   # azure
]


def _det_color(class_id: int) -> Tuple[int, int, int]:
    return _DET_COLORS[class_id % len(_DET_COLORS)]


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection2D],
    show_labels: bool = True,
    show_conf: bool = True,
) -> None:
    """Render bounding boxes and labels for a list of Detection2D results."""
    fh, fw = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = _det_color(det.class_id)

        # Box with glow effect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        if show_labels:
            label = det.class_name
            if show_conf:
                label = f"{label}  {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            by = y1 - 8 if y1 > th + 12 else y2 + th + 8
            cv2.rectangle(frame, (x1, by - th - 4), (x1 + tw + 8, by + 2), (20, 20, 20), -1)
            cv2.rectangle(frame, (x1, by - th - 4), (x1 + tw + 8, by + 2), color, 1)
            cv2.putText(frame, label, (x1 + 4, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        # Centroid dot
        cx, cy = int(det.centroid[0]), int(det.centroid[1])
        cv2.circle(frame, (cx, cy), 4, (0, 0, 0), -1)
        cv2.circle(frame, (cx, cy), 3, color, -1)


# ──────────────────────────────────────────────────────────────────────────────
# Skeleton + Keypoint Rendering
# ──────────────────────────────────────────────────────────────────────────────

def draw_pose(
    frame: np.ndarray,
    pose: PoseEstimate,
    person_idx: int = 0,
    min_kp_conf: float = 0.20,
    show_skeleton: bool = True,
    show_boxes: bool = True,
    show_angles: bool = True,
    show_kp_labels: bool = False,
) -> None:
    """Render skeleton, keypoints, bounding box, and joint angles on frame."""
    p_color = person_color(person_idx)
    fh, fw = frame.shape[:2]

    # ── Bounding Box ──────────────────────────────────────────────────────────
    if show_boxes and any(kp.score >= min_kp_conf for kp in pose.keypoints.values()):
        x1, y1, x2, y2 = pose.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), p_color, 2, cv2.LINE_AA)
        # Label badge: person index + avg confidence
        badge = f"Person {person_idx + 1}  ({pose.score:.0%})"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        by = y1 - 8 if y1 > th + 12 else y2 + th + 8
        cv2.rectangle(frame, (x1, by - th - 4), (x1 + tw + 8, by + 2), (20, 20, 20), -1)
        cv2.rectangle(frame, (x1, by - th - 4), (x1 + tw + 8, by + 2), p_color, 1)
        cv2.putText(frame, badge, (x1 + 4, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    # ── Skeleton Limbs ────────────────────────────────────────────────────────
    if show_skeleton:
        for (j1, j2, group) in SKELETON_EDGES:
            kp1 = pose.keypoints.get(j1)
            kp2 = pose.keypoints.get(j2)
            if (kp1 and kp2 and kp1.score >= min_kp_conf and kp2.score >= min_kp_conf):
                edge_color = BODY_PART_COLORS.get(group, p_color)
                # Scale alpha by minimum confidence of the two endpoints
                alpha = min(kp1.score, kp2.score)
                # Draw thicker background for depth effect
                cv2.line(frame, kp1.point, kp2.point, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.line(frame, kp1.point, kp2.point, edge_color, 2, cv2.LINE_AA)

    # ── Keypoint Circles ──────────────────────────────────────────────────────
    for name, kp in pose.keypoints.items():
        if kp.score < min_kp_conf:
            continue
        px, py = kp.point
        radius = 5 if kp.score >= 0.6 else 3
        # Glow ring
        cv2.circle(frame, (px, py), radius + 2, (0, 0, 0), -1)
        cv2.circle(frame, (px, py), radius, (255, 255, 255), -1)

        if show_kp_labels:
            short = name.replace("left_", "L.").replace("right_", "R.").replace("_", "")
            cv2.putText(frame, short, (px + 6, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Joint Angle Labels ────────────────────────────────────────────────────
    if show_angles and pose.joint_angles:
        angle_label_map = {
            "left_elbow":    "left_elbow",
            "right_elbow":   "right_elbow",
            "left_knee":     "left_knee",
            "right_knee":    "right_knee",
            "left_shoulder": "left_shoulder",
            "right_shoulder":"right_shoulder",
        }
        for angle_name, kp_name in angle_label_map.items():
            angle = pose.joint_angles.get(angle_name)
            kp = pose.keypoints.get(kp_name)
            if angle is not None and kp and kp.score >= min_kp_conf:
                px, py = kp.point
                label = f"{angle:.0f}\u00b0"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                # Slight offset from the joint
                lx = max(0, min(fw - tw - 4, px - tw // 2))
                ly = max(th + 4, min(fh - 4, py - 14))
                cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 4, ly + 2), (10, 10, 10), -1)
                cv2.putText(frame, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# HUD Overlay
# ──────────────────────────────────────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    fps: float,
    inference_ms: float,
    person_count: int,
    model_name: str,
    tpu_active: bool,
    multi_mode: bool,
    stream_url: str,
    show_help: bool,
    detect_mode: Optional[str] = None,   # non-None in detection-only mode
) -> None:
    """Render a premium telemetry HUD onto the frame."""
    fh, fw = frame.shape[:2]

    # ── Top-left telemetry panel ──────────────────────────────────────────────
    if detect_mode:
        obj_label = f"Det: {person_count} object{'s' if person_count != 1 else ''}"
        lines = [
            (f"FPS:  {fps:5.1f}", (0, 255, 128)),
            (f"Lat:  {inference_ms:.1f} ms", (0, 215, 255)),
            (obj_label, (255, 255, 255)),
            (f"Detector: {detect_mode}", (200, 200, 200)),
            ("Mode: Detection-Only", (255, 165, 0)),
        ]
    else:
        lines = [
            (f"FPS:  {fps:5.1f}", (0, 255, 128)),
            (f"Lat:  {inference_ms:.1f} ms", (0, 215, 255)),
            (f"Pose: {person_count} person{'s' if person_count != 1 else ''}", (255, 255, 255)),
            (f"Model: MoveNet {model_name.capitalize()}", (200, 200, 200)),
            (f"Accel: {'Coral TPU  ' if tpu_active else 'CPU (Fallback)'}", (0, 200, 255) if tpu_active else (100, 100, 255)),
            (f"Mode: {'Multi-Person' if multi_mode else 'Single-Person'}", (200, 200, 200)),
        ]

    panel_w, panel_h = 240, len(lines) * 20 + 16
    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (50, 50, 50), 1)

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (16, 26 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    # ── Top-right badge ───────────────────────────────────────────────────────
    if detect_mode:
        badge = "DETECT"
        badge_color = (0, 165, 255)   # orange
    else:
        badge = "CORAL TPU" if tpu_active else "CPU MODE"
        badge_color = (0, 200, 80) if tpu_active else (60, 60, 255)
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    bx = fw - bw - 20
    by = 24
    cv2.rectangle(frame, (bx - 8, by - bh - 6), (bx + bw + 8, by + 6), (10, 10, 10), -1)
    cv2.rectangle(frame, (bx - 8, by - bh - 6), (bx + bw + 8, by + 6), badge_color, 2)
    cv2.putText(frame, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.52, badge_color, 2, cv2.LINE_AA)

    # ── Stream URL (bottom left) ──────────────────────────────────────────────
    url_label = f"[Stream] {stream_url}"
    cv2.putText(frame, url_label, (10, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    # ── Timestamp (bottom right) ──────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:12]
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(frame, ts, (fw - tw - 10, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)

    # ── Help Overlay ──────────────────────────────────────────────────────────
    if show_help:
        keys = [
            ("q / ESC", "Quit"),
            ("m",       "Switch model (Lightning / Thunder)"),
            ("t",       "Toggle TPU / CPU"),
            ("s",       "Toggle skeleton"),
            ("b",       "Toggle bounding boxes"),
            ("a",       "Toggle joint angles"),
            ("k",       "Toggle keypoint labels"),
            ("h",       "Toggle this help"),
            ("c",       "Save snapshot"),
            ("r",       "Reset smoother"),
            ("f",       "Fullscreen"),
        ]
        hx, hy = fw // 2 - 160, fh // 2 - len(keys) * 14 - 10
        hp_h = len(keys) * 22 + 24
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (hx - 10, hy - 10), (hx + 330, hy + hp_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay2, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (hx - 10, hy - 10), (hx + 330, hy + hp_h), (80, 80, 80), 1)
        cv2.putText(frame, "KEYBOARD CONTROLS", (hx + 60, hy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)
        for i, (key, desc) in enumerate(keys):
            ty = hy + 36 + i * 22
            cv2.putText(frame, f"[{key}]", (hx + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (hx + 90, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)


def draw_no_stream(frame: np.ndarray, source_label: str, is_webcam: bool = False) -> None:
    """Render a 'Waiting for stream / webcam' placeholder."""
    fh, fw = frame.shape[:2]
    frame[:] = (15, 15, 25)  # Dark background
    msg1 = "Waiting for webcam..." if is_webcam else "Connecting to RTSP stream..."
    msg2 = source_label
    msg3 = ("Check device index / permissions" if is_webcam
             else "Check stream URL and network")
    cv2.putText(frame, msg1, (fw // 2 - 180, fh // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, msg2, (fw // 2 - min(fw // 2 - 20, len(msg2) * 7), fh // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, msg3, (fw // 2 - 160, fh // 2 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RTSP / Webcam Pose Viewer – Google Coral TPU Human Pose Estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source — mutually exclusive: RTSP URL vs local webcam
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--url", default="rtsp://127.0.0.1:8554/live",
                              help="RTSP stream URL (ignored when --webcam is set)")
    source_group.add_argument("--webcam", nargs="?", const=0, type=int, metavar="INDEX",
                              help="Use a local webcam instead of an RTSP stream. "
                                   "Optionally specify the device index (default: 0).")

    # Pose model (ignored in --detect mode)
    parser.add_argument("--model", default="lightning", choices=["lightning", "thunder"],
                        help="MoveNet model variant (lightning=fast, thunder=accurate)")
    parser.add_argument("--multi", action="store_true", default=False,
                        help="Enable multi-person mode (SSD MobileNet + MoveNet per person)")
    parser.add_argument("--no-tpu", dest="tpu", action="store_false", default=True,
                        help="Disable Coral TPU, run inference on CPU")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Minimum keypoint confidence threshold [0.0–1.0]")
    parser.add_argument("--max-persons", type=int, default=4,
                        help="Maximum number of persons to track in multi-person mode")

    # Detection-only mode
    parser.add_argument(
        "--detect", metavar="BACKEND", default=None,
        help=(
            "Switch to detection-only mode (no pose estimation). "
            "BACKEND choices: 'face', 'fullbody', 'upperbody' (Haar cascade, zero setup), "
            "or 'dnn' (ONNX model via --detect-model). "
            "Example: --detect face  |  --detect dnn --detect-model yolov8n.onnx"
        ),
    )
    parser.add_argument("--detect-model", metavar="PATH", default=None,
                        help="Path to an ONNX model file (required when --detect dnn).")
    parser.add_argument(
        "--detect-classes", metavar="PATH_OR_LIST", default=None,
        help=(
            "Class names for the DNN detector. Either a path to a text file "
            "(one class per line) or a comma-separated list, e.g. 'person,car,dog'."
        ),
    )
    parser.add_argument("--detect-conf", type=float, default=0.35,
                        help="Confidence threshold for the detection-only detector [0.0–1.0].")
    parser.add_argument("--detect-input-size", type=int, default=640,
                        help="Square input resolution fed to DNN detector (default 640).")

    # Display
    parser.add_argument("--width", type=int, default=None,
                        help="Resize display frame width (e.g. 1280)")
    parser.add_argument("--height", type=int, default=None,
                        help="Resize display frame height (e.g. 720)")
    parser.add_argument("--no-skeleton", dest="skeleton", action="store_false", default=True,
                        help="Disable skeleton limb overlay")
    parser.add_argument("--no-boxes", dest="boxes", action="store_false", default=True,
                        help="Disable bounding box overlay")
    parser.add_argument("--no-angles", dest="angles", action="store_false", default=True,
                        help="Disable joint angle labels")
    parser.add_argument("--kp-labels", action="store_true", default=False,
                        help="Show keypoint name labels on frame")
    parser.add_argument("--record", type=str, default=None,
                        help="Record output video to this path (e.g. output.mp4)")

    args = parser.parse_args()

    # ── Resolve source ────────────────────────────────────────────────────────
    use_webcam: bool = args.webcam is not None
    webcam_index: int = args.webcam if use_webcam else 0
    source_label: str = f"webcam:{webcam_index}" if use_webcam else args.url

    # ── Resolve detection mode ────────────────────────────────────────────────
    use_detect: bool = args.detect is not None
    detector: Optional[BaseDetector] = None
    detect_label: Optional[str] = None   # shown in HUD

    if use_detect:
        backend = args.detect.lower()
        if backend == "dnn":
            if not args.detect_model:
                print("[ERROR] --detect dnn requires --detect-model <path.onnx>")
                sys.exit(1)
            # Parse class names
            classes: Optional[List[str]] = None
            if args.detect_classes:
                if os.path.isfile(args.detect_classes):
                    with open(args.detect_classes) as f:
                        classes = [l.strip() for l in f if l.strip()]
                else:
                    classes = [c.strip() for c in args.detect_classes.split(",") if c.strip()]
            detect_label = f"DNN/{os.path.basename(args.detect_model)}"
            detector = OpenCVDNNDetector(
                model_path=args.detect_model,
                conf_threshold=args.detect_conf,
                input_size=(args.detect_input_size, args.detect_input_size),
                classes=classes,
            )
        else:
            # Haar cascade: face / fullbody / upperbody
            detect_label = f"Cascade/{backend}"
            detector = CascadeDetector(cascade_type=backend)
        print(f"[Detector] Initialized: {detect_label}")

    engine: Optional[MultiPersonCoralPoseEngine] = None

    # ── Initialize Pose Engine (skipped in detect mode) ───────────────────────
    print(f"\n{'='*60}")
    print("  RTSP / Webcam Pose & Detection Viewer")
    print(f"{'='*60}")
    if use_webcam:
        print(f"  Source      : Webcam (device {webcam_index})")
    else:
        print(f"  Stream URL  : {args.url}")
    if use_detect:
        print(f"  Mode        : Detection-Only ({detect_label})")
        print(f"  Det Conf    : {args.detect_conf}")
    else:
        print(f"  Mode        : Pose Estimation")
        print(f"  Pose Model  : MoveNet {args.model.capitalize()}")
        print(f"  Multi-Person: {'Yes' if args.multi else 'No'}")
        print(f"  Coral TPU   : {'Enabled' if args.tpu else 'Disabled (CPU)'}")
        print(f"  Conf Thresh : {args.conf}")
    print(f"{'='*60}\n")

    if not use_detect:
        engine = MultiPersonCoralPoseEngine(
            model_type=args.model,
            use_tpu=args.tpu,
            multi_person=args.multi,
            conf_threshold=args.conf,
            max_persons=args.max_persons,
        )

    # ── Stream / Webcam ───────────────────────────────────────────────────────
    if use_webcam:
        stream: RTSPStreamReader | WebcamReader = WebcamReader(device=webcam_index, name="PoseWebcam")
    else:
        stream = RTSPStreamReader(src=args.url, name="PoseStream")
    stream.start()

    # ── OpenCV Window ─────────────────────────────────────────────────────────
    win_title = "RTSP Pose Viewer — Coral TPU"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    if args.width and args.height:
        cv2.resizeWindow(win_title, args.width, args.height)

    # ── Video Writer ──────────────────────────────────────────────────────────
    writer: Optional[cv2.VideoWriter] = None
    writer_path = args.record

    # ── State ─────────────────────────────────────────────────────────────────
    show_skeleton = args.skeleton
    show_boxes    = args.boxes
    show_angles   = args.angles
    show_kp_labels = args.kp_labels
    show_help     = True       # show help on first launch
    is_fullscreen = False

    fps_counter_frames = 0
    fps_t0 = time.perf_counter()
    fps = 0.0
    last_inference_ms = 0.0
    last_person_count = 0

    # Placeholder frame when stream is disconnected
    placeholder_w = args.width or 1280
    placeholder_h = args.height or 720
    placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)

    print("\nInteractive Keyboard Controls:")
    print("  q / ESC  : Quit")
    if not use_detect:
        print("  m        : Switch model (Lightning ↔ Thunder)")
        print("  t        : Toggle Coral TPU / CPU inference")
        print("  s        : Toggle skeleton overlay")
    print("  b        : Toggle bounding boxes")
    if not use_detect:
        print("  a        : Toggle joint angle labels")
        print("  k        : Toggle keypoint labels")
    print("  h        : Toggle help / HUD")
    print("  c        : Save snapshot")
    if not use_detect:
        print("  r        : Reset keypoint smoother")
    print("  f        : Toggle fullscreen\n")

    try:
        while True:
            # ── Read Frame ────────────────────────────────────────────────────
            grabbed, frame = stream.read()

            if not grabbed or frame is None:
                draw_no_stream(placeholder, source_label, is_webcam=use_webcam)
                display = placeholder.copy()
                draw_hud(
                    display,
                    fps=0.0,
                    inference_ms=0.0,
                    person_count=0,
                    model_name=engine.pose_detector.model_type if engine else "",
                    tpu_active=engine.pose_detector.is_tpu_active if engine else False,
                    multi_mode=engine.multi_person if engine else False,
                    stream_url=source_label,
                    show_help=show_help,
                    detect_mode=detect_label,
                )
                cv2.imshow(win_title, display)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            # ── Optional resize ───────────────────────────────────────────────
            if args.width and args.height:
                frame = cv2.resize(frame, (args.width, args.height))
            elif args.width:
                h, w = frame.shape[:2]
                new_h = int(h * args.width / w)
                frame = cv2.resize(frame, (args.width, new_h))
            elif args.height:
                h, w = frame.shape[:2]
                new_w = int(w * args.height / h)
                frame = cv2.resize(frame, (new_w, args.height))

            # ── Run Inference ─────────────────────────────────────────────────
            t_inf_start = time.perf_counter()
            if use_detect:
                detections: List[Detection2D] = detector.detect(frame)  # type: ignore[union-attr]
                last_person_count = len(detections)
            else:
                poses: List[PoseEstimate] = engine.process_frame(frame)  # type: ignore[union-attr]
                last_person_count = len(poses)
            t_inf_end = time.perf_counter()
            last_inference_ms = (t_inf_end - t_inf_start) * 1000.0

            # ── Render Overlays ───────────────────────────────────────────────
            if use_detect:
                draw_detections(frame, detections, show_labels=show_boxes)
            else:
                for person_idx, pose in enumerate(poses):
                    draw_pose(
                        frame,
                        pose,
                        person_idx=person_idx,
                        min_kp_conf=args.conf,
                        show_skeleton=show_skeleton,
                        show_boxes=show_boxes,
                        show_angles=show_angles,
                        show_kp_labels=show_kp_labels,
                    )

            # ── FPS Calculation ───────────────────────────────────────────────
            fps_counter_frames += 1
            elapsed = time.perf_counter() - fps_t0
            if elapsed >= 1.0:
                fps = fps_counter_frames / elapsed
                fps_counter_frames = 0
                fps_t0 = time.perf_counter()

            # ── HUD ───────────────────────────────────────────────────────────
            draw_hud(
                frame,
                fps=fps,
                inference_ms=last_inference_ms,
                person_count=last_person_count,
                model_name=engine.pose_detector.model_type if engine else "",
                tpu_active=engine.pose_detector.is_tpu_active if engine else False,
                multi_mode=engine.multi_person if engine else False,
                stream_url=source_label,
                show_help=show_help,
                detect_mode=detect_label,
            )

            # ── Display ───────────────────────────────────────────────────────
            cv2.imshow(win_title, frame)

            # ── Optional recording ────────────────────────────────────────────
            if writer_path:
                if writer is None:
                    fh, fw = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(writer_path, fourcc, 30.0, (fw, fh))
                    print(f"[Recorder] Writing to: {writer_path}")
                writer.write(frame)

            # ── Keyboard Input ────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):           # Quit
                break

            elif key == ord("m") and not use_detect:   # Switch pose model
                next_model = "thunder" if engine.pose_detector.model_type == "lightning" else "lightning"  # type: ignore
                print(f"[Viewer] Switching to MoveNet {next_model.capitalize()}...")
                engine.pose_detector.set_model(next_model)  # type: ignore
                print(f"[Viewer] Now using MoveNet {next_model.capitalize()}")

            elif key == ord("t") and not use_detect:   # Toggle TPU/CPU
                active = engine.pose_detector.toggle_tpu()  # type: ignore
                print(f"[Viewer] Inference mode: {'Coral TPU' if active else 'CPU'}")

            elif key == ord("s") and not use_detect:   # Toggle skeleton
                show_skeleton = not show_skeleton
                print(f"[Viewer] Skeleton: {'ON' if show_skeleton else 'OFF'}")

            elif key == ord("b"):               # Toggle bounding boxes / detection labels
                show_boxes = not show_boxes
                print(f"[Viewer] Boxes/Labels: {'ON' if show_boxes else 'OFF'}")

            elif key == ord("a") and not use_detect:   # Toggle joint angles
                show_angles = not show_angles
                print(f"[Viewer] Angles: {'ON' if show_angles else 'OFF'}")

            elif key == ord("k") and not use_detect:   # Toggle keypoint labels
                show_kp_labels = not show_kp_labels
                print(f"[Viewer] Keypoint Labels: {'ON' if show_kp_labels else 'OFF'}")

            elif key == ord("h"):               # Toggle help
                show_help = not show_help

            elif key == ord("r") and not use_detect:   # Reset smoother
                if engine.pose_detector.smoother:  # type: ignore
                    engine.pose_detector.smoother.reset()  # type: ignore
                print("[Viewer] Keypoint smoother reset.")

            elif key == ord("c"):               # Snapshot
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = f"pose_snapshot_{ts}.jpg"
                cv2.imwrite(snap_path, frame)
                print(f"[Viewer] Snapshot saved: {snap_path}")

            elif key == ord("f"):               # Toggle fullscreen
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(win_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(win_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    except KeyboardInterrupt:
        print("\n[Viewer] Interrupted by user.")
    finally:
        print("[Viewer] Shutting down...")
        stream.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[Viewer] Done.")


if __name__ == "__main__":
    main()
