#!/usr/bin/env python3
"""
RTSP Pose Viewer — Google Coral TPU USB Accelerated Human Pose Estimation.

Connects to a live RTSP video stream and runs real-time human pose estimation
via off-the-shelf MoveNet models (Lightning or Thunder) on the Google Coral
USB Edge TPU.

Usage:
    conda run -n Laser python rtsp_pose_viewer.py --url rtsp://192.168.1.100/stream1
    conda run -n Laser python rtsp_pose_viewer.py --url rtsp://192.168.1.100/stream1 --model thunder
    conda run -n Laser python rtsp_pose_viewer.py --url rtsp://192.168.1.100/stream1 --multi

Keyboard Controls:
    q / ESC   : Quit
    m         : Switch model (Lightning ↔ Thunder)
    t         : Toggle TPU / CPU inference
    s         : Toggle skeleton overlay
    b         : Toggle bounding boxes
    a         : Toggle joint angle labels
    k         : Toggle keypoint dot labels
    h         : Toggle help / HUD overlay
    c         : Save snapshot to disk
    r         : Reset keypoint smoother
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
) -> None:
    """Render a premium telemetry HUD onto the frame."""
    fh, fw = frame.shape[:2]

    # ── Top-left telemetry panel ──────────────────────────────────────────────
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

    # ── Coral TPU badge (top-right) ───────────────────────────────────────────
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


def draw_no_stream(frame: np.ndarray, url: str) -> None:
    """Render a 'Waiting for stream' placeholder."""
    fh, fw = frame.shape[:2]
    frame[:] = (15, 15, 25)  # Dark background
    # Animated spinner idea via corner dots
    msg1 = "Connecting to RTSP stream..."
    msg2 = url
    msg3 = "Check stream URL and network"
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
        description="RTSP Pose Viewer – Google Coral TPU Human Pose Estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stream
    parser.add_argument("--url", default="rtsp://127.0.0.1:8554/live",
                        help="RTSP stream URL")

    # Model
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

    # ── Initialize Pose Engine ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RTSP Pose Viewer — Google Coral TPU Edition")
    print(f"{'='*60}")
    print(f"  Stream URL : {args.url}")
    print(f"  Model      : MoveNet {args.model.capitalize()}")
    print(f"  Multi-Person: {'Yes' if args.multi else 'No'}")
    print(f"  Coral TPU  : {'Enabled' if args.tpu else 'Disabled (CPU)'}")
    print(f"  Conf Thresh: {args.conf}")
    print(f"{'='*60}\n")

    engine = MultiPersonCoralPoseEngine(
        model_type=args.model,
        use_tpu=args.tpu,
        multi_person=args.multi,
        conf_threshold=args.conf,
        max_persons=args.max_persons,
    )

    # ── RTSP Stream ───────────────────────────────────────────────────────────
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
    print("  m        : Switch model (Lightning ↔ Thunder)")
    print("  t        : Toggle Coral TPU / CPU inference")
    print("  s        : Toggle skeleton overlay")
    print("  b        : Toggle bounding boxes")
    print("  a        : Toggle joint angle labels")
    print("  k        : Toggle keypoint labels")
    print("  h        : Toggle help / HUD")
    print("  c        : Save snapshot")
    print("  r        : Reset keypoint smoother")
    print("  f        : Toggle fullscreen\n")

    try:
        while True:
            # ── Read Frame ────────────────────────────────────────────────────
            grabbed, frame = stream.read()

            if not grabbed or frame is None:
                draw_no_stream(placeholder, args.url)
                display = placeholder.copy()
                draw_hud(
                    display,
                    fps=0.0,
                    inference_ms=0.0,
                    person_count=0,
                    model_name=engine.pose_detector.model_type,
                    tpu_active=engine.pose_detector.is_tpu_active,
                    multi_mode=engine.multi_person,
                    stream_url=args.url,
                    show_help=show_help,
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

            # ── Run Pose Estimation on Coral TPU ─────────────────────────────
            t_inf_start = time.perf_counter()
            poses: List[PoseEstimate] = engine.process_frame(frame)
            t_inf_end = time.perf_counter()
            last_inference_ms = (t_inf_end - t_inf_start) * 1000.0
            last_person_count = len(poses)

            # ── Render Pose Overlays ──────────────────────────────────────────
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
                model_name=engine.pose_detector.model_type,
                tpu_active=engine.pose_detector.is_tpu_active,
                multi_mode=engine.multi_person,
                stream_url=args.url,
                show_help=show_help,
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

            elif key == ord("m"):               # Switch model
                next_model = "thunder" if engine.pose_detector.model_type == "lightning" else "lightning"
                print(f"[Viewer] Switching to MoveNet {next_model.capitalize()}...")
                engine.pose_detector.set_model(next_model)
                print(f"[Viewer] Now using MoveNet {next_model.capitalize()}")

            elif key == ord("t"):               # Toggle TPU/CPU
                active = engine.pose_detector.toggle_tpu()
                print(f"[Viewer] Inference mode: {'Coral TPU' if active else 'CPU'}")

            elif key == ord("s"):               # Toggle skeleton
                show_skeleton = not show_skeleton
                print(f"[Viewer] Skeleton: {'ON' if show_skeleton else 'OFF'}")

            elif key == ord("b"):               # Toggle bounding boxes
                show_boxes = not show_boxes
                print(f"[Viewer] Boxes: {'ON' if show_boxes else 'OFF'}")

            elif key == ord("a"):               # Toggle joint angles
                show_angles = not show_angles
                print(f"[Viewer] Angles: {'ON' if show_angles else 'OFF'}")

            elif key == ord("k"):               # Toggle keypoint labels
                show_kp_labels = not show_kp_labels
                print(f"[Viewer] Keypoint Labels: {'ON' if show_kp_labels else 'OFF'}")

            elif key == ord("h"):               # Toggle help
                show_help = not show_help

            elif key == ord("r"):               # Reset smoother
                if engine.pose_detector.smoother:
                    engine.pose_detector.smoother.reset()
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
