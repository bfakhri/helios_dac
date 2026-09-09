#!/usr/bin/env python3
"""
Time-Aligned Dual RTSP Stream 3D Object Detection & Triangulation System.
Captures dual RTSP video streams, aligns frames by timestamp, detects objects,
associates stereo correspondences, and triangulates 3D real-world coordinates.
"""

import argparse
import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from object_detector import BaseDetector, Detection2D, create_detector
from stereo_geometry import StereoCameraModel
from stereo_matcher import StereoMatch, StereoObjectMatcher, TrackSmoother3D
from stream_synchronizer import StreamSynchronizer, TimestampedFrame

# Low-latency FFmpeg capture configuration
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


class RTSPStreamReader:
    """Threaded RTSP reader that continuously captures timestamped frames into a synchronizer."""

    def __init__(self, src: str, stream_id: int, synchronizer: StreamSynchronizer, name: str = "Stream"):
        self.src = src
        self.stream_id = stream_id
        self.synchronizer = synchronizer
        self.name = name

        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.latest_frame = self.cap.read()
        self.running = self.grabbed
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        if self.grabbed and self.latest_frame is not None:
            self.synchronizer.push_frame(self.stream_id, self.latest_frame, time.perf_counter_ns())

    def start(self) -> "RTSPStreamReader":
        if not self.running:
            return self
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self) -> None:
        while self.running:
            grabbed, frame = self.cap.read()
            t_now = time.perf_counter_ns()
            if not grabbed:
                print(f"[{self.name}] Stream disconnected or ended.")
                self.running = False
                break

            with self.lock:
                self.grabbed = grabbed
                self.latest_frame = frame

            self.synchronizer.push_frame(self.stream_id, frame, timestamp_ns=t_now)

    def get_latest_fallback(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Fallback read if synchronization buffer is empty."""
        with self.lock:
            if not self.grabbed or self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate deterministic, visually distinct BGR colors per track ID."""
    colors = [
        (0, 255, 0),      # Bright Green
        (255, 128, 0),    # Cyan/Sky Blue
        (0, 165, 255),    # Orange
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 255),    # Purple
        (0, 215, 255),    # Gold
        (255, 191, 0),    # Deep Sky Blue
    ]
    return colors[(track_id - 1) % len(colors)]


def create_placeholder(width: int, height: int, text: str) -> np.ndarray:
    """Create a placeholder image when a stream is offline."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (max(20, width // 10), height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def draw_epipolar_lines(
    frame1: np.ndarray,
    frame2: np.ndarray,
    matches: List[StereoMatch],
    camera_model: StereoCameraModel
) -> None:
    """Draw epipolar line overlays for matched detections."""
    w1, w2 = frame1.shape[1], frame2.shape[1]
    for m in matches:
        color = get_track_color(m.track_id)

        # Epipolar line in Cam2 from Cam1 centroid
        l2 = camera_model.compute_epipolar_line_in_cam2(m.det1.centroid)
        a, b, c = l2[0], l2[1], l2[2]
        if abs(b) > 1e-6:
            y0 = int(-c / b)
            yw = int(-(a * w2 + c) / b)
            cv2.line(frame2, (0, y0), (w2, yw), color, 1, cv2.LINE_AA)

        # Epipolar line in Cam1 from Cam2 centroid
        l1 = camera_model.compute_epipolar_line_in_cam1(m.det2.centroid)
        a1, b1, c1 = l1[0], l1[1], l1[2]
        if abs(b1) > 1e-6:
            y0_1 = int(-c1 / b1)
            yw_1 = int(-(a1 * w1 + c1) / b1)
            cv2.line(frame1, (0, y0_1), (w1, yw_1), color, 1, cv2.LINE_AA)


def draw_3d_detections(
    frame1: np.ndarray,
    frame2: np.ndarray,
    matches: List[StereoMatch]
) -> None:
    """Render bounding boxes, track IDs, and 3D coordinate badges on both camera frames."""
    for m in matches:
        color = get_track_color(m.track_id)
        x, y, z = m.point_3d[0], m.point_3d[1], m.point_3d[2]
        dist = m.distance_m

        # Draw Camera 1 Box & Label
        x1_1, y1_1, x2_1, y2_1 = m.det1.bbox
        cv2.rectangle(frame1, (x1_1, y1_1), (x2_1, y2_1), color, 2)
        cv2.circle(frame1, (int(m.det1.centroid[0]), int(m.det1.centroid[1])), 4, (0, 0, 255), -1)

        label_title = f"[ID:{m.track_id}] {m.class_name.capitalize()}"
        label_coord = f"3D: X={x:+.2f} Y={y:+.2f} Z={z:.2f}m (D:{dist:.2f}m)"

        # Label background pill
        (tw1, th1), _ = cv2.getTextSize(label_title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (tw2, th2), _ = cv2.getTextSize(label_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        max_tw = max(tw1, tw2)

        box_top = max(0, y1_1 - 36)
        cv2.rectangle(frame1, (x1_1, box_top), (x1_1 + max_tw + 10, y1_1), (30, 30, 30), -1)
        cv2.rectangle(frame1, (x1_1, box_top), (x1_1 + max_tw + 10, y1_1), color, 1)
        cv2.putText(frame1, label_title, (x1_1 + 5, box_top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame1, label_coord, (x1_1 + 5, box_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw Camera 2 Box & Label
        x1_2, y1_2, x2_2, y2_2 = m.det2.bbox
        cv2.rectangle(frame2, (x1_2, y1_2), (x2_2, y2_2), color, 2)
        cv2.circle(frame2, (int(m.det2.centroid[0]), int(m.det2.centroid[1])), 4, (0, 0, 255), -1)

        box_top2 = max(0, y1_2 - 36)
        cv2.rectangle(frame2, (x1_2, box_top2), (x1_2 + max_tw + 10, y1_2), (30, 30, 30), -1)
        cv2.rectangle(frame2, (x1_2, box_top2), (x1_2 + max_tw + 10, y1_2), color, 1)
        cv2.putText(frame2, label_title, (x1_2 + 5, box_top2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame2, label_coord, (x1_2 + 5, box_top2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


def render_bev_map(
    matches: List[StereoMatch],
    width: int = 320,
    height: int = 480,
    max_range_m: float = 6.0,
    camera_baseline_m: float = 0.35,
) -> np.ndarray:
    """
    Render a top-down Bird's-Eye View (BEV) 3D coordinate mini-map (X vs Z plane).
    """
    bev = np.full((height, width, 3), 24, dtype=np.uint8)

    # Origin (Camera 1) placed near bottom-center
    origin_x = width // 2
    origin_y = height - 40
    scale = (height - 80) / max_range_m  # pixels per meter

    # Draw range distance rings
    for r in np.arange(1.0, max_range_m + 0.1, 1.0):
        radius_px = int(r * scale)
        cv2.circle(bev, (origin_x, origin_y), radius_px, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(bev, f"{r:.0f}m", (origin_x + 4, origin_y - radius_px + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # Draw FOV cone lines (~65 deg)
    cone_len = height - 60
    angle = np.deg2rad(32.5)
    dx = int(cone_len * np.sin(angle))
    cv2.line(bev, (origin_x, origin_y), (origin_x - dx, origin_y - cone_len), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(bev, (origin_x, origin_y), (origin_x + dx, origin_y - cone_len), (60, 60, 60), 1, cv2.LINE_AA)

    # Draw Camera Baselines
    cam2_x_px = int(origin_x + (camera_baseline_m * scale))
    cv2.line(bev, (origin_x, origin_y), (cam2_x_px, origin_y), (0, 200, 255), 2)
    cv2.circle(bev, (origin_x, origin_y), 6, (0, 255, 0), -1)      # Cam 1
    cv2.circle(bev, (cam2_x_px, origin_y), 6, (255, 128, 0), -1)   # Cam 2
    cv2.putText(bev, "C1", (origin_x - 18, origin_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(bev, "C2", (cam2_x_px + 5, origin_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)

    # Header
    cv2.putText(bev, "BIRD'S-EYE VIEW (3D)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bev, "Top-Down (X-Z)", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # Plot 3D targets
    for m in matches:
        x, z = m.point_3d[0], m.point_3d[2]
        px = int(origin_x + (x * scale))
        py = int(origin_y - (z * scale))
        color = get_track_color(m.track_id)

        if 0 <= px < width and 0 <= py < height:
            # Ray line from Camera 1 to target
            cv2.line(bev, (origin_x, origin_y), (px, py), (color[0] // 3, color[1] // 3, color[2] // 3), 1, cv2.LINE_AA)
            # Target circle & halo
            cv2.circle(bev, (px, py), 8, color, -1)
            cv2.circle(bev, (px, py), 12, color, 1, cv2.LINE_AA)
            # Label
            tag = f"ID:{m.track_id} ({x:+.1f},{z:.1f}m)"
            cv2.putText(bev, tag, (px + 10, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    return bev


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time-Aligned Dual RTSP 3D Object Detection & Triangulation System"
    )
    # Stream URLs
    parser.add_argument("--url1", "--url", dest="url1", type=str, default="rtsp://127.0.0.1:8554/live", help="RTSP Camera 1 URL")
    parser.add_argument("--url2", type=str, default="rtsp://127.0.0.1:8554/live2", help="RTSP Camera 2 URL")

    # Geometry & Calibration
    parser.add_argument("--calib", type=str, default=None, help="Path to stereo calibration JSON/YAML file")
    parser.add_argument("--baseline", type=float, default=0.35, help="Stereo baseline distance in meters (default: 0.35)")
    parser.add_argument("--fov", type=float, default=65.0, help="Camera horizontal FOV in degrees (default: 65.0)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Convergence yaw angle in degrees (default: 0.0)")
    parser.add_argument("--save-calib", type=str, default=None, help="Save synthetic or active calibration to JSON file")

    # Synchronization & Performance
    parser.add_argument("--max-sync-delta-ms", type=float, default=40.0, help="Max allowed time discrepancy between frames in ms")
    parser.add_argument("--buffer-size", type=int, default=60, help="Ring buffer frame queue depth")

    # Detection & Matching
    parser.add_argument(
        "--detector",
        type=str,
        default="face",
        choices=["face", "fullbody", "upperbody", "red", "green", "blue", "color", "dnn", "none"],
        help="Object detector backend (default: face)"
    )
    parser.add_argument("--dnn-model", type=str, default=None, help="Path to ONNX model when --detector dnn is selected")
    parser.add_argument("--max-epi-px", type=float, default=45.0, help="Max epipolar pixel threshold for matching")

    # Display & Visuals
    parser.add_argument("--width", type=int, default=None, help="Resize individual stream width (e.g. 640)")
    parser.add_argument("--height", type=int, default=None, help="Resize individual stream height (e.g. 480)")
    parser.add_argument("--title1", type=str, default="Camera 1 (Left)", help="Overlay title for Camera 1")
    parser.add_argument("--title2", type=str, default="Camera 2 (Right)", help="Overlay title for Camera 2")
    parser.add_argument("--bev", action="store_true", default=True, help="Enable Bird's-Eye View (BEV) 3D mini-map")
    parser.add_argument("--no-bev", dest="bev", action="store_false", help="Disable Bird's-Eye View (BEV) 3D mini-map")
    parser.add_argument("--epipolar", action="store_true", default=False, help="Enable epipolar lines by default")
    parser.add_argument("--export-json", type=str, default=None, help="Path to write live 3D telemetry stream in JSON Lines format")

    args = parser.parse_args()

    # Initialize Stereo Camera Model
    target_dims = (args.width or 640, args.height or 480)
    if args.calib and os.path.exists(args.calib):
        print(f"Loading stereo calibration from: {args.calib}")
        camera_model = StereoCameraModel.load(args.calib)
    else:
        print(f"Initializing synthetic stereo camera model (Baseline: {args.baseline}m, FOV: {args.fov}°, Conv. Yaw: {args.yaw}°)")
        camera_model = StereoCameraModel.generate_synthetic(
            image_size=target_dims,
            fov_deg=args.fov,
            baseline_m=args.baseline,
            yaw_deg=args.yaw,
        )
        if args.save_calib:
            camera_model.save(args.save_calib)
            print(f"Saved initial calibration model to: {args.save_calib}")

    # Initialize Object Detector & Matcher
    detector: Optional[BaseDetector] = None
    if args.detector != "none":
        print(f"Initializing Object Detector: [{args.detector}]")
        detector_kwargs = {}
        if args.detector == "dnn" and args.dnn_model:
            detector_kwargs["model_path"] = args.dnn_model
        detector = create_detector(args.detector, **detector_kwargs)

    matcher = StereoObjectMatcher(camera_model=camera_model, max_epipolar_dist_px=args.max_epi_px)
    smoother = TrackSmoother3D(alpha=0.35, max_match_dist_m=0.8)

    # Initialize Stream Synchronizer & Readers
    synchronizer = StreamSynchronizer(
        max_buffer_size=args.buffer_size,
        max_delta_ms=args.max_sync_delta_ms
    )

    print(f"\nConnecting Stream 1 -> {args.url1}")
    stream1 = RTSPStreamReader(args.url1, stream_id=1, synchronizer=synchronizer, name="Stream 1")

    print(f"Connecting Stream 2 -> {args.url2}")
    stream2 = RTSPStreamReader(args.url2, stream_id=2, synchronizer=synchronizer, name="Stream 2")

    if not stream1.running and not stream2.running:
        print(f"\n[Warning] Neither RTSP stream could be reached immediately.\n  Stream 1: {args.url1}\n  Stream 2: {args.url2}\nRunning viewer with reconnect/placeholder mode.")

    stream1.start()
    stream2.start()

    window_name = "Time-Aligned Dual RTSP 3D Object Detection & Triangulation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Optional JSON Telemetry File
    telemetry_file = open(args.export_json, "a", encoding="utf-8") if args.export_json else None

    # Performance & State Variables
    prev_time = time.time()
    fps = 0.0
    show_epipolar = args.epipolar
    show_bev = args.bev
    detect_enabled = (detector is not None)
    is_fullscreen = False

    print("\nInteractive Keyboard Controls:")
    print("  'q' or ESC : Quit")
    print("  'd'        : Toggle Object Detection & 3D Triangulation")
    print("  'e'        : Toggle Epipolar Lines Overlay")
    print("  'm'        : Toggle Bird's-Eye View (BEV) 3D Mini-Map")
    print("  's'        : Save Snapshot (Images + 3D Coordinates JSON)")
    print("  'r'        : Reset 3D Tracks & Synchronization Queues")
    print("  'f'        : Toggle Fullscreen\n")

    try:
        while stream1.running or stream2.running:
            # 1. Fetch time-aligned pair
            synced, tf1, tf2 = synchronizer.get_aligned_pair(max_delta_ms=args.max_sync_delta_ms)

            if synced and tf1 is not None and tf2 is not None:
                frame1 = tf1.frame.copy()
                frame2 = tf2.frame.copy()
                ret1, ret2 = True, True
            else:
                # Fallback to latest available frame if queues are priming
                ret1, frame1 = stream1.get_latest_fallback()
                ret2, frame2 = stream2.get_latest_fallback()
                time.sleep(0.005)

            # Determine frame dimensions
            default_w, default_h = 640, 480
            if ret1 and frame1 is not None:
                h1, w1 = frame1.shape[:2]
            elif ret2 and frame2 is not None:
                h1, w1 = frame2.shape[:2]
            else:
                h1, w1 = default_h, default_w

            target_h = args.height if args.height else h1
            target_w = args.width if args.width else w1

            # Prepare frame 1
            if ret1 and frame1 is not None:
                if frame1.shape[:2] != (target_h, target_w):
                    frame1 = cv2.resize(frame1, (target_w, target_h))
            else:
                frame1 = create_placeholder(target_w, target_h, f"{args.title1}: Waiting for Stream...")

            # Prepare frame 2
            if ret2 and frame2 is not None:
                if frame2.shape[:2] != (target_h, target_w):
                    frame2 = cv2.resize(frame2, (target_w, target_h))
            else:
                frame2 = create_placeholder(target_w, target_h, f"{args.title2}: Waiting for Stream...")

            # 2. Perception & 3D Localization Pipeline
            matches: List[StereoMatch] = []
            if detect_enabled and detector is not None and ret1 and ret2:
                dets1 = detector.detect(frame1)
                dets2 = detector.detect(frame2)
                matches = matcher.match_and_triangulate(dets1, dets2, smoother=smoother)

                # Draw 3D Overlays
                draw_3d_detections(frame1, frame2, matches)

                if show_epipolar:
                    draw_epipolar_lines(frame1, frame2, matches, camera_model)

                # Log / Export continuous 3D telemetry
                if telemetry_file and matches:
                    telemetry_payload = {
                        "timestamp": time.time(),
                        "sync_delta_ms": synchronizer.last_sync_delta_ms,
                        "targets": [
                            {
                                "track_id": m.track_id,
                                "class": m.class_name,
                                "confidence": round(m.confidence, 3),
                                "point_3d_m": [round(float(v), 3) for v in m.point_3d],
                                "distance_m": round(float(m.distance_m), 3),
                                "bbox_cam1": list(m.det1.bbox),
                                "bbox_cam2": list(m.det2.bbox),
                            }
                            for m in matches
                        ]
                    }
                    telemetry_file.write(json.dumps(telemetry_payload) + "\n")
                    telemetry_file.flush()

            # 3. HUD Titles & Sync Diagnostics
            cv2.putText(frame1, args.title1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame2, args.title2, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Combine Side-by-Side
            combined_display = cv2.hconcat([frame1, frame2])

            # Optional BEV Side Panel
            if show_bev:
                bev_map = render_bev_map(
                    matches,
                    width=320,
                    height=combined_display.shape[0],
                    camera_baseline_m=float(camera_model.T[0, 0])
                )
                combined_display = cv2.hconcat([combined_display, bev_map])

            # Calculate FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # Bottom Status Bar HUD
            sync_stats = synchronizer.get_stats()
            status_text = (
                f"FPS: {fps:.1f} | Δt: {sync_stats['last_delta_ms']:.1f}ms "
                f"(Avg: {sync_stats['avg_delta_ms']:.1f}ms ±{sync_stats['jitter_ms']:.1f}) | "
                f"Synced: {sync_stats['synced_pairs']} | 3D Targets: {len(matches)}"
            )
            cv2.rectangle(combined_display, (0, combined_display.shape[0] - 30), (combined_display.shape[1], combined_display.shape[0]), (20, 20, 20), -1)
            cv2.putText(
                combined_display,
                status_text,
                (15, combined_display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0) if sync_stats['last_delta_ms'] < args.max_sync_delta_ms else (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, combined_display)

            # 4. Handle Key Controls
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("d"):
                detect_enabled = not detect_enabled
                print(f"[Controls] Detection & 3D Triangulation: {'ON' if detect_enabled else 'OFF'}")
            elif key == ord("e"):
                show_epipolar = not show_epipolar
                print(f"[Controls] Epipolar Lines: {'ON' if show_epipolar else 'OFF'}")
            elif key == ord("m"):
                show_bev = not show_bev
                print(f"[Controls] Bird's-Eye View Mini-Map: {'ON' if show_bev else 'OFF'}")
            elif key == ord("r"):
                synchronizer.clear()
                smoother = TrackSmoother3D(alpha=0.35, max_match_dist_m=0.8)
                print("[Controls] Reset synchronizer queues and 3D object tracks.")
            elif key == ord("s"):
                ts = int(time.time())
                cv2.imwrite(f"snapshot_combined_{ts}.jpg", combined_display)
                cv2.imwrite(f"snapshot_cam1_{ts}.jpg", frame1)
                cv2.imwrite(f"snapshot_cam2_{ts}.jpg", frame2)

                # Save 3D telemetry metadata
                meta = {
                    "timestamp": ts,
                    "sync_delta_ms": synchronizer.last_sync_delta_ms,
                    "baseline_m": float(camera_model.T[0, 0]),
                    "detections_3d": [
                        {
                            "track_id": m.track_id,
                            "class": m.class_name,
                            "point_3d_m": [float(v) for v in m.point_3d],
                            "distance_m": float(m.distance_m),
                            "epipolar_error_px": float(m.epipolar_error_px),
                            "bbox_cam1": list(m.det1.bbox),
                            "bbox_cam2": list(m.det2.bbox),
                        }
                        for m in matches
                    ]
                }
                with open(f"snapshot_3d_telemetry_{ts}.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                print(f"[Snapshot saved] -> snapshot_combined_{ts}.jpg and snapshot_3d_telemetry_{ts}.json")
            elif key == ord("c"):
                calib_export_path = f"stereo_calib_{int(time.time())}.json"
                camera_model.save(calib_export_path)
                print(f"[Calibration saved] -> {calib_export_path}")
            elif key == ord("f"):
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if telemetry_file:
            telemetry_file.close()
        stream1.stop()
        stream2.stop()
        cv2.destroyAllWindows()
        print("Streams stopped and resources released cleanly.")


if __name__ == "__main__":
    main()
