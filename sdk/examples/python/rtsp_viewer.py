#!/usr/bin/env python3
"""
Dual RTSP Stream Viewer with Low-Latency Threaded Capture (Side-by-Side)
"""

import argparse
import os
import sys
import threading
import time
import cv2
import numpy as np

# Set FFmpeg options for low latency before opening VideoCapture
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


class RTSPStreamReader:
    """Threaded RTSP reader that continuously grabs the newest frame."""

    def __init__(self, src: str, name: str = "Stream"):
        self.src = src
        self.name = name
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.running = self.grabbed
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if not self.running:
            return self
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if not grabbed:
                print(f"[{self.name}] Stream disconnected or ended.")
                self.running = False
                break

            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()


def create_placeholder(width: int, height: int, text: str) -> np.ndarray:
    """Create a placeholder image when a stream is offline or disconnected."""
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


def main():
    parser = argparse.ArgumentParser(description="Ingest and display two RTSP streams side-by-side.")
    parser.add_argument(
        "--url1",
        "--url",
        dest="url1",
        type=str,
        default="rtsp://127.0.0.1:8554/live",
        help="First RTSP Stream URL (default: rtsp://127.0.0.1:8554/live)",
    )
    parser.add_argument(
        "--url2",
        type=str,
        default="rtsp://127.0.0.1:8554/live2",
        help="Second RTSP Stream URL (default: rtsp://127.0.0.1:8554/live2)",
    )
    parser.add_argument("--width", type=int, default=None, help="Resize individual stream width (e.g. 640)")
    parser.add_argument("--height", type=int, default=None, help="Resize individual stream height (e.g. 480)")
    parser.add_argument("--title1", type=str, default="Stream 1", help="Overlay title for Stream 1")
    parser.add_argument("--title2", type=str, default="Stream 2", help="Overlay title for Stream 2")
    args = parser.parse_args()

    print(f"Connecting Stream 1 to: {args.url1}")
    stream1 = RTSPStreamReader(args.url1, name="Stream 1")

    print(f"Connecting Stream 2 to: {args.url2}")
    stream2 = RTSPStreamReader(args.url2, name="Stream 2")

    if not stream1.running and not stream2.running:
        print(f"Error: Unable to connect to either RTSP stream:\n  1: {args.url1}\n  2: {args.url2}")
        sys.exit(1)

    if not stream1.running:
        print(f"Warning: Stream 1 could not be opened ({args.url1}). Displaying placeholder.")
    else:
        stream1.start()

    if not stream2.running:
        print(f"Warning: Stream 2 could not be opened ({args.url2}). Displaying placeholder.")
    else:
        stream2.start()

    window_name = "Dual RTSP Live Stream (Side-by-Side)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Performance tracking
    prev_time = time.time()
    fps = 0.0

    print("\nControls:")
    print("  'q' or ESC : Quit")
    print("  's'        : Save snapshots (combined and individual)")
    print("  'f'        : Toggle Fullscreen\n")

    is_fullscreen = False

    try:
        while stream1.running or stream2.running:
            ret1, frame1 = stream1.read()
            ret2, frame2 = stream2.read()

            if not ret1 and not ret2 and not stream1.running and not stream2.running:
                print("Both streams have ended.")
                break

            # Determine baseline dimensions for normalization
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
                frame1 = create_placeholder(target_w, target_h, f"{args.title1}: Offline")

            # Prepare frame 2
            if ret2 and frame2 is not None:
                if frame2.shape[:2] != (target_h, target_w):
                    frame2 = cv2.resize(frame2, (target_w, target_h))
            else:
                frame2 = create_placeholder(target_w, target_h, f"{args.title2}: Offline")

            # Draw individual titles
            cv2.putText(
                frame1,
                args.title1,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame2,
                args.title2,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Combine side-by-side horizontally
            combined_frame = cv2.hconcat([frame1, frame2])

            # Calculate FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # Draw FPS overlay
            cv2.putText(
                combined_frame,
                f"FPS: {fps:.1f}",
                (15, combined_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, combined_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("s"):
                ts = int(time.time())
                cv2.imwrite(f"snapshot_combined_{ts}.jpg", combined_frame)
                if ret1 and frame1 is not None:
                    cv2.imwrite(f"snapshot_stream1_{ts}.jpg", frame1)
                if ret2 and frame2 is not None:
                    cv2.imwrite(f"snapshot_stream2_{ts}.jpg", frame2)
                print(f"[Snapshot saved] -> snapshot_combined_{ts}.jpg (plus individual stream snapshots)")
            elif key == ord("f"):
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        stream1.stop()
        stream2.stop()
        cv2.destroyAllWindows()
        print("Streams stopped and resources released.")


if __name__ == "__main__":
    main()
