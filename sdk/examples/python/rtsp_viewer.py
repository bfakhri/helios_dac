#!/usr/bin/env python3
"""
RTSP Stream Viewer with Low-Latency Threaded Capture
"""

import argparse
import os
import sys
import threading
import time
import cv2

# Set FFmpeg options for low latency before opening VideoCapture
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


class RTSPStreamReader:
    """Threaded RTSP reader that continuously grabs the newest frame."""

    def __init__(self, src: str):
        self.src = src
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
                print("Stream disconnected or ended.")
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


def main():
    parser = argparse.ArgumentParser(description="Ingest and display an RTSP stream.")
    parser.add_argument(
        "--url",
        type=str,
        default="rtsp://127.0.0.1:8554/live",
        help="RTSP Stream URL (default: rtsp://127.0.0.1:8554/live)",
    )
    parser.add_argument("--width", type=int, default=None, help="Resize display width")
    parser.add_argument("--height", type=int, default=None, help="Resize display height")
    args = parser.parse_args()

    print(f"Connecting to: {args.url}")
    stream = RTSPStreamReader(args.url)

    if not stream.running:
        print(f"Error: Unable to connect to RTSP stream: {args.url}")
        sys.exit(1)

    stream.start()

    window_name = "RTSP Live Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Performance tracking
    prev_time = time.time()
    fps = 0.0

    print("\nControls:")
    print("  'q' or ESC : Quit")
    print("  's'        : Save snapshot")
    print("  'f'        : Toggle Fullscreen\n")

    is_fullscreen = False

    try:
        while stream.running:
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Optional resizing
            if args.width and args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            # Calculate FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # Draw FPS on overlay
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("s"):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[Snapshot saved] -> {filename}")
            elif key == ord("f"):
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("Stream stopped and resources released.")


if __name__ == "__main__":
    main()

