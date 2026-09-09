#!/usr/bin/env python3
"""
End-to-end integration test for the Stereo 3D detection pipeline.
Generates dual-camera frames containing colored moving target spheres at known 3D positions,
pushes them through the StreamSynchronizer, runs ColorBlobDetector, StereoObjectMatcher,
and checks that the resulting 3D coordinates match ground-truth positions.
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from object_detector import ColorBlobDetector
from stereo_geometry import StereoCameraModel
from stereo_matcher import StereoObjectMatcher, TrackSmoother3D
from stream_synchronizer import StreamSynchronizer


def render_scene_frame(
    camera_proj: np.ndarray,
    sphere_3d_m: np.ndarray,
    sphere_radius_m: float = 0.10,
    width: int = 640,
    height: int = 480,
    color_bgr: tuple = (0, 0, 255)
) -> np.ndarray:
    """Render a synthetic scene frame with a colored 3D sphere projected onto camera image plane."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Project center
    p_hom = camera_proj @ np.append(sphere_3d_m, 1.0)
    if p_hom[2] <= 0:
        return img

    cx = int(p_hom[0] / p_hom[2])
    cy = int(p_hom[1] / p_hom[2])

    # Project radius (approx: r_px = f * (r_m / Z))
    # Using fx from proj matrix
    fx = camera_proj[0, 0]
    r_px = max(5, int(fx * (sphere_radius_m / sphere_3d_m[2])))

    if 0 <= cx < width and 0 <= cy < height:
        cv2.circle(img, (cx, cy), r_px, color_bgr, -1)

    return img


def test_end_to_end_stereo_pipeline():
    # 1. Setup Camera Geometry (Baseline = 0.30m)
    baseline_m = 0.30
    camera_model = StereoCameraModel.generate_synthetic(
        image_size=(640, 480),
        fov_deg=65.0,
        baseline_m=baseline_m
    )

    # 2. Setup Modules
    sync = StreamSynchronizer(max_buffer_size=30, max_delta_ms=30.0)
    detector = ColorBlobDetector(color="red", min_area=50)
    matcher = StereoObjectMatcher(camera_model=camera_model, max_epipolar_dist_px=20.0)
    smoother = TrackSmoother3D(alpha=0.5, max_match_dist_m=0.5)

    # 3. Known 3D Target: moving in 3D space
    # (X = 0.15m, Y = -0.05m, Z = 2.20m)
    true_target_3d = np.array([0.15, -0.05, 2.20], dtype=np.float64)

    t0 = time.perf_counter_ns()

    # Simulate 5 frames
    for f_idx in range(5):
        t_frame = t0 + f_idx * 33_333_333
        # Slight jitter in arrival time (3ms)
        t_cam1 = t_frame
        t_cam2 = t_frame + 3_000_000

        img1 = render_scene_frame(camera_model.P1, true_target_3d, sphere_radius_m=0.12)
        img2 = render_scene_frame(camera_model.P2, true_target_3d, sphere_radius_m=0.12)

        sync.push_frame(1, img1, timestamp_ns=t_cam1)
        sync.push_frame(2, img2, timestamp_ns=t_cam2)

        # Pull aligned pair
        ok, tf1, tf2 = sync.get_aligned_pair()
        assert ok, f"Frame {f_idx} was not synchronized!"

        # Detect
        dets1 = detector.detect(tf1.frame)
        dets2 = detector.detect(tf2.frame)

        assert len(dets1) == 1, f"Expected 1 detection in Cam 1, got {len(dets1)}"
        assert len(dets2) == 1, f"Expected 1 detection in Cam 2, got {len(dets2)}"

        # Match & Triangulate
        matches = matcher.match_and_triangulate(dets1, dets2, smoother=smoother)
        assert len(matches) == 1, f"Expected 1 matched 3D target, got {len(matches)}"

        m = matches[0]
        err_3d = np.linalg.norm(m.point_3d - true_target_3d)
        print(f"Frame {f_idx}: Triangulated 3D = ({m.point_3d[0]:.3f}, {m.point_3d[1]:.3f}, {m.point_3d[2]:.3f})m | True = ({true_target_3d[0]:.3f}, {true_target_3d[1]:.3f}, {true_target_3d[2]:.3f})m | Error = {err_3d:.4f}m")

        # Reconstructed 3D error should be very small (within a couple centimeters due to discrete pixel quantization of sphere circle)
        assert err_3d < 0.05, f"3D error too high: {err_3d}m"

    print("✓ test_end_to_end_stereo_pipeline passed successfully!")


if __name__ == "__main__":
    test_end_to_end_stereo_pipeline()
