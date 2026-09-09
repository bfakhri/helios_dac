#!/usr/bin/env python3
"""
Unit test for StereoCameraModel & 3D Triangulation.
Projects synthetic 3D points to 2D camera coordinates, tests triangulation recovery,
epipolar error metrics, and calibration save/load.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stereo_geometry import StereoCameraModel


def test_synthetic_triangulation():
    # Construct synthetic camera model (baseline = 0.40m, FOV = 60 deg)
    model = StereoCameraModel.generate_synthetic(
        image_size=(640, 480),
        fov_deg=60.0,
        baseline_m=0.40,
        yaw_deg=5.0,     # 5 deg inward angle
        pitch_deg=0.0
    )

    # Define test 3D points in Camera 1 frame (X, Y, Z) in meters
    ground_truth_3d = np.array([
        [0.0, 0.0, 2.0],       # Center at 2m depth
        [-0.5, -0.2, 3.5],     # Left, slightly above at 3.5m
        [0.8, 0.4, 1.8],       # Right, slightly down at 1.8m
        [-0.2, 0.5, 4.0],      # Far point at 4.0m
    ], dtype=np.float64)

    # Project to Camera 1: p1_hom = K1 * [I | 0] * P_3d
    pts_hom1 = model.P1 @ np.hstack([ground_truth_3d, np.ones((len(ground_truth_3d), 1))]).T
    pts2d_1 = (pts_hom1[:2, :] / pts_hom1[2, :]).T

    # Project to Camera 2: p2_hom = K2 * [R | T] * P_3d
    pts_hom2 = model.P2 @ np.hstack([ground_truth_3d, np.ones((len(ground_truth_3d), 1))]).T
    pts2d_2 = (pts_hom2[:2, :] / pts_hom2[2, :]).T

    # Triangulate back to 3D
    reconstructed_3d = model.triangulate_points(pts2d_1, pts2d_2, undistort=False)

    diff = np.abs(reconstructed_3d - ground_truth_3d)
    max_err = float(np.max(diff))
    print(f"Max triangulation reconstruction error: {max_err:.8f} meters")
    assert max_err < 1e-4, f"Triangulation error too high: {max_err}"

    # Verify epipolar line alignment error on corresponding points
    for i in range(len(pts2d_1)):
        p1 = (pts2d_1[i, 0], pts2d_1[i, 1])
        p2 = (pts2d_2[i, 0], pts2d_2[i, 1])
        epi_err = model.epipolar_error(p1, p2)
        assert epi_err < 1e-3, f"Epipolar error too high for point {i}: {epi_err}"

    print("✓ test_synthetic_triangulation passed.")


def test_save_load_calibration(tmp_path="/tmp"):
    model = StereoCameraModel.generate_synthetic(baseline_m=0.25)
    test_file = os.path.join(tmp_path, "test_calib.json")

    model.save(test_file)
    loaded_model = StereoCameraModel.load(test_file)

    assert np.allclose(model.K1, loaded_model.K1)
    assert np.allclose(model.K2, loaded_model.K2)
    assert np.allclose(model.R, loaded_model.R)
    assert np.allclose(model.T, loaded_model.T)
    assert model.image_size == loaded_model.image_size

    if os.path.exists(test_file):
        os.remove(test_file)
    print("✓ test_save_load_calibration passed.")


if __name__ == "__main__":
    test_synthetic_triangulation()
    test_save_load_calibration()
    print("All StereoGeometry tests passed successfully!")
