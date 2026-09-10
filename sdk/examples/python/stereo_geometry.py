#!/usr/bin/env python3
"""
Stereo Geometry & Camera Calibration Module.
Handles camera intrinsics, extrinsics, epipolar geometry derivation, and 3D triangulation.
"""

import json
import os
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class StereoCameraModel:
    """
    Represents calibrated stereo camera geometry.
    Computes projection matrices, Fundamental matrix, epipolar lines, and 3D triangulation.
    """

    def __init__(
        self,
        K1: np.ndarray,
        D1: np.ndarray,
        K2: np.ndarray,
        D2: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        image_size: Tuple[int, int] = (640, 480),
    ):
        """
        Args:
            K1: 3x3 Intrinsic matrix for Camera 1
            D1: Distortion coefficients for Camera 1 (1x5 or 1x8)
            K2: 3x3 Intrinsic matrix for Camera 2
            D2: Distortion coefficients for Camera 2 (1x5 or 1x8)
            R: 3x3 Rotation matrix transforming Camera 1 coordinates to Camera 2
            T: 3x1 Translation vector (in meters) from Camera 1 to Camera 2
            image_size: (width, height)
        """
        self.K1 = np.array(K1, dtype=np.float64).reshape((3, 3))
        self.D1 = np.array(D1, dtype=np.float64).reshape((-1, 1))
        self.K2 = np.array(K2, dtype=np.float64).reshape((3, 3))
        self.D2 = np.array(D2, dtype=np.float64).reshape((-1, 1))
        self.R = np.array(R, dtype=np.float64).reshape((3, 3))
        self.T = np.array(T, dtype=np.float64).reshape((3, 1))
        self.image_size = image_size

        # Projection matrices: P1 = K1 * [I | 0], P2 = K2 * [R | T]
        self.P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2 = self.K2 @ np.hstack([self.R, self.T])

        # Essential & Fundamental Matrices
        # [T]_x skew-symmetric matrix
        tx, ty, tz = float(self.T[0, 0]), float(self.T[1, 0]), float(self.T[2, 0])
        self.T_skew = np.array([
            [0.0, -tz, ty],
            [tz, 0.0, -tx],
            [-ty, tx, 0.0]
        ], dtype=np.float64)

        self.E = self.T_skew @ self.R
        self.F = np.linalg.inv(self.K2).T @ self.E @ np.linalg.inv(self.K1)

    @classmethod
    def generate_synthetic(
        cls,
        image_size: Tuple[int, int] = (640, 480),
        fov_deg: float = 65.0,
        baseline_m: float = 0.30,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
    ) -> "StereoCameraModel":
        """
        Generate an estimated pinhole camera model based on assumed field-of-view and baseline.
        Useful for uncalibrated instant prototyping.
        """
        w, h = image_size
        fov_rad = np.deg2rad(fov_deg)
        fx = (w / 2.0) / np.tan(fov_rad / 2.0)
        fy = fx
        cx = w / 2.0
        cy = h / 2.0

        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        D = np.zeros((5, 1), dtype=np.float64)

        # Rotation
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        R = Ry @ Rx

        # Translation: camera 2 is shifted +baseline along X axis
        T = np.array([[baseline_m], [0.0], [0.0]], dtype=np.float64)

        return cls(K1=K, D1=D, K2=K, D2=D, R=R, T=T, image_size=image_size)

    def triangulate_points(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        undistort: bool = True
    ) -> np.ndarray:
        """
        Triangulate corresponding 2D points from Camera 1 and Camera 2 into 3D world coordinates.

        Args:
            pts1: (N, 2) array of pixel coordinates in Camera 1
            pts2: (N, 2) array of pixel coordinates in Camera 2
            undistort: Whether to undistort pixel coordinates prior to triangulation

        Returns:
            (N, 3) 3D coordinates in Camera 1 reference frame (meters).
        """
        pts1 = np.asarray(pts1, dtype=np.float64).reshape((-1, 2))
        pts2 = np.asarray(pts2, dtype=np.float64).reshape((-1, 2))

        if len(pts1) == 0 or len(pts2) == 0:
            return np.empty((0, 3), dtype=np.float64)

        if undistort:
            # Undistort points (using identity camera matrix for normalized coordinates)
            pts1_norm = cv2.undistortPoints(
                pts1.reshape(-1, 1, 2), self.K1, self.D1, P=None
            ).reshape(-1, 2)
            pts2_norm = cv2.undistortPoints(
                pts2.reshape(-1, 1, 2), self.K2, self.D2, P=None
            ).reshape(-1, 2)

            # Normalized projection matrices: [I | 0] and [R | T]
            P1_norm = np.hstack([np.eye(3), np.zeros((3, 1))])
            P2_norm = np.hstack([self.R, self.T])

            # cv2.triangulatePoints expects 2xN arrays
            points_4d_hom = cv2.triangulatePoints(
                P1_norm, P2_norm, pts1_norm.T, pts2_norm.T
            )
        else:
            points_4d_hom = cv2.triangulatePoints(
                self.P1, self.P2, pts1.T, pts2.T
            )

        # Convert from homogeneous coordinates (4, N) -> (N, 3)
        w = points_4d_hom[3, :]
        w = np.where(np.abs(w) < 1e-9, 1e-9, w)
        points_3d = (points_4d_hom[:3, :] / w).T

        return points_3d

    def compute_epipolar_line_in_cam2(self, pt1: Tuple[float, float]) -> np.ndarray:
        """
        Given a point (u, v) in Camera 1, returns the line coefficients [a, b, c] in Camera 2 (ax + by + c = 0).
        """
        p1 = np.array([pt1[0], pt1[1], 1.0], dtype=np.float64)
        l2 = self.F @ p1
        # Normalize
        norm = np.hypot(l2[0], l2[1])
        if norm > 1e-9:
            l2 /= norm
        return l2

    def compute_epipolar_line_in_cam1(self, pt2: Tuple[float, float]) -> np.ndarray:
        """
        Given a point (u, v) in Camera 2, returns the line coefficients [a, b, c] in Camera 1.
        """
        p2 = np.array([pt2[0], pt2[1], 1.0], dtype=np.float64)
        l1 = self.F.T @ p2
        norm = np.hypot(l1[0], l1[1])
        if norm > 1e-9:
            l1 /= norm
        return l1

    def point_to_line_distance(self, pt: Tuple[float, float], line: np.ndarray) -> float:
        """Compute perpendicular distance from 2D point (x, y) to line ax + by + c = 0."""
        a, b, c = line[0], line[1], line[2]
        denom = np.hypot(a, b)
        if denom < 1e-9:
            return float("inf")
        return float(abs(a * pt[0] + b * pt[1] + c) / denom)

    def epipolar_error(self, pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """
        Symmetric epipolar distance error between point in Cam1 and point in Cam2.
        """
        l2 = self.compute_epipolar_line_in_cam2(pt1)
        d2 = self.point_to_line_distance(pt2, l2)

        l1 = self.compute_epipolar_line_in_cam1(pt2)
        d1 = self.point_to_line_distance(pt1, l1)

        return (d1 + d2) / 2.0

    def rescale(self, new_image_size: Tuple[int, int]) -> "StereoCameraModel":
        """
        Rescale camera intrinsic parameters and projection matrices to match a new image resolution.
        """
        orig_w, orig_h = self.image_size
        new_w, new_h = new_image_size

        if orig_w == new_w and orig_h == new_h:
            return self

        sx = float(new_w) / float(orig_w)
        sy = float(new_h) / float(orig_h)

        K1_new = self.K1.copy()
        K1_new[0, 0] *= sx
        K1_new[0, 2] *= sx
        K1_new[1, 1] *= sy
        K1_new[1, 2] *= sy

        K2_new = self.K2.copy()
        K2_new[0, 0] *= sx
        K2_new[0, 2] *= sx
        K2_new[1, 1] *= sy
        K2_new[1, 2] *= sy

        return StereoCameraModel(
            K1=K1_new,
            D1=self.D1.copy(),
            K2=K2_new,
            D2=self.D2.copy(),
            R=self.R.copy(),
            T=self.T.copy(),
            image_size=(int(new_w), int(new_h)),
        )

    def save(self, filepath: str) -> None:
        """Save calibration model to JSON file."""
        data = {
            "image_size": list(self.image_size),
            "K1": self.K1.tolist(),
            "D1": self.D1.flatten().tolist(),
            "K2": self.K2.tolist(),
            "D2": self.D2.flatten().tolist(),
            "R": self.R.tolist(),
            "T": self.T.flatten().tolist(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "StereoCameraModel":
        """Load calibration model from JSON or YAML file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file not found: {filepath}")

        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            import yaml
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

        return cls(
            K1=np.array(data["K1"]),
            D1=np.array(data["D1"]),
            K2=np.array(data["K2"]),
            D2=np.array(data["D2"]),
            R=np.array(data["R"]),
            T=np.array(data["T"]),
            image_size=tuple(data.get("image_size", (640, 480))),
        )
