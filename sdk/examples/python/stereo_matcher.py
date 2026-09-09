#!/usr/bin/env python3
"""
Stereo Object Matcher & 3D Tracker.
Performs cross-camera detection correspondence using epipolar constraints and appearance similarity,
triangulates 3D coordinates, and smooths object trajectories over time.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from object_detector import Detection2D
from stereo_geometry import StereoCameraModel


@dataclass
class StereoMatch:
    """Represents an associated 3D object detection matched across both cameras."""
    track_id: int
    class_id: int
    class_name: str
    det1: Detection2D
    det2: Detection2D
    point_3d: np.ndarray          # 3D coordinate (X, Y, Z) in meters (Camera 1 frame)
    ground_point_3d: np.ndarray   # 3D ground contact point (X, Y, Z) in meters
    distance_m: float             # Euclidean distance from Camera 1
    epipolar_error_px: float      # Epipolar line alignment discrepancy in pixels
    match_cost: float             # Total correspondence matching cost
    confidence: float             # Mean detection confidence


class StereoObjectMatcher:
    """
    Solves correspondence matching between dual 2D detections using epipolar geometry
    and Hungarian optimization (linear sum assignment).
    """

    def __init__(
        self,
        camera_model: StereoCameraModel,
        max_epipolar_dist_px: float = 45.0,
        epipolar_weight: float = 0.7,
        appearance_weight: float = 0.3,
        min_depth_m: float = 0.1,
        max_depth_m: float = 50.0,
    ):
        self.camera_model = camera_model
        self.max_epipolar_dist_px = max_epipolar_dist_px
        self.epipolar_weight = epipolar_weight
        self.appearance_weight = appearance_weight
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def match_and_triangulate(
        self,
        dets1: List[Detection2D],
        dets2: List[Detection2D],
        smoother: Optional["TrackSmoother3D"] = None,
    ) -> List[StereoMatch]:
        """
        Match 2D detections between Camera 1 and Camera 2 and triangulate into 3D.
        """
        if not dets1 or not dets2:
            return []

        n1 = len(dets1)
        n2 = len(dets2)
        cost_matrix = np.full((n1, n2), 1e6, dtype=np.float64)

        for i, d1 in enumerate(dets1):
            for j, d2 in enumerate(dets2):
                # Hard filter: Semantic class match
                if d1.class_id != d2.class_id:
                    continue

                # Epipolar geometric error
                epi_err = self.camera_model.epipolar_error(d1.centroid, d2.centroid)
                if epi_err > self.max_epipolar_dist_px * 2.0:
                    continue

                # Appearance similarity (HSV histogram correlation)
                app_dist = 0.0
                if d1.color_hist is not None and d2.color_hist is not None:
                    # cv2.HISTCMP_BHATTACHARYYA returns distance in [0, 1]
                    bhat = cv2.compareHist(d1.color_hist, d2.color_hist, cv2.HISTCMP_BHATTACHARYYA)
                    app_dist = float(bhat)

                total_cost = (
                    self.epipolar_weight * epi_err
                    + self.appearance_weight * (app_dist * self.max_epipolar_dist_px)
                )
                cost_matrix[i, j] = total_cost

        # Hungarian algorithm matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches: List[StereoMatch] = []

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost >= 1e5 or cost > (self.max_epipolar_dist_px * 2.5):
                continue

            d1 = dets1[r]
            d2 = dets2[c]

            # Triangulate centroid & ground point
            p3d_arr = self.camera_model.triangulate_points([d1.centroid], [d2.centroid])
            gp3d_arr = self.camera_model.triangulate_points([d1.ground_point], [d2.ground_point])

            if len(p3d_arr) == 0:
                continue

            p3d = p3d_arr[0]
            gp3d = gp3d_arr[0]

            # Depth sanity check (Z must be positive and within realistic range)
            z = float(p3d[2])
            if z < self.min_depth_m or z > self.max_depth_m:
                continue

            epi_err = self.camera_model.epipolar_error(d1.centroid, d2.centroid)
            dist_m = float(np.linalg.norm(p3d))
            conf = float((d1.confidence + d2.confidence) / 2.0)

            match = StereoMatch(
                track_id=0,
                class_id=d1.class_id,
                class_name=d1.class_name,
                det1=d1,
                det2=d2,
                point_3d=p3d,
                ground_point_3d=gp3d,
                distance_m=dist_m,
                epipolar_error_px=epi_err,
                match_cost=cost,
                confidence=conf,
            )
            matches.append(match)

        # Apply temporal smoothing and ID assignment if smoother is supplied
        if smoother is not None:
            matches = smoother.update(matches)

        return matches


class TrackSmoother3D:
    """
    Maintains 3D tracked object identities across frames and applies temporal smoothing
    (Exponential Moving Average) to eliminate bounding-box noise.
    """

    def __init__(self, alpha: float = 0.35, max_match_dist_m: float = 1.0, max_missed_frames: int = 10):
        """
        Args:
            alpha: EMA smoothing factor (0 < alpha <= 1). Higher = more responsive, Lower = smoother.
            max_match_dist_m: Max 3D Euclidean distance to associate detection with existing track.
            max_missed_frames: Number of missed frames before terminating a track ID.
        """
        self.alpha = alpha
        self.max_match_dist_m = max_match_dist_m
        self.max_missed_frames = max_missed_frames
        self._next_track_id = 1
        self._tracks: Dict[int, dict] = {}  # track_id -> {p3d, gp3d, class_name, missed, last_seen}

    def update(self, current_matches: List[StereoMatch]) -> List[StereoMatch]:
        """Update track states with new stereo matches."""
        if not current_matches:
            # Increment missed counters for all active tracks
            dead_tracks = []
            for tid, tdata in self._tracks.items():
                tdata["missed"] += 1
                if tdata["missed"] > self.max_missed_frames:
                    dead_tracks.append(tid)
            for tid in dead_tracks:
                del self._tracks[tid]
            return []

        active_tids = list(self._tracks.keys())
        updated_matches: List[StereoMatch] = []

        if not active_tids:
            # Initialize new tracks for all current matches
            for m in current_matches:
                tid = self._next_track_id
                self._next_track_id += 1
                m.track_id = tid
                self._tracks[tid] = {
                    "p3d": m.point_3d.copy(),
                    "gp3d": m.ground_point_3d.copy(),
                    "class_name": m.class_name,
                    "missed": 0,
                    "last_seen": time.time(),
                }
                updated_matches.append(m)
            return updated_matches

        # Build 3D distance association matrix
        d_matrix = np.full((len(current_matches), len(active_tids)), 1e6, dtype=np.float64)
        for i, m in enumerate(current_matches):
            for j, tid in enumerate(active_tids):
                track_p3d = self._tracks[tid]["p3d"]
                if self._tracks[tid]["class_name"] == m.class_name:
                    d_matrix[i, j] = float(np.linalg.norm(m.point_3d - track_p3d))

        row_ind, col_ind = linear_sum_assignment(d_matrix)
        matched_match_indices = set()
        matched_track_indices = set()

        for r, c in zip(row_ind, col_ind):
            dist = d_matrix[r, c]
            if dist <= self.max_match_dist_m:
                tid = active_tids[c]
                m = current_matches[r]
                m.track_id = tid

                # Apply EMA smoothing
                prev_p3d = self._tracks[tid]["p3d"]
                prev_gp3d = self._tracks[tid]["gp3d"]
                smoothed_p3d = self.alpha * m.point_3d + (1.0 - self.alpha) * prev_p3d
                smoothed_gp3d = self.alpha * m.ground_point_3d + (1.0 - self.alpha) * prev_gp3d

                m.point_3d = smoothed_p3d
                m.ground_point_3d = smoothed_gp3d
                m.distance_m = float(np.linalg.norm(smoothed_p3d))

                self._tracks[tid]["p3d"] = smoothed_p3d
                self._tracks[tid]["gp3d"] = smoothed_gp3d
                self._tracks[tid]["missed"] = 0
                self._tracks[tid]["last_seen"] = time.time()

                updated_matches.append(m)
                matched_match_indices.add(r)
                matched_track_indices.add(c)

        # Unmatched new detections get new track IDs
        for i, m in enumerate(current_matches):
            if i not in matched_match_indices:
                tid = self._next_track_id
                self._next_track_id += 1
                m.track_id = tid
                self._tracks[tid] = {
                    "p3d": m.point_3d.copy(),
                    "gp3d": m.ground_point_3d.copy(),
                    "class_name": m.class_name,
                    "missed": 0,
                    "last_seen": time.time(),
                }
                updated_matches.append(m)

        # Unmatched tracks get missed increment
        dead_tracks = []
        for j, tid in enumerate(active_tids):
            if j not in matched_track_indices:
                self._tracks[tid]["missed"] += 1
                if self._tracks[tid]["missed"] > self.max_missed_frames:
                    dead_tracks.append(tid)

        for tid in dead_tracks:
            del self._tracks[tid]

        return updated_matches
