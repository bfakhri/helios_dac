#!/usr/bin/env python3
"""
Unit test for StereoObjectMatcher and TrackSmoother3D.
Tests cross-camera object correspondence matching, Hungarian assignment, and 3D temporal smoothing.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from object_detector import Detection2D
from stereo_geometry import StereoCameraModel
from stereo_matcher import StereoObjectMatcher, TrackSmoother3D


def test_stereo_matching_and_tracking():
    model = StereoCameraModel.generate_synthetic(baseline_m=0.30)
    matcher = StereoObjectMatcher(camera_model=model, max_epipolar_dist_px=30.0)
    smoother = TrackSmoother3D(alpha=0.4, max_match_dist_m=0.5)

    # Ground truth 3D targets in meters:
    # Target A: Person at (X=-0.3, Y=0.0, Z=2.0)
    # Target B: Face at (X=0.4, Y=-0.1, Z=2.5)
    targets_3d = [
        np.array([-0.3, 0.0, 2.0]),
        np.array([0.4, -0.1, 2.5]),
    ]

    # Project to Cam1 and Cam2
    def project_pt(cam_P, p3d):
        p_hom = cam_P @ np.append(p3d, 1.0)
        return p_hom[:2] / p_hom[2]

    p1_a = project_pt(model.P1, targets_3d[0])
    p2_a = project_pt(model.P2, targets_3d[0])

    p1_b = project_pt(model.P1, targets_3d[1])
    p2_b = project_pt(model.P2, targets_3d[1])

    # Create Detection2D instances (with intentionally shuffled order in camera 2)
    dets_cam1 = [
        Detection2D(
            bbox=(int(p1_a[0]-20), int(p1_a[1]-30), int(p1_a[0]+20), int(p1_a[1]+30)),
            confidence=0.92,
            class_id=1,
            class_name="face",
            centroid=(float(p1_a[0]), float(p1_a[1])),
            ground_point=(float(p1_a[0]), float(p1_a[1]+30)),
        ),
        Detection2D(
            bbox=(int(p1_b[0]-15), int(p1_b[1]-20), int(p1_b[0]+15), int(p1_b[1]+20)),
            confidence=0.88,
            class_id=1,
            class_name="face",
            centroid=(float(p1_b[0]), float(p1_b[1])),
            ground_point=(float(p1_b[0]), float(p1_b[1]+20)),
        )
    ]

    # Inverted order in camera 2 to test Hungarian association
    dets_cam2 = [
        Detection2D(
            bbox=(int(p2_b[0]-15), int(p2_b[1]-20), int(p2_b[0]+15), int(p2_b[1]+20)),
            confidence=0.89,
            class_id=1,
            class_name="face",
            centroid=(float(p2_b[0]), float(p2_b[1])),
            ground_point=(float(p2_b[0]), float(p2_b[1]+20)),
        ),
        Detection2D(
            bbox=(int(p2_a[0]-20), int(p2_a[1]-30), int(p2_a[0]+20), int(p2_a[1]+30)),
            confidence=0.94,
            class_id=1,
            class_name="face",
            centroid=(float(p2_a[0]), float(p2_a[1])),
            ground_point=(float(p2_a[0]), float(p2_a[1]+30)),
        )
    ]

    # Frame 1 matching
    matches = matcher.match_and_triangulate(dets_cam1, dets_cam2, smoother=smoother)
    assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"

    # Check 3D triangulated coordinates
    for m in matches:
        if m.det1.centroid == (float(p1_a[0]), float(p1_a[1])):
            err_a = np.linalg.norm(m.point_3d - targets_3d[0])
            assert err_a < 1e-3, f"Target A 3D error too high: {err_a}"
            track_a_id = m.track_id
        else:
            err_b = np.linalg.norm(m.point_3d - targets_3d[1])
            assert err_b < 1e-3, f"Target B 3D error too high: {err_b}"
            track_b_id = m.track_id

    # Frame 2: Slight motion and verify track ID persistence
    dets_cam1_f2 = [
        Detection2D(
            bbox=(int(p1_a[0]-18), int(p1_a[1]-28), int(p1_a[0]+22), int(p1_a[1]+32)),
            confidence=0.92,
            class_id=1,
            class_name="face",
            centroid=(float(p1_a[0] + 1.0), float(p1_a[1])),
            ground_point=(float(p1_a[0] + 1.0), float(p1_a[1]+30)),
        )
    ]
    dets_cam2_f2 = [
        Detection2D(
            bbox=(int(p2_a[0]-18), int(p2_a[1]-28), int(p2_a[0]+22), int(p2_a[1]+32)),
            confidence=0.94,
            class_id=1,
            class_name="face",
            centroid=(float(p2_a[0] + 1.0), float(p2_a[1])),
            ground_point=(float(p2_a[0] + 1.0), float(p2_a[1]+30)),
        )
    ]

    matches_f2 = matcher.match_and_triangulate(dets_cam1_f2, dets_cam2_f2, smoother=smoother)
    assert len(matches_f2) == 1
    assert matches_f2[0].track_id == track_a_id, "Track ID was not preserved across frames!"

    print("✓ test_stereo_matching_and_tracking passed.")


if __name__ == "__main__":
    test_stereo_matching_and_tracking()
    print("All StereoMatcher tests passed successfully!")
