#!/usr/bin/env python3
"""
Unit test for StreamSynchronizer.
Verifies time-pairing accuracy, jitter measurement, and stale frame dropping.
"""

import os
import sys
import time
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stream_synchronizer import StreamSynchronizer, TimestampedFrame


def test_perfect_synchronization():
    sync = StreamSynchronizer(max_buffer_size=10, max_delta_ms=20.0)
    t0 = time.perf_counter_ns()

    # Push frames with identical timestamps
    for i in range(5):
        t = t0 + i * 33_333_333  # ~30fps
        img1 = np.zeros((10, 10, 3), dtype=np.uint8)
        img2 = np.ones((10, 10, 3), dtype=np.uint8)
        sync.push_frame(1, img1, timestamp_ns=t)
        sync.push_frame(2, img2, timestamp_ns=t)

    # Verify pairing
    for i in range(5):
        ok, f1, f2 = sync.get_aligned_pair()
        assert ok, f"Failed to pair frame {i}"
        assert f1 is not None and f2 is not None
        assert f1.timestamp_ns == f2.timestamp_ns
        assert f1.seq == i + 1
        assert f2.seq == i + 1

    stats = sync.get_stats()
    assert stats["synced_pairs"] == 5
    assert stats["last_delta_ms"] == 0.0
    print("✓ test_perfect_synchronization passed.")


def test_jittered_synchronization():
    sync = StreamSynchronizer(max_buffer_size=20, max_delta_ms=30.0)
    t0 = time.perf_counter_ns()

    # Stream 1 arrives slightly before Stream 2 (+5ms offset)
    t1 = t0
    t2 = t0 + 5_000_000  # 5ms offset

    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
    img2 = np.ones((10, 10, 3), dtype=np.uint8)

    sync.push_frame(1, img1, timestamp_ns=t1)
    sync.push_frame(2, img2, timestamp_ns=t2)

    ok, f1, f2 = sync.get_aligned_pair(max_delta_ms=10.0)
    assert ok
    assert f1 is not None and f2 is not None
    assert abs(f1.timestamp_ns - f2.timestamp_ns) == 5_000_000

    # Stream offset beyond max_delta_ms (e.g. 50ms) should NOT pair with max_delta_ms=20.0
    t1_late = t0 + 100_000_000
    t2_early = t0 + 40_000_000
    sync.push_frame(1, img1, timestamp_ns=t1_late)
    sync.push_frame(2, img2, timestamp_ns=t2_early)

    ok, _, _ = sync.get_aligned_pair(max_delta_ms=20.0)
    assert not ok
    print("✓ test_jittered_synchronization passed.")


if __name__ == "__main__":
    test_perfect_synchronization()
    test_jittered_synchronization()
    print("All StreamSynchronizer tests passed successfully!")
