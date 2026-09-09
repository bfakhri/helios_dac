#!/usr/bin/env python3
"""
Stream Synchronizer for Dual RTSP Video Streams.
Buffers timestamped frames and pairs them based on temporal proximity (|t1 - t2| < max_delta).
"""

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Optional, Tuple
import numpy as np


@dataclass
class TimestampedFrame:
    """Represents a single video frame with acquisition timing metadata."""
    frame: np.ndarray
    timestamp_ns: int      # Monotonic acquisition timestamp in nanoseconds
    stream_id: int         # 1 or 2
    seq: int               # Sequential frame counter


class StreamSynchronizer:
    """
    Time-alignment synchronizer for dual video streams.
    Maintains ring buffers for each stream and extracts optimal timestamped pairs.
    """

    def __init__(self, max_buffer_size: int = 60, max_delta_ms: float = 40.0):
        """
        Args:
            max_buffer_size: Maximum frames stored per stream queue before dropping oldest.
            max_delta_ms: Maximum allowed timestamp difference (in ms) to consider two frames aligned.
        """
        self.max_buffer_size = max_buffer_size
        self.max_delta_ns = int(max_delta_ms * 1_000_000)

        self._buf1: deque[TimestampedFrame] = deque(maxlen=max_buffer_size)
        self._buf2: deque[TimestampedFrame] = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()

        # Sequence counters
        self._seq1 = 0
        self._seq2 = 0

        # Diagnostics / Statistics
        self.synced_pairs_count = 0
        self.dropped_frames_1 = 0
        self.dropped_frames_2 = 0
        self.last_sync_delta_ms = 0.0
        self.avg_sync_delta_ms = 0.0
        self.jitter_ms = 0.0
        self._delta_history: deque[float] = deque(maxlen=30)

    def push_frame(self, stream_id: int, frame: np.ndarray, timestamp_ns: Optional[int] = None) -> None:
        """Push a newly captured frame into the stream's buffer."""
        if frame is None:
            return

        if timestamp_ns is None:
            timestamp_ns = time.perf_counter_ns()

        with self._lock:
            if stream_id == 1:
                self._seq1 += 1
                tf = TimestampedFrame(frame=frame, timestamp_ns=timestamp_ns, stream_id=1, seq=self._seq1)
                if len(self._buf1) == self.max_buffer_size:
                    self.dropped_frames_1 += 1
                self._buf1.append(tf)
            elif stream_id == 2:
                self._seq2 += 1
                tf = TimestampedFrame(frame=frame, timestamp_ns=timestamp_ns, stream_id=2, seq=self._seq2)
                if len(self._buf2) == self.max_buffer_size:
                    self.dropped_frames_2 += 1
                self._buf2.append(tf)

    def get_aligned_pair(
        self,
        max_delta_ms: Optional[float] = None,
        drop_stale: bool = True
    ) -> Tuple[bool, Optional[TimestampedFrame], Optional[TimestampedFrame]]:
        """
        Search for the closest frame pair between Stream 1 and Stream 2 within time tolerance.

        Returns:
            (success, frame1, frame2)
        """
        allowed_delta_ns = int(max_delta_ms * 1_000_000) if max_delta_ms is not None else self.max_delta_ns

        with self._lock:
            if not self._buf1 or not self._buf2:
                return False, None, None

            best_i = None
            best_j = None
            best_diff = float("inf")

            # Find the pair in buffers that minimizes absolute timestamp difference
            # Buffers are chronological, so we inspect available items
            for i, f1 in enumerate(self._buf1):
                for j, f2 in enumerate(self._buf2):
                    diff = abs(f1.timestamp_ns - f2.timestamp_ns)
                    if diff < best_diff:
                        best_diff = diff
                        best_i = i
                        best_j = j

            if best_i is not None and best_j is not None and best_diff <= allowed_delta_ns:
                f1_matched = self._buf1[best_i]
                f2_matched = self._buf2[best_j]

                # Update metrics
                delta_ms = best_diff / 1_000_000.0
                self.last_sync_delta_ms = delta_ms
                self._delta_history.append(delta_ms)
                self.avg_sync_delta_ms = float(np.mean(self._delta_history))
                self.jitter_ms = float(np.std(self._delta_history)) if len(self._delta_history) > 1 else 0.0
                self.synced_pairs_count += 1

                if drop_stale:
                    # Drop all frames up to and including the matched ones
                    for _ in range(best_i + 1):
                        self._buf1.popleft()
                    for _ in range(best_j + 1):
                        self._buf2.popleft()

                return True, f1_matched, f2_matched

            # If no pair meets max_delta, check if one buffer is drastically ahead of the other
            if drop_stale and self._buf1 and self._buf2:
                oldest_1 = self._buf1[0].timestamp_ns
                oldest_2 = self._buf2[0].timestamp_ns

                # If oldest in buf1 is way older than newest in buf2, drop oldest in buf1
                if oldest_1 < oldest_2 - allowed_delta_ns:
                    self._buf1.popleft()
                    self.dropped_frames_1 += 1
                elif oldest_2 < oldest_1 - allowed_delta_ns:
                    self._buf2.popleft()
                    self.dropped_frames_2 += 1

            return False, None, None

    def get_stats(self) -> dict:
        """Return real-time synchronization diagnostics."""
        with self._lock:
            return {
                "synced_pairs": self.synced_pairs_count,
                "buf1_size": len(self._buf1),
                "buf2_size": len(self._buf2),
                "dropped1": self.dropped_frames_1,
                "dropped2": self.dropped_frames_2,
                "last_delta_ms": self.last_sync_delta_ms,
                "avg_delta_ms": self.avg_sync_delta_ms,
                "jitter_ms": self.jitter_ms,
            }

    def clear(self) -> None:
        """Clear all buffers and reset diagnostics."""
        with self._lock:
            self._buf1.clear()
            self._buf2.clear()
            self._delta_history.clear()
