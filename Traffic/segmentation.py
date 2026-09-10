"""Maneuver-centered segments: lane keeping, lane change left/right."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from Traffic.config import (
    LC_WINDOW_STEPS,
    LK_MAX_HEADING_STD_RAD,
    LK_STRIDE_STEPS,
    MNV_LCL,
    MNV_LCR,
    MNV_LK,
    N_RESAMPLE,
)


@dataclass
class SegmentIndex:
    maneuver: int
    i0: int
    i1: int  # inclusive


def _heading_std(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return 1e9
    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    psi = np.arctan2(dy, dx)
    return float(np.std(np.diff(psi)))


def extract_lane_change_segments(
    lane_ids: np.ndarray,
) -> List[SegmentIndex]:
    """3 s before + 3 s after each lane-id transition → ``N_RESAMPLE`` frames."""
    segs: List[SegmentIndex] = []
    n = len(lane_ids)
    half = LC_WINDOW_STEPS // 2
    for i in range(1, n):
        if int(lane_ids[i]) == int(lane_ids[i - 1]):
            continue
        dl = int(lane_ids[i]) - int(lane_ids[i - 1])
        if dl > 0:
            mnv = MNV_LCL
        elif dl < 0:
            mnv = MNV_LCR
        else:
            continue
        i0 = i - half
        i1 = i + half - 1
        if i0 < 0 or i1 >= n:
            continue
        if i1 - i0 + 1 != N_RESAMPLE:
            continue
        segs.append(SegmentIndex(maneuver=mnv, i0=i0, i1=i1))
    return segs


def extract_lane_keeping_segments(
    lane_ids: np.ndarray,
    xy: np.ndarray,
) -> List[SegmentIndex]:
    """Constant-lane runs of length ≥ ``N_RESAMPLE``; one centered window per run."""
    segs: List[SegmentIndex] = []
    n = len(lane_ids)
    w = N_RESAMPLE
    start = 0
    while start < n:
        end = start
        lid = int(lane_ids[start])
        while end + 1 < n and int(lane_ids[end + 1]) == lid:
            end += 1
        run_len = end - start + 1
        if run_len >= w:
            # Multiple non-overlapping (or lightly overlapping) LK windows per long run
            for off in range(0, run_len - w + 1, LK_STRIDE_STEPS):
                i0 = start + off
                i1 = i0 + w - 1
                sub = xy[i0 : i1 + 1]
                if _heading_std(sub) <= LK_MAX_HEADING_STD_RAD * 4:
                    segs.append(SegmentIndex(maneuver=MNV_LK, i0=i0, i1=i1))
        start = end + 1
    return segs
