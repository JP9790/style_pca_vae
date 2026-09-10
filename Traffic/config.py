"""NGSIM US-101 / I-80 preprocessing constants."""

from __future__ import annotations

# Sampling (NGSIM vehicle data are typically 10 Hz)
DT_S = 0.1
FPS = 10

# Raw track: keep vehicles with at least this duration (seconds)
MIN_TRACK_DURATION_S = 8.0
MIN_TRACK_STEPS = int(MIN_TRACK_DURATION_S / DT_S)  # 80

# Driver retention: minimum valid segments after segmentation
MIN_SEGMENTS_PER_DRIVER = 10
# Require segments from at least this many distinct maneuver labels {LK, LCL, LCR}
MIN_MANEUVER_TYPES = 2

# Lane-change centered window (seconds before + after transition); total = 6 s
LC_HALF_WINDOW_S = 3.0
LC_WINDOW_STEPS = int(2 * LC_HALF_WINDOW_S / DT_S)  # 60

# Resampled segment length (10 Hz × 6 s)
N_RESAMPLE = 60
SEGMENT_DURATION_S = N_RESAMPLE * DT_S  # 6.0

# Lane-keeping: heading stability (std of yaw differences, radians)
LK_MAX_HEADING_STD_RAD = 0.12
# Sliding 6 s windows along a stable-lane run (steps at 10 Hz); 30 = 3 s stride
LK_STRIDE_STEPS = 30

# EKF / process noise (feet^2, (ft/s)^2 — NGSIM uses feet)
EKF_MEASUREMENT_NOISE_POS_FT2 = 2.0**2
EKF_PROCESS_NOISE_VEL_FT2_S2 = 1.0**2

# Default road subset
DEFAULT_LOCATIONS = ("us-101", "i-80")

# Maneuver labels
MNV_LK = 0
MNV_LCL = 1
MNV_LCR = 2
