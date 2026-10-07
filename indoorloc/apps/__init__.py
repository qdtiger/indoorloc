"""L5: applications built on the other layers (numpy only).

    tracking     KalmanTracker (CV / CA, adaptive noise from Prediction.spread, RTS smoother),
                 ExtendedKalmanTracker (raw ranges to anchors), ConstantVelocityKF + track (0.2 sketch)
    particle     ParticleFilter: random-walk or PDR motion, fix / range likelihoods, walls kill particles,
                 augmented-MCL recovery
    maps         FloorMap (walls, bounds, floors, Connector stairs/elevators), OccupancyGrid
    pdr          StepDetector, Weinberg / Kim step length, gyro / compass / fused heading, PDR -> StepTrack
    fusion       PDRFusion: PDR steps + fixes from any localizer in a map-constrained particle filter
    streaming    OnlineLocalizer: (t, scan) stream -> StreamEstimate, missing scans, latency statistics
    navigation   A* on occupancy grids and across floors, line-of-sight smoothing, turn-by-turn Instructions

apps may import core, signals, methods and evaluation, never datasets (data arrives as arrays,
a SampleTable or a stream of scans); no layer imports apps. Both rules are import contracts.
Conventions: positions in the dataset frame (metres), time in seconds, heading in radians
counter-clockwise from +x; offline methods return ``core.Prediction`` so L4 scores them directly.
"""
from __future__ import annotations

from .fusion import PDRFusion
from .maps import Connector, FloorMap, OccupancyGrid
from .navigation import Instruction, Navigator, Route, astar, path_length, smooth_path, turn_instructions
from .particle import ParticleFilter, effective_sample_size, systematic_resample
from .pdr import (PDR, StepDetector, StepEvents, StepTrack, calibrate_step_length, fill_gaps, imu_arrays,
                  kim_step_length, step_lengths, weinberg_step_length)
from .streaming import OnlineLocalizer, StreamEstimate, stack_estimates
from .tracking import (ConstantVelocityKF, ExtendedKalmanTracker, KalmanTracker, motion_model, multilaterate,
                       rts_smooth, track)

__all__ = ["Connector", "ConstantVelocityKF", "ExtendedKalmanTracker", "FloorMap", "Instruction", "KalmanTracker",
           "Navigator", "OccupancyGrid", "OnlineLocalizer", "PDR", "PDRFusion", "ParticleFilter", "Route",
           "StepDetector", "StepEvents", "StepTrack", "StreamEstimate", "astar", "calibrate_step_length",
           "effective_sample_size", "fill_gaps", "imu_arrays", "kim_step_length", "motion_model", "multilaterate",
           "path_length", "rts_smooth", "smooth_path", "stack_estimates", "step_lengths", "systematic_resample",
           "track", "turn_instructions", "weinberg_step_length"]
