"""L1 simulators: physically grounded synthetic data for every modality, numpy only, no download.

Building blocks (usable on their own):

``propagation``   path loss (Friis, log-distance, COST 231 multi-wall, 3GPP TR 38.901
                  InH-Office), LOS probability, correlated shadowing maps, sensitivity -> NaN.
``measurements``  ToA / RTT / UWB ranges with NLOS bias, TDoA, AoA bearings.
``channel``       multipath (LOS, image-method wall reflections, scatterers) and ULA/OFDM CSI.
``building``      floor plans with walls and connectors, anchor layouts, reference grids,
                  pedestrian walks (``Route``) and consistent IMU signals.
``geometry``      segment/wall intersection, distances, mirror images.
``optical``       ceiling LED layouts, Lambertian line-of-sight gain, photodiode shot/thermal noise (VLC).
``magnetic``      a synthetic indoor magnetic field: Earth's field plus dipoles in the floor slabs,
                  flat-device magnetometer readings and their ``[B, B_h, B_v]`` features.

Datasets (registered in ``indoorloc.datasets``):

``SyntheticOffice``  ``load_dataset("synthetic_office", seed=0, modality="wifi_rssi")``;
                     deterministic from ``seed``; splits train / test / trajectory.
``DeepMIMO``         ray-traced DeepMIMO v4 scenarios (needs ``pip install 'indoorloc[sim]'``).

Every simulated table marks ``groups["source"] == "simulated"`` and keeps physical units
(dBm, metres, radians, complex channel gains, watts, microtesla), like the measured datasets.
"""
from __future__ import annotations

from .building import (FloorPlan, NavigationGraph, Route, Trajectory, navigation_graph, office_floor_plan,
                       place_anchors, random_points, random_walk, reference_grid, synthesize_imu)
from .channel import Paths, multipath, ofdm_csi, paths_to_csi, subcarrier_offsets, ula_steering
from .deepmimo import DeepMIMO
from .magnetic import device_readings, dipole_field, earth_field, place_dipoles
from .measurements import bearings, distances, simulate_aoa, simulate_ranges, simulate_tdoa
from .office import SyntheticOffice
from .optical import lambertian_gain, room_grid_leds
from .propagation import (ShadowingMap, apply_sensitivity, free_space_path_loss, inh_office_los_probability,
                          inh_office_path_loss, log_distance_path_loss, multi_wall_path_loss)

__all__ = ["DeepMIMO", "FloorPlan", "NavigationGraph", "Paths", "Route", "ShadowingMap", "SyntheticOffice",
           "Trajectory", "apply_sensitivity", "bearings", "device_readings", "dipole_field", "distances",
           "earth_field", "free_space_path_loss", "inh_office_los_probability", "inh_office_path_loss",
           "lambertian_gain", "log_distance_path_loss", "multi_wall_path_loss", "multipath", "navigation_graph",
           "office_floor_plan", "ofdm_csi", "paths_to_csi", "place_anchors", "place_dipoles", "random_points",
           "random_walk", "reference_grid", "room_grid_leds", "simulate_aoa", "simulate_ranges", "simulate_tdoa",
           "subcarrier_offsets", "synthesize_imu", "ula_steering"]
