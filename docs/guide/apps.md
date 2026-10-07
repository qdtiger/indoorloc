# L5 applications

`indoorloc.apps` builds tracking, pedestrian dead reckoning (PDR), map-constrained fusion,
streaming and navigation on top of the other layers. It imports numpy only, and it never imports
L1: data arrives as arrays, a `SampleTable` or a stream of `(t, scan)` pairs.

[Guide home](index.md) · [中文](../zh/apps.md)

## Conventions

* Positions are in the dataset's frame (metres), time is in seconds, and headings are radians
  counter-clockwise from +x.
* Classes with parameters are `core.Estimator`s: `get_params`, `clone`, `save` and `load_model`
  work as in L3. A `FloorMap` held by a particle filter is saved as arrays, without pickle.
* Trackers share one online protocol: `update(t, z, spread=None)`, `predict_to(t)`, `estimate()`
  and `reset(t)`. An L3 `Prediction.spread` can serve as the measurement noise of each fix.
* **Every time argument is absolute**: `t` is a time stamp in seconds on the stream's clock, never
  an increment. `predict_to(t)` has that meaning on every tracker; the older `predict` does not
  (`KalmanTracker.predict(t)` takes an absolute time, `ParticleFilter.predict(dt)` an increment), so
  use `predict_to`.
* What a batch method returns depends on what its rows are:

  | Method | Returns | Rows |
  | --- | --- | --- |
  | `KalmanTracker.filter` / `smooth`, `ExtendedKalmanTracker.filter` / `smooth`, `ParticleFilter.filter` | `core.Prediction` | one per input row (NaN before the track starts) |
  | `PDR.run` | `StepTrack` (`t`, `pos`, `length`, `heading`; `position_at(t)` interpolates) | one per detected step |
  | `PDRFusion.run` | `(t, core.Prediction)` | one per event, steps and fixes merged in time order |
  | `OnlineLocalizer.run` | an iterator of `StreamEstimate`; `stack_estimates` turns them into `(t, core.Prediction)` | one per stream item |

  A tracker's `spread` is `sqrt(trace P_pos)` (for particles, of the weighted cloud). L4 scores
  every `Prediction` like the output of an L3 method: a tracker's directly against the table, the
  others against the truth at their times `t`.

## Catalog

<!-- catalog:apps -->
| Name | Module | Kind | What it does | Reference |
| --- | --- | --- | --- | --- |
| [`ConstantVelocityKF`](../../indoorloc/apps/tracking.py) | `apps.tracking` | class | Minimal 2-D constant-velocity Kalman filter used by `track` (the 0.2 sketch API). | Y. Bar-Shalom, X. R. Li, T. Kirubarajan, "Estimation with Applications to Tracking and Navigation", Wiley, 2001, Sec. 6.3.2 (discrete white noise acceleration model). [doi:10.1002/0471221279](https://doi.org/10.1002/0471221279). |
| [`ExtendedKalmanTracker`](../../indoorloc/apps/tracking.py) | `apps.tracking` | class | Extended Kalman tracker fed directly by ranges to known anchors (tight coupling). | Y. Bar-Shalom, X. R. Li, T. Kirubarajan, "Estimation with Applications to Tracking and Navigation", Wiley, 2001, Sec. 10.3 (extended Kalman filter). [doi:10.1002/0471221279](https://doi.org/10.1002/0471221279). (+1 more in the docstring) |
| [`KalmanTracker`](../../indoorloc/apps/tracking.py) | `apps.tracking` | class | Kalman tracker of a localizer's fixes with adaptive measurement noise and an RTS smoother. | R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems", Journal of Basic Engineering 82(1):35-45, 1960. [doi:10.1115/1.3662552](https://doi.org/10.1115/1.3662552). (+2 more in the docstring) |
| [`motion_model`](../../indoorloc/apps/tracking.py) | `apps.tracking` | function | Transition `F` and process noise `Q` of a nearly-constant velocity / acceleration model. | — |
| [`multilaterate`](../../indoorloc/apps/tracking.py) | `apps.tracking` | function | Least-squares position from ranges to anchors (NaN ranges are ignored), or None. | — |
| [`rts_smooth`](../../indoorloc/apps/tracking.py) | `apps.tracking` | function | Rauch-Tung-Striebel fixed-interval smoother. | — |
| [`track`](../../indoorloc/apps/tracking.py) | `apps.tracking` | function | stream yields (t_seconds, scan (F,) dBm with NaN). Yields (t, raw fix, filtered position). | — |
| [`ParticleFilter`](../../indoorloc/apps/particle.py) | `apps.particle` | class | Sequential Monte Carlo position filter with map constraints. | N. J. Gordon, D. J. Salmond, A. F. M. Smith, "Novel approach to nonlinear/non-Gaussian Bayesian state estimation", IEE Proceedings F 140(2):107-113, 1993. [doi:10.1049/ip-f-2.1993.0015](https://doi.org/10.1049/ip-f-2.1993.0015). (+3 more in the docstring) |
| [`effective_sample_size`](../../indoorloc/apps/particle.py) | `apps.particle` | function | `1 / sum(w_i^2)` of normalised weights: N for uniform weights, 1 for a single particle. | — |
| [`systematic_resample`](../../indoorloc/apps/particle.py) | `apps.particle` | function | Indices of a systematic resample (Kitagawa 1996): one uniform offset `u`, points | — |
| [`Connector`](../../indoorloc/apps/maps.py) | `apps.maps` | class | A vertical link between floors: stairs, an elevator, an escalator or a ramp. | — |
| [`FloorMap`](../../indoorloc/apps/maps.py) | `apps.maps` | class | Walls, bounds and floor connectors of a building, in the dataset frame (metres). | T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, "Introduction to Algorithms", 3rd ed., MIT Press, 2009, Sec. 33.1 (segment intersection, used by `crosses`). (+1 more in the docstring) |
| [`OccupancyGrid`](../../indoorloc/apps/maps.py) | `apps.maps` | class | A rasterised floor: `blocked[r, c]` is True where a wall (plus clearance) lies. | — |
| [`PDR`](../../indoorloc/apps/pdr.py) | `apps.pdr` | class | Pedestrian dead reckoning: detect steps, size them, orient them, add them up. | R. Harle, "A Survey of Indoor Inertial Positioning Systems for Pedestrians", IEEE Communications Surveys & Tutorials 15(3):1281-1293, 2013. [doi:10.1109/SURV.2012.121912.00075](https://doi.org/10.1109/SURV.2012.121912.00075). (+2 more in the docstring) |
| [`StepDetector`](../../indoorloc/apps/pdr.py) | `apps.pdr` | class | Step detection by peak picking on the acceleration magnitude. | H. Weinberg, "Using the ADXL202 in Pedometer and Personal Navigation Applications", Analog Devices AN-602, 2002. (+1 more in the docstring) |
| [`StepEvents`](../../indoorloc/apps/pdr.py) | `apps.pdr` | class | Detected steps as parallel arrays (one entry per step). | — |
| [`StepTrack`](../../indoorloc/apps/pdr.py) | `apps.pdr` | class | A PDR trajectory: one entry per step (parallel arrays), plus where it started. | — |
| [`calibrate_step_length`](../../indoorloc/apps/pdr.py) | `apps.pdr` | function | The `k` that makes the steps of a walk of known `distance` (metres) add up to it: | — |
| [`fill_gaps`](../../indoorloc/apps/pdr.py) | `apps.pdr` | function | Missing samples (NaN, the L1 convention) linearly interpolated in time, per column; | — |
| [`imu_arrays`](../../indoorloc/apps/pdr.py) | `apps.pdr` | function | `{"acc", "gyro", "mag", "t"}` arrays from an IMU `SampleTable`. | — |
| [`kim_step_length`](../../indoorloc/apps/pdr.py) | `apps.pdr` | function | Kim et al. (2004): `L = k (mean \|a\|)^(1/3)` over the samples of each step. | — |
| [`step_lengths`](../../indoorloc/apps/pdr.py) | `apps.pdr` | function | `(S,)` step lengths in metres. `model` is `"weinberg"`, `"kim"` or `"constant"` | — |
| [`weinberg_step_length`](../../indoorloc/apps/pdr.py) | `apps.pdr` | function | Weinberg (2002): `L = k (a_max - a_min)^(1/4)` per step (accelerations in m/s^2). | — |
| [`PDRFusion`](../../indoorloc/apps/fusion.py) | `apps.fusion` | class | Particle-filter fusion of PDR steps with position fixes from any localizer. | F. Evennou, F. Marx, "Advanced integration of WiFi and inertial navigation systems for indoor mobile positioning", EURASIP J. Adv. Signal Process. 2006:086706, 2006. [doi:10.1155/ASP/2006/86706](https://doi.org/10.1155/ASP/2006/86706). (+1 more in the docstring) |
| [`OnlineLocalizer`](../../indoorloc/apps/streaming.py) | `apps.streaming` | class | Online localization of a stream of scans with an optional tracker. | — |
| [`StreamEstimate`](../../indoorloc/apps/streaming.py) | `apps.streaming` | class | The output for one `(t, scan)` of the stream. | — |
| [`stack_estimates`](../../indoorloc/apps/streaming.py) | `apps.streaming` | function | `(t (N,), Prediction)` from a sequence of stream estimates, for L4 evaluation | — |
| [`Instruction`](../../indoorloc/apps/navigation.py) | `apps.navigation` | class | One turn-by-turn instruction. | — |
| [`Navigator`](../../indoorloc/apps/navigation.py) | `apps.navigation` | class | Shortest walking routes on a (multi-floor) floor plan. | P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths", IEEE Trans. Systems Science and Cybernetics 4(2):100-107, 1968. [doi:10.1109/TSSC.1968.300136](https://doi.org/10.1109/TSSC.1968.300136). (+1 more in the docstring) |
| [`Route`](../../indoorloc/apps/navigation.py) | `apps.navigation` | class | A route: `points` `(K, 2)` metric vertices, `floor` `(K,)`, `length` walked in the | — |
| [`astar`](../../indoorloc/apps/navigation.py) | `apps.navigation` | function | Shortest path on a grid: `(K, 2)` int `(row, col)` cells from `start` to `goal` | — |
| [`path_length`](../../indoorloc/apps/navigation.py) | `apps.navigation` | function | Length of a polyline `(K, D)` (sum of segment lengths; 0 for fewer than 2 points). | — |
| [`smooth_path`](../../indoorloc/apps/navigation.py) | `apps.navigation` | function | Greedy line-of-sight post-smoothing: from each kept vertex jump to the farthest later | — |
| [`turn_instructions`](../../indoorloc/apps/navigation.py) | `apps.navigation` | function | Turn-by-turn instructions for a polyline `(K, 2)` (optionally with `(K,)` floors). | — |
<!-- /catalog:apps -->

## Tracking fixes

The simulated office provides walks sampled once per second (WiFi) or five times per second
(ranges), with `groups["trajectory"]` and `groups["time"]`. All numbers on this page come from
that simulation.

```python
import numpy as np
import indoorloc as iloc
from indoorloc.apps import ExtendedKalmanTracker, KalmanTracker

train, test = iloc.load_dataset("synthetic_office")
model = iloc.create_model("wknn", preprocess=iloc.FillMissing(-104)).fit(train)
walks = iloc.load_dataset("synthetic_office", split="trajectory")
walk = walks[walks.groups["trajectory"] == 0]
fixes = model.localize(walk)                                   # one WKNN fix per scan
kf = KalmanTracker(motion="cv", process_noise=0.5)
filtered = kf.filter(fixes, t=walk.groups["time"])             # causal
smoothed = kf.smooth(fixes, t=walk.groups["time"])             # + Rauch-Tung-Striebel pass
print([round(iloc.evaluate(walk, p).mean_error, 2) for p in (fixes, filtered, smoothed)])
# [2.97, 2.53, 2.11]
```

`update` runs the same filter one fix at a time:

```python
kf = KalmanTracker().reset(0.0)
for t, z, s in zip(walk.groups["time"], fixes.pos, fixes.spread):
    kf.update(t, z, spread=s)
pos, spread = kf.estimate()
print(np.round(pos, 2), round(float(spread), 2), np.round(walk.pos[-1], 2))
# [25.49  9.17] 2.88 [24.87  9.25]
```

`predict_to(t)` extrapolates a tracker to the absolute time `t` without a fix. The particle filter
runs the same protocol, with a random walk between fixes:

```python
from indoorloc.apps import ParticleFilter

pf = ParticleFilter(500, random_state=0)
for t, z, s in zip(walk.groups["time"], fixes.pos, fixes.spread):
    pf.update(t, z, spread=s)
t_end = walk.groups["time"][-1]                                # absolute times, not increments
print(np.round(kf.predict_to(t_end + 2.0), 2), np.round(pf.predict_to(t_end + 2.0), 2))
# [28.92 12.88] [22.79  6.87]
```

The Kalman tracker carries its velocity forward; the random walk of the particle filter leaves the
mean roughly where it was and widens the cloud.

The extended Kalman tracker takes raw ranges to known anchors:

```python
ranges = iloc.load_dataset("synthetic_office", split="trajectory", modality="ranges")
r_walk = ranges[ranges.groups["trajectory"] == 0]
ekf = ExtendedKalmanTracker(r_walk.meta["anchors"], range_std=0.1)
per_epoch = iloc.create_model("trilateration", anchors=r_walk.meta["anchors"]).fit(r_walk)
print(round(iloc.evaluate(r_walk, per_epoch.localize(r_walk)).mean_error, 3),
      round(iloc.evaluate(r_walk, ekf.smooth(r_walk.X, t=r_walk.groups["time"])).mean_error, 3))
# 0.564 0.314
```

## Streaming

`OnlineLocalizer` localizes a stream of `(t, scan)` pairs with an optional tracker. A scan that
is `None`, or has fewer than `min_readings` readings, counts as missing. The time spent on each
scan is recorded:

```python
from indoorloc.apps import OnlineLocalizer

online = OnlineLocalizer(model, tracker=KalmanTracker(), floor_window=3)
stream = zip(walk.groups["time"], walk.X)                      # any iterable of (t, scan)
estimates = list(online.run(stream))
last = estimates[-1]
print(len(estimates), np.round(last.pos, 2), last.floor, last.missing, sorted(online.latency_stats())[:3])
# 60 [25.49  9.17] 0 False ['max_ms', 'mean_ms', 'n']
```

## PDR and map-constrained fusion

`PDR` detects steps in an IMU table, sizes them (Weinberg or Kim model) and orients them (gyro,
compass, a complementary fusion of both, or a given yaw). `PDRFusion` combines the steps with fixes
from any localizer in a particle filter. The floor plan comes from `meta["floor_plan"]`, and a
particle whose step crosses a wall is removed. Neither returns a bare `Prediction`: `PDR.run`
returns a `StepTrack` (one entry per step) and `PDRFusion.run` returns `(t, Prediction)` (one row
per step or fix), so the example scores both against the true path interpolated at their times.

```python
from indoorloc.apps import PDR, FloorMap, ParticleFilter, PDRFusion

imu = iloc.load_dataset("synthetic_office", split="trajectory", modality="imu")    # 50 Hz acc + gyro
floor_map = FloorMap.from_dict(walks.meta["floor_plan"])
template = ParticleFilter(1000, step_length_std=0.1, heading_std=0.05, heading_drift_std=0.005,
                          motion_std=0.0, recovery=(0.05, 0.5), floor_map=floor_map, floor=0)
errors = {"WKNN fixes": [], "PDR": [], "fused": []}
for k in range(4):
    steps_in = imu[imu.groups["trajectory"] == k]
    scans = walks[walks.groups["trajectory"] == k]
    t_imu = steps_in.groups["time"]
    first = steps_in.pos[50] - steps_in.pos[0]                  # the gyro needs the start heading
    track = PDR(initial_heading=np.arctan2(first[1], first[0])).run(steps_in, start=steps_in.pos[0])
    fused_t, fused = PDRFusion(template, random_state=0).run(track, model.localize(scans), scans.groups["time"])
    truth = lambda t: np.column_stack([np.interp(t, t_imu, steps_in.pos[:, d]) for d in (0, 1)])  # noqa: E731
    errors["WKNN fixes"].append(iloc.evaluate(scans, model.localize(scans)).mean_error)
    errors["PDR"].append(iloc.evaluate(truth(track.t), track.pos).mean_error)
    errors["fused"].append(iloc.evaluate(truth(fused_t), fused.pos).mean_error)
print({name: round(float(np.mean(e)), 2) for name, e in errors.items()})
# {'WKNN fixes': 2.8, 'PDR': 3.89, 'fused': 2.11}
```

The `template` above is `PDRFusion`'s default filter (step-length and heading noise, no random
walk between steps, augmented-MCL recovery) plus the floor map. A bare `ParticleFilter(n,
floor_map=...)` keeps `ParticleFilter`'s own defaults, which are meant for random-walk tracking:
on these four walks it gave a fused error of 9.62 m with 500 particles. Start from the template
above when adding a map.

## Navigation

`Navigator` rasterizes a `FloorMap`, keeps a clearance from the walls, and returns A* routes
(across floors through stairs and elevators) with turn-by-turn instructions:

```python
from indoorloc.apps import Navigator

nav = Navigator(resolution=0.25, clearance=0.2).fit(floor_map)
route = nav.route((2.0, 2.0), (36.0, 17.0))
print(round(route.length, 1), [step.text for step in route.instructions][:2])
# 47.2 ['Start and walk 7.2 m', 'Turn right, then walk 33.5 m']
```

## Measured data

`tests/apps/test_apps_realdata.py` runs the L5 components on the ILC 2020 sample (real phone
traces from two shopping malls): step lengths, PDR with the phone's heading from a known start,
the tilt-compensated compass against the phone's orientation, the floor plan against the walked
paths, and WiFi tracking with PDR fusion. The tests need the
`ilc2020` files and are skipped without them. `apps.pdr.imu_arrays` and
`FloorMap.from_dict(table.meta["floor_plan"])` connect the loader's tables to the classes above.
