# L5 应用

`indoorloc.apps` 在其他各层之上实现跟踪、行人航位推算（PDR）、地图约束融合、流式定位与导航。它只导入 numpy，
并且从不导入 L1：数据以数组、`SampleTable` 或 `(t, scan)` 流的形式传入。

[指南首页](index.md) · [English](../guide/apps.md)

## 约定

* 位置使用数据集自己的坐标系（米），时间单位为秒，航向为从 +x 轴逆时针的弧度。
* 带参数的类都是 `core.Estimator`：`get_params`、`clone`、`save`、`load_model` 与 L3 相同。
  粒子滤波器持有的 `FloorMap` 以数组形式保存，不使用 pickle。
* 跟踪器共享同一个在线接口：`update(t, z, spread=None)`、`predict_to(t)`、`estimate()` 和 `reset(t)`。
  L3 的 `Prediction.spread` 可以作为每个定位结果的观测噪声。
* **所有时间参数都是绝对时间**：`t` 是数据流时钟上的时间戳（秒），而不是时间增量。`predict_to(t)` 在每个跟踪器上都是这个含义；
  旧的 `predict` 则不统一（`KalmanTracker.predict(t)` 接受绝对时间，`ParticleFilter.predict(dt)` 接受时间增量），因此请使用 `predict_to`。
* 批处理方法的返回值取决于它的每一行代表什么：

  | 方法 | 返回 | 行 |
  | --- | --- | --- |
  | `KalmanTracker.filter` / `smooth`、`ExtendedKalmanTracker.filter` / `smooth`、`ParticleFilter.filter` | `core.Prediction` | 每个输入行一行（跟踪开始前为 NaN） |
  | `PDR.run` | `StepTrack`（`t`、`pos`、`length`、`heading`；`position_at(t)` 做插值） | 每个检测到的步一行 |
  | `PDRFusion.run` | `(t, core.Prediction)` | 每个事件一行，步与定位结果按时间合并 |
  | `OnlineLocalizer.run` | `StreamEstimate` 的迭代器；`stack_estimates` 把它们转成 `(t, core.Prediction)` | 数据流中每一项一行 |

  跟踪器的 `spread` 为 `sqrt(trace P_pos)`（粒子滤波取加权粒子云的值）。L4 像评测 L3 方法的输出一样评测每个 `Prediction`：
  跟踪器的结果直接与表比较，其余结果与各自时间 `t` 处的真值比较。

## 目录

表中“作用”与参考文献取自英文 docstring。

<!-- catalog:apps -->
| 名称 | 模块 | 类型 | 作用（取自 docstring） | 参考文献 |
| --- | --- | --- | --- | --- |
| [`ConstantVelocityKF`](../../indoorloc/apps/tracking.py) | `apps.tracking` | 类 | Minimal 2-D constant-velocity Kalman filter used by `track` (the 0.2 sketch API). | Y. Bar-Shalom, X. R. Li, T. Kirubarajan, "Estimation with Applications to Tracking and Navigation", Wiley, 2001, Sec. 6.3.2 (discrete white noise acceleration model). [doi:10.1002/0471221279](https://doi.org/10.1002/0471221279). |
| [`ExtendedKalmanTracker`](../../indoorloc/apps/tracking.py) | `apps.tracking` | 类 | Extended Kalman tracker fed directly by ranges to known anchors (tight coupling). | Y. Bar-Shalom, X. R. Li, T. Kirubarajan, "Estimation with Applications to Tracking and Navigation", Wiley, 2001, Sec. 10.3 (extended Kalman filter). [doi:10.1002/0471221279](https://doi.org/10.1002/0471221279). (docstring 中另有 1 篇) |
| [`KalmanTracker`](../../indoorloc/apps/tracking.py) | `apps.tracking` | 类 | Kalman tracker of a localizer's fixes with adaptive measurement noise and an RTS smoother. | R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems", Journal of Basic Engineering 82(1):35-45, 1960. [doi:10.1115/1.3662552](https://doi.org/10.1115/1.3662552). (docstring 中另有 2 篇) |
| [`motion_model`](../../indoorloc/apps/tracking.py) | `apps.tracking` | 函数 | Transition `F` and process noise `Q` of a nearly-constant velocity / acceleration model. | — |
| [`multilaterate`](../../indoorloc/apps/tracking.py) | `apps.tracking` | 函数 | Least-squares position from ranges to anchors (NaN ranges are ignored), or None. | — |
| [`rts_smooth`](../../indoorloc/apps/tracking.py) | `apps.tracking` | 函数 | Rauch-Tung-Striebel fixed-interval smoother. | — |
| [`track`](../../indoorloc/apps/tracking.py) | `apps.tracking` | 函数 | stream yields (t_seconds, scan (F,) dBm with NaN). Yields (t, raw fix, filtered position). | — |
| [`ParticleFilter`](../../indoorloc/apps/particle.py) | `apps.particle` | 类 | Sequential Monte Carlo position filter with map constraints. | N. J. Gordon, D. J. Salmond, A. F. M. Smith, "Novel approach to nonlinear/non-Gaussian Bayesian state estimation", IEE Proceedings F 140(2):107-113, 1993. [doi:10.1049/ip-f-2.1993.0015](https://doi.org/10.1049/ip-f-2.1993.0015). (docstring 中另有 3 篇) |
| [`effective_sample_size`](../../indoorloc/apps/particle.py) | `apps.particle` | 函数 | `1 / sum(w_i^2)` of normalised weights: N for uniform weights, 1 for a single particle. | — |
| [`systematic_resample`](../../indoorloc/apps/particle.py) | `apps.particle` | 函数 | Indices of a systematic resample (Kitagawa 1996): one uniform offset `u`, points | — |
| [`Connector`](../../indoorloc/apps/maps.py) | `apps.maps` | 类 | A vertical link between floors: stairs, an elevator, an escalator or a ramp. | — |
| [`FloorMap`](../../indoorloc/apps/maps.py) | `apps.maps` | 类 | Walls, bounds and floor connectors of a building, in the dataset frame (metres). | T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, "Introduction to Algorithms", 3rd ed., MIT Press, 2009, Sec. 33.1 (segment intersection, used by `crosses`). (docstring 中另有 1 篇) |
| [`OccupancyGrid`](../../indoorloc/apps/maps.py) | `apps.maps` | 类 | A rasterised floor: `blocked[r, c]` is True where a wall (plus clearance) lies. | — |
| [`PDR`](../../indoorloc/apps/pdr.py) | `apps.pdr` | 类 | Pedestrian dead reckoning: detect steps, size them, orient them, add them up. | R. Harle, "A Survey of Indoor Inertial Positioning Systems for Pedestrians", IEEE Communications Surveys & Tutorials 15(3):1281-1293, 2013. [doi:10.1109/SURV.2012.121912.00075](https://doi.org/10.1109/SURV.2012.121912.00075). (docstring 中另有 2 篇) |
| [`StepDetector`](../../indoorloc/apps/pdr.py) | `apps.pdr` | 类 | Step detection by peak picking on the acceleration magnitude. | H. Weinberg, "Using the ADXL202 in Pedometer and Personal Navigation Applications", Analog Devices AN-602, 2002. (docstring 中另有 1 篇) |
| [`StepEvents`](../../indoorloc/apps/pdr.py) | `apps.pdr` | 类 | Detected steps as parallel arrays (one entry per step). | — |
| [`StepTrack`](../../indoorloc/apps/pdr.py) | `apps.pdr` | 类 | A PDR trajectory: one entry per step (parallel arrays), plus where it started. | — |
| [`calibrate_step_length`](../../indoorloc/apps/pdr.py) | `apps.pdr` | 函数 | The `k` that makes the steps of a walk of known `distance` (metres) add up to it: | — |
| [`fill_gaps`](../../indoorloc/apps/pdr.py) | `apps.pdr` | 函数 | Missing samples (NaN, the L1 convention) linearly interpolated in time, per column; | — |
| [`imu_arrays`](../../indoorloc/apps/pdr.py) | `apps.pdr` | 函数 | `{"acc", "gyro", "mag", "t"}` arrays from an IMU `SampleTable`. | — |
| [`kim_step_length`](../../indoorloc/apps/pdr.py) | `apps.pdr` | 函数 | Kim et al. (2004): `L = k (mean \|a\|)^(1/3)` over the samples of each step. | — |
| [`step_lengths`](../../indoorloc/apps/pdr.py) | `apps.pdr` | 函数 | `(S,)` step lengths in metres. `model` is `"weinberg"`, `"kim"` or `"constant"` | — |
| [`weinberg_step_length`](../../indoorloc/apps/pdr.py) | `apps.pdr` | 函数 | Weinberg (2002): `L = k (a_max - a_min)^(1/4)` per step (accelerations in m/s^2). | — |
| [`PDRFusion`](../../indoorloc/apps/fusion.py) | `apps.fusion` | 类 | Particle-filter fusion of PDR steps with position fixes from any localizer. | F. Evennou, F. Marx, "Advanced integration of WiFi and inertial navigation systems for indoor mobile positioning", EURASIP J. Adv. Signal Process. 2006:086706, 2006. [doi:10.1155/ASP/2006/86706](https://doi.org/10.1155/ASP/2006/86706). (docstring 中另有 1 篇) |
| [`OnlineLocalizer`](../../indoorloc/apps/streaming.py) | `apps.streaming` | 类 | Online localization of a stream of scans with an optional tracker. | — |
| [`StreamEstimate`](../../indoorloc/apps/streaming.py) | `apps.streaming` | 类 | The output for one `(t, scan)` of the stream. | — |
| [`stack_estimates`](../../indoorloc/apps/streaming.py) | `apps.streaming` | 函数 | `(t (N,), Prediction)` from a sequence of stream estimates, for L4 evaluation | — |
| [`Instruction`](../../indoorloc/apps/navigation.py) | `apps.navigation` | 类 | One turn-by-turn instruction. | — |
| [`Navigator`](../../indoorloc/apps/navigation.py) | `apps.navigation` | 类 | Shortest walking routes on a (multi-floor) floor plan. | P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths", IEEE Trans. Systems Science and Cybernetics 4(2):100-107, 1968. [doi:10.1109/TSSC.1968.300136](https://doi.org/10.1109/TSSC.1968.300136). (docstring 中另有 1 篇) |
| [`Route`](../../indoorloc/apps/navigation.py) | `apps.navigation` | 类 | A route: `points` `(K, 2)` metric vertices, `floor` `(K,)`, `length` walked in the | — |
| [`astar`](../../indoorloc/apps/navigation.py) | `apps.navigation` | 函数 | Shortest path on a grid: `(K, 2)` int `(row, col)` cells from `start` to `goal` | — |
| [`path_length`](../../indoorloc/apps/navigation.py) | `apps.navigation` | 函数 | Length of a polyline `(K, D)` (sum of segment lengths; 0 for fewer than 2 points). | — |
| [`smooth_path`](../../indoorloc/apps/navigation.py) | `apps.navigation` | 函数 | Greedy line-of-sight post-smoothing: from each kept vertex jump to the farthest later | — |
| [`turn_instructions`](../../indoorloc/apps/navigation.py) | `apps.navigation` | 函数 | Turn-by-turn instructions for a polyline `(K, 2)` (optionally with `(K,)` floors). | — |
<!-- /catalog:apps -->

## 跟踪定位结果

仿真办公楼提供带 `groups["trajectory"]` 和 `groups["time"]` 的行走轨迹，WiFi 每秒采样一次，测距每秒五次。
本页的所有数字都来自这一仿真。

```python
import numpy as np
import indoorloc as iloc
from indoorloc.apps import ExtendedKalmanTracker, KalmanTracker

train, test = iloc.load_dataset("synthetic_office")
model = iloc.create_model("wknn", preprocess=iloc.FillMissing(-104)).fit(train)
walks = iloc.load_dataset("synthetic_office", split="trajectory")
walk = walks[walks.groups["trajectory"] == 0]
fixes = model.localize(walk)                                   # 每次扫描一个 WKNN 定位结果
kf = KalmanTracker(motion="cv", process_noise=0.5)
filtered = kf.filter(fixes, t=walk.groups["time"])             # 因果滤波
smoothed = kf.smooth(fixes, t=walk.groups["time"])             # 再加 Rauch-Tung-Striebel 反向平滑
print([round(iloc.evaluate(walk, p).mean_error, 2) for p in (fixes, filtered, smoothed)])
# [2.97, 2.53, 2.11]
```

`update` 逐个处理定位结果，运行的是同一个滤波器：

```python
kf = KalmanTracker().reset(0.0)
for t, z, s in zip(walk.groups["time"], fixes.pos, fixes.spread):
    kf.update(t, z, spread=s)
pos, spread = kf.estimate()
print(np.round(pos, 2), round(float(spread), 2), np.round(walk.pos[-1], 2))
# [25.49  9.17] 2.88 [24.87  9.25]
```

`predict_to(t)` 在没有定位结果时把跟踪器外推到绝对时间 `t`。粒子滤波器使用同一个接口，在两次定位之间做随机游走：

```python
from indoorloc.apps import ParticleFilter

pf = ParticleFilter(500, random_state=0)
for t, z, s in zip(walk.groups["time"], fixes.pos, fixes.spread):
    pf.update(t, z, spread=s)
t_end = walk.groups["time"][-1]                                # 绝对时间，不是增量
print(np.round(kf.predict_to(t_end + 2.0), 2), np.round(pf.predict_to(t_end + 2.0), 2))
# [28.92 12.88] [22.79  6.87]
```

Kalman 跟踪器沿速度向前外推；粒子滤波器的随机游走让均值大致留在原处，只让粒子云变宽。

扩展 Kalman 跟踪器直接使用到已知锚点的原始测距：

```python
ranges = iloc.load_dataset("synthetic_office", split="trajectory", modality="ranges")
r_walk = ranges[ranges.groups["trajectory"] == 0]
ekf = ExtendedKalmanTracker(r_walk.meta["anchors"], range_std=0.1)
per_epoch = iloc.create_model("trilateration", anchors=r_walk.meta["anchors"]).fit(r_walk)
print(round(iloc.evaluate(r_walk, per_epoch.localize(r_walk)).mean_error, 3),
      round(iloc.evaluate(r_walk, ekf.smooth(r_walk.X, t=r_walk.groups["time"])).mean_error, 3))
# 0.564 0.314
```

## 流式定位

`OnlineLocalizer` 对 `(t, scan)` 流逐个定位，可选配一个跟踪器。为 `None` 或读数少于 `min_readings` 的扫描被视为缺失；每次扫描的处理耗时都会被记录：

```python
from indoorloc.apps import OnlineLocalizer

online = OnlineLocalizer(model, tracker=KalmanTracker(), floor_window=3)
stream = zip(walk.groups["time"], walk.X)                      # 任何 (t, scan) 可迭代对象
estimates = list(online.run(stream))
last = estimates[-1]
print(len(estimates), np.round(last.pos, 2), last.floor, last.missing, sorted(online.latency_stats())[:3])
# 60 [25.49  9.17] 0 False ['max_ms', 'mean_ms', 'n']
```

## PDR 与地图约束融合

`PDR` 在 IMU 表中检测步伐，估计步长（Weinberg 或 Kim 模型）并确定方向（陀螺仪、罗盘、二者的互补融合，或给定的偏航角）。
`PDRFusion` 用粒子滤波把步伐与任何定位器的结果融合。平面图取自 `meta["floor_plan"]`，一步穿墙的粒子会被移除。
二者都不返回单独的 `Prediction`：`PDR.run` 返回 `StepTrack`（每步一项），`PDRFusion.run` 返回 `(t, Prediction)`（每个步或定位结果一行），
因此示例把两者都与在各自时间插值得到的真实路径比较。

```python
from indoorloc.apps import PDR, FloorMap, ParticleFilter, PDRFusion

imu = iloc.load_dataset("synthetic_office", split="trajectory", modality="imu")    # 50 Hz 加速度计 + 陀螺仪
floor_map = FloorMap.from_dict(walks.meta["floor_plan"])
template = ParticleFilter(1000, step_length_std=0.1, heading_std=0.05, heading_drift_std=0.005,
                          motion_std=0.0, recovery=(0.05, 0.5), floor_map=floor_map, floor=0)
errors = {"WKNN fixes": [], "PDR": [], "fused": []}
for k in range(4):
    steps_in = imu[imu.groups["trajectory"] == k]
    scans = walks[walks.groups["trajectory"] == k]
    t_imu = steps_in.groups["time"]
    first = steps_in.pos[50] - steps_in.pos[0]                  # 陀螺仪需要初始航向
    track = PDR(initial_heading=np.arctan2(first[1], first[0])).run(steps_in, start=steps_in.pos[0])
    fused_t, fused = PDRFusion(template, random_state=0).run(track, model.localize(scans), scans.groups["time"])
    truth = lambda t: np.column_stack([np.interp(t, t_imu, steps_in.pos[:, d]) for d in (0, 1)])  # noqa: E731
    errors["WKNN fixes"].append(iloc.evaluate(scans, model.localize(scans)).mean_error)
    errors["PDR"].append(iloc.evaluate(truth(track.t), track.pos).mean_error)
    errors["fused"].append(iloc.evaluate(truth(fused_t), fused.pos).mean_error)
print({name: round(float(np.mean(e)), 2) for name, e in errors.items()})
# {'WKNN fixes': 2.8, 'PDR': 3.89, 'fused': 2.11}
```

上面的 `template` 就是 `PDRFusion` 的默认滤波器（步长与航向噪声、步与步之间不做随机游走、增强 MCL 恢复）再加上平面图。
直接写 `ParticleFilter(n, floor_map=...)` 会沿用 `ParticleFilter` 自己的默认值，这些默认值是为随机游走跟踪设计的：
在这四条轨迹上，用 500 个粒子得到的融合误差为 9.62 m。加入地图时，请从上面的模板开始。

## 导航

`Navigator` 把 `FloorMap` 栅格化，与墙体保持一定距离，并返回 A* 路线（可经楼梯、电梯跨楼层）和逐段转向指引：

```python
from indoorloc.apps import Navigator

nav = Navigator(resolution=0.25, clearance=0.2).fit(floor_map)
route = nav.route((2.0, 2.0), (36.0, 17.0))
print(round(route.length, 1), [step.text for step in route.instructions][:2])
# 47.2 ['Start and walk 7.2 m', 'Turn right, then walk 33.5 m']
```

## 实测数据

`tests/apps/test_apps_realdata.py` 在 ILC 2020 样本（两座购物中心的真实手机轨迹）上运行 L5 组件：步长、
从已知起点出发并使用手机航向的 PDR、倾斜补偿罗盘与手机姿态的比较、平面图与实际行走路径的一致性，以及 WiFi 跟踪与 PDR 融合。
这些测试需要 `ilc2020` 的文件，缺少时自动跳过。`apps.pdr.imu_arrays` 与 `FloorMap.from_dict(table.meta["floor_plan"])`
把加载器输出的表接到上述各类上。
