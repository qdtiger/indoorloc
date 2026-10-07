# L2 信号

`indoorloc.signals` 提供预处理与信号表示：遵循 scikit-learn 变换器契约的 fit/transform 类，以及针对 RSSI、CSI、测距、
IMU、磁力计和可见光信号的纯函数。它只导入 numpy。

[指南首页](index.md) · [English](../guide/signals.md)

## 变换契约

* 变换接受单次扫描 `(F,)`、批量 `(N, ...)`、`SampleTable` 或扫描视图（`WiFiSignal`、`BLESignal`），并返回同类对象。
  表的坐标、标签、分组和 id 保持不变。
* 统计量在 `fit` 中只从训练数据学习，之后冻结，`transform` 把它们用于任何数据。`fit_transform(train)` 一步完成拟合与变换。
  学到的属性以 `_` 结尾（例如 `min_dbm_`）。
* 改变列的变换（选 AP、选子载波、CSI 转幅度）会让 `meta["feature_names"]` 和 `meta["subcarriers"]` 与新的列保持一致。
* 数据增强接受 `random_state`，从 `np.random.default_rng(random_state)` 取随机数。
* `Compose([...])` 串联变换；变换也可以放进 `sklearn.pipeline.Pipeline`。

```python
import numpy as np
import indoorloc as iloc
from indoorloc.signals import Compose, ExponentialRepresentation, FillMissing, WiFiSignal

train, test = iloc.load_dataset("synthetic_office")
prep = Compose([FillMissing(-104), ExponentialRepresentation()])
train_x = prep.fit_transform(train)            # 学到 min_dbm_ = 训练数据最小值 - 1 dBm
test_x = prep.transform(test)                  # 使用同样冻结的统计量
print(type(test_x).__name__, prep.transforms[1].min_dbm_, np.round(test_x.X.max(), 3))
# SampleTable -105.0 0.535
scan = WiFiSignal(test.X[0], ap_ids=test.meta["feature_names"])
print(prep.transform(scan).rssi.round(2))      # 输入一次扫描，输出一次扫描
# [0.15 0.13 0.11 0.05 0.14 0.26 0.11 0.08]
```

### 预处理属于模型

把变换传给 `create_model(..., preprocess=...)`（即 `LocalizerPipeline`）。这样它在模型拟合时只在训练数据上拟合，随模型一起保存，
并在预测时作用于原始扫描。预处理要么放在模型里，要么放在模型之前，不要两处都放：变换被应用两次会改变数据。

```python
model = iloc.create_model("wknn", k=5, preprocess=Compose([FillMissing(-104), ExponentialRepresentation()]))
print(round(model.fit(train).evaluate(test).mean_error, 4))
# 1.9703
```

在这座仿真办公楼上，指数表示比单纯填补更差（后者为 1.8055 m，见[上手教程](index.md#五分钟上手)）。前面的填补可有可无：`FillMissing(-104)` 之后训练数据的最小值就是填补值，所以 `min_dbm_` 为 -105 dBm；单独使用 `ExponentialRepresentation()` 时，它从最弱的实际读数学到 -95 dBm，并把缺失读数映射为 `exp(min / alpha)`。两者在这里都得到 1.9703 m；基准测试中的单元格只用了该表示本身。

[docs/benchmarks_zh.md](../benchmarks_zh.md) 的基准表在 15 张真实数据表上比较了 -104 dBm 填补与 Torres-Sospedra 等（2015）的
positive、exponential、powed 三种表示；那里实测指数表示在 15 张表中的 12 张上降低了 WKNN 的平均误差。

## 目录

表中“作用”一栏取自英文 docstring。

<!-- catalog:transforms -->
| 类 | 模块 | 作用（取自 docstring） | `fit` 学习统计量 | 参考文献 |
| --- | --- | --- | --- | --- |
| [`APFilter`](../../indoorloc/signals/transforms.py) | `signals.transforms` | Treat weak readings as not heard: RSSI below `threshold_dbm` becomes NaN. | 否 | J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems", Expert Systems with Applications 42(23):9263-9278, 2015. <https://doi.org/10.1016/j.eswa.2015.08.013> |
| [`APSelect`](../../indoorloc/signals/transforms.py) | `signals.transforms` | Keep the `k` most useful access points (columns), chosen on the training data. | 是 | M. Youssef, A. Agrawala, A. U. Shankar, "WLAN location determination via clustering and probability distributions", IEEE PerCom 2003, pp. 143-150. <https://doi.org/10.1109/PERCOM.2003.1192736> (docstring 中另有 2 篇) |
| [`Compose`](../../indoorloc/signals/transforms.py) | `signals.transforms` | Apply transforms in order (torchvision's name for an sklearn Pipeline of transformers). | 取决于参数 | L. Buitinck et al., "API design for machine learning software: experiences from the scikit-learn project", ECML PKDD Workshop on Languages for Data Mining and Machine Learning, 2013, pp. 108-122. <https://arxiv.org/abs/1309.0238> |
| [`ExponentialRepresentation`](../../indoorloc/signals/transforms.py) | `signals.transforms` | Exponential RSSI representation: `exp(positive / alpha) / exp(-min / alpha)`. | 取决于参数 | J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems", Expert Systems with Applications 42(23):9263-9278, 2015. <https://doi.org/10.1016/j.eswa.2015.08.013> |
| [`FillMissing`](../../indoorloc/signals/transforms.py) | `signals.transforms` | Replace missing readings (NaN, or `missing=` e.g. 100) with `value` dBm. | 否 | J. Torres-Sospedra, R. Montoliu, A. Martinez-Uso, J. P. Avariento, T. J. Arnau, M. Benedito-Bordonau, J. Huerta, "UJIIndoorLoc: a new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems", IPIN 2014, pp. 261-270 (weakest reading -104 dBm, "not detected" stored as 100). <https://doi.org/10.1109/IPIN.2014.7275492> |
| [`HampelFilter`](../../indoorloc/signals/transforms.py) | `signals.transforms` | Moving-window Hampel filter: outliers along `axis` are replaced by the window median. | 否 | F. R. Hampel, "The influence curve and its role in robust estimation", Journal of the American Statistical Association 69(346):383-393, 1974. <https://doi.org/10.1080/01621459.1974.10482962> (docstring 中另有 2 篇) |
| [`PositiveRepresentation`](../../indoorloc/signals/transforms.py) | `signals.transforms` | Positive RSSI representation: `RSSI - min` for heard readings, 0 otherwise. | 取决于参数 | J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems", Expert Systems with Applications 42(23):9263-9278, 2015. <https://doi.org/10.1016/j.eswa.2015.08.013> (docstring 中另有 1 篇) |
| [`PowedRepresentation`](../../indoorloc/signals/transforms.py) | `signals.transforms` | Powed RSSI representation: `positive ** beta / (-min) ** beta`. | 取决于参数 | J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems", Expert Systems with Applications 42(23):9263-9278, 2015. <https://doi.org/10.1016/j.eswa.2015.08.013> |
| [`RSSINormalize`](../../indoorloc/signals/transforms.py) | `signals.transforms` | Min-max scale RSSI to [0, 1]. `lo`/`hi` = None learns them from the training data. | 取决于参数 | J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems", Expert Systems with Applications 42(23):9263-9278, 2015. <https://doi.org/10.1016/j.eswa.2015.08.013> |
| [`APDropout`](../../indoorloc/signals/augment.py) | `signals.augment` | Drop each heard reading independently with probability `p` (it becomes NaN). | 否 | N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov, "Dropout: a simple way to prevent neural networks from overfitting", Journal of Machine Learning Research 15(56): 1929-1958, 2014. <https://jmlr.org/papers/v15/srivastava14a.html> (docstring 中另有 1 篇) |
| [`GaussianNoise`](../../indoorloc/signals/augment.py) | `signals.augment` | Add zero-mean Gaussian noise of `std_db` dB to every heard reading (NaN stays NaN). | 否 | T. S. Rappaport, "Wireless Communications: Principles and Practice", 2nd ed., Prentice Hall, 2002, ch. 4 (log-normal shadowing). (docstring 中另有 1 篇) |
| [`DeviceCalibration`](../../indoorloc/signals/calibration.py) | `signals.calibration` | Learn `reference ~ g(device)` and apply `g` to the device's readings. | 是 | A. Haeberlen, E. Flannery, A. M. Ladd, A. Rudys, D. S. Wallach, L. E. Kavraki, "Practical robust localization over large-scale 802.11 wireless networks", ACM MobiCom 2004, pp. 70-84. <https://doi.org/10.1145/1023720.1023728> (docstring 中另有 4 篇) |
| [`CSIAmplitude`](../../indoorloc/signals/csi.py) | `signals.csi` | CSI amplitude `\|H\|`, or `20 log10 \|H\|` dB with `db=True` (zero amplitude -> NaN). | 否 | X. Wang, L. Gao, S. Mao, S. Pandey, "CSI-based fingerprinting for indoor localization: a deep learning approach", IEEE Transactions on Vehicular Technology 66(1):763-776, 2017. <https://doi.org/10.1109/TVT.2016.2545523> |
| [`CSIPhaseSanitize`](../../indoorloc/signals/csi.py) | `signals.csi` | Unwrap CSI phase across subcarriers and remove its linear trend (STO/CFO offsets). | 否 | S. Sen, B. Radunovic, R. R. Choudhury, T. Minka, "You are facing the Mona Lisa: spot localization using PHY layer information", ACM MobiSys 2012, pp. 183-196. <https://doi.org/10.1145/2307636.2307654> (docstring 中另有 2 篇) |
| [`SubcarrierSelect`](../../indoorloc/signals/csi.py) | `signals.csi` | Keep the subcarriers at positions `indices` of the last axis, in the given order. | 否 | D. Halperin, W. Hu, A. Sheth, D. Wetherall, "Tool release: gathering 802.11n traces with channel state information", ACM SIGCOMM Computer Communication Review 41(1):53, 2011 (the 30 grouped subcarriers of `INTEL5300_SUBCARRIERS_20MHZ`). <https://doi.org/10.1145/1925861.1925870> |
| [`MagneticFeatures`](../../indoorloc/signals/magnetic.py) | `signals.magnetic` | Per-sample magnetic fingerprint features `[B, B_h, B_v]` (the `magnetic` layout). | 否 | B. Li, T. Gallagher, A. G. Dempster, C. Rizos, "How feasible is the use of magnetic field alone for indoor positioning?", IPIN 2012. <https://doi.org/10.1109/IPIN.2012.6418880> |
| [`MagnetometerCalibration`](../../indoorloc/signals/magnetic.py) | `signals.magnetic` | Hard-iron and soft-iron calibration learned from readings in many orientations. | 是 | J. F. Vasconcelos, G. Elkaim, C. Silvestre, P. Oliveira, B. Cardeira, "Geometric approach to strapdown magnetometer calibration in sensor frame", IEEE Transactions on Aerospace and Electronic Systems 47(2):1293-1306, 2011. <https://doi.org/10.1109/TAES.2011.5751259> (docstring 中另有 1 篇) |
<!-- /catalog:transforms -->

扫描视图（一次扫描及其 AP 或信标 id；`WiFiSignal.from_raw(row, missing=100)` 转换原始文件中的一行）：

<!-- catalog:views -->
| 类 | 作用（取自 docstring） |
| --- | --- |
| [`WiFiSignal`](../../indoorloc/signals/wifi.py) | One RSSI scan: a (F,) array in dBm (NaN = not heard) plus optional AP ids. |
| [`BLESignal`](../../indoorloc/signals/ble.py) | One BLE scan: a (F,) array in dBm (NaN = not heard) plus optional beacon ids. |
<!-- /catalog:views -->

### 函数

<!-- catalog:signal-functions -->
**`indoorloc.signals.functional`**: Pure RSSI functions. Each takes one signal `(F,)` or a batch `(N, F)`.

| 函数 | 作用（取自 docstring） |
| --- | --- |
| `aggregate_rssi` | Combine several scans of the same transmitters into one reading per transmitter. |
| `aggregate_rssi_groups` | Aggregate the rows of `x` (N, F) that share a key, e.g. repeated scans at one |
| `dbm_to_mw` | Power in milliwatts: `10 ** (dBm / 10)`. NaN stays NaN. |
| `exponential` | Exponential representation: `exp(positive / alpha) / exp(-min / alpha)`. |
| `fill_missing` | Replace missing readings with `value` (default -104 dBm). |
| `hampel` | Hampel identifier: replace outliers by the median of their moving window. |
| `minmax_scale` | `(x - lo) / (hi - lo)`: maps [lo, hi] dBm to [0, 1]; NaN stays NaN. |
| `missing_mask` | True where a reading is missing: NaN, or a file sentinel such as UJIIndoorLoc's 100. |
| `mw_to_dbm` | Power in dBm: `10 log10(mW)`. Zero or negative power (nothing received) is NaN. |
| `positive` | Positive representation (Torres-Sospedra et al., 2015): `RSSI - min` for heard |
| `powed` | Powed representation: `positive ** beta / (-min) ** beta`. |

**`indoorloc.signals.csi`**: WiFi channel state information (CSI): pure functions and fit/transform classes.

| 函数 | 作用（取自 docstring） |
| --- | --- |
| `amplitude` | `\|H\|` (float32 for complex64 input), or `20 log10 \|H\|` in dB with `db=True`. |
| `conjugate_multiply` | `H * conj(H_ref)`: each antenna times the conjugate of antenna `ref` along `axis`. |
| `csi_ratio` | `H / H_ref` along `axis`: cancels common phase offsets and common amplitude noise |
| `phase` | Phase in radians; `unwrap=True` unwraps it across subcarriers (last axis). |
| `sanitize_phase` | Remove the linear phase error of CSI across subcarriers. |

**`indoorloc.signals.ranging`**: Ranging helpers: times and signal strength to metres, bias calibration, simple NLOS flags.

| 函数 | 作用（取自 docstring） |
| --- | --- |
| `correct_range_bias` | `(measured - offset) / scale`: the ranges with the fitted linear bias removed. |
| `distance_to_rssi` | Log-distance path loss model: `p0 - 10 n log10(d / d0)` dBm (`p0` at `d0`). |
| `ds_twr_distance` | `speed * ds_twr_tof(...)` metres (asymmetric double-sided two-way ranging). |
| `ds_twr_tof` | Asymmetric double-sided two-way ranging: time of flight in seconds. |
| `fit_range_bias` | Least-squares linear error model `measured ~ scale * true + offset`. |
| `nlos_flags_geometry` | Flag ranges that violate the triangle inequality between anchors. |
| `nlos_flags_power` | Flag NLOS when the total received power exceeds the first-path power by more than |
| `nlos_flags_std` | Flag NLOS when the range jitter over the last `window` samples exceeds `threshold_m`. |
| `rf_ultrasound_distance` | Distance from the arrival-time difference of a simultaneous RF and ultrasound pulse. |
| `rssi_to_distance` | Inverse of the log-distance model: `d0 * 10 ** ((p0 - rssi) / (10 n))` metres. |
| `rtt_to_distance` | Round-trip time (s) -> distance (m): `c * (rtt - turnaround) / 2`. |
| `speed_of_sound` | Speed of sound in dry air, `331.3 sqrt(1 + T / 273.15)` m/s at `T` degrees Celsius. |
| `toa_to_distance` | One-way time of flight (s) -> distance (m): `c * t`. |
| `twr_distance` | `speed * twr_tof(round, reply)` metres (single-sided two-way ranging). |
| `twr_tof` | Single-sided two-way ranging: time of flight `(T_round - T_reply) / 2` seconds. |
| `ultrasound_tof_to_distance` | One-way ultrasonic time of flight (s) -> distance (m): `speed_of_sound(T) * t`. |

**`indoorloc.signals.imu`**: Inertial (IMU) helpers for one time series: magnitude, smoothing, gravity removal.

| 函数 | 作用（取自 docstring） |
| --- | --- |
| `low_pass` | First-order IIR low-pass (exponential smoothing) along `axis`: |
| `magnitude` | Euclidean norm along `axis`, e.g. `(T, 3)` accelerometer -> `(T,)` in m/s^2. |
| `moving_average` | Mean over `window` samples along `axis`; NaN ignored, windows truncated at the edges. |
| `remove_gravity` | Split accelerometer data `(T, 3)` into linear acceleration and gravity. |
| `smoothing_factor` | The `alpha` of a first-order RC low-pass: `dt / (RC + dt)`, `RC = 1 / (2 pi f_c)`. |

**`indoorloc.signals.magnetic`**: Magnetometer science: field components, tilt-compensated heading, hard/soft-iron calibration.

| 函数 | 作用（取自 docstring） |
| --- | --- |
| `apply_calibration` | Calibrated readings `W (m - h)` (`(N, 3)`, or `(3,)` for one reading). |
| `field_components` | Horizontal intensity and vertical component `(B_h, B_v)` of the field. |
| `field_magnitude` | Total intensity `\|b\|` of `(N, 3)` readings (`(N,)`; a scalar for one reading). |
| `fit_ellipsoid` | Hard-iron offset and soft-iron correction from readings taken in many orientations. |
| `heading` | Tilt-compensated compass heading of the device `forward` axis, radians in `[-pi, pi)`. |
| `inclination` | Magnetic inclination (dip) in radians, `atan2(-B_v, B_h)`: positive when the field |
| `magnetic_features` | `(N, 3)` float64 `[B, B_h, B_v]`: the `magnetic` modality layout (module docstring). |
| `wrap_angle` | Wrap radians to `[-pi, pi)`. |

**`indoorloc.signals.vlc`**: Visible light positioning: the Lambertian line-of-sight channel, its inversion and receiver noise.

| 函数 | 作用（取自 docstring） |
| --- | --- |
| `channel_gain` | Line-of-sight DC gain `H` `(N, A)` from `A` LEDs to receivers at `positions` `(N, 3)`. |
| `concentrator_gain` | Gain `n^2 / sin^2(FOV)` of an ideal non-imaging concentrator (Kahn & Barry 1997). |
| `distance_to_power` | `P_r = C h^(m+1) / d^(m+3)` for an LED facing down `height` metres above a receiver |
| `half_power_angle` | Inverse of `lambertian_order`: `arccos(2^(-1/m))` radians. |
| `lambertian_order` | `m = -ln 2 / ln cos(Phi_1/2)` for a half-power semi-angle in radians, `0 < Phi < pi/2`. |
| `noise_variance` | Receiver noise variance (A^2) at received optical `power` (W): shot plus thermal noise. |
| `power_to_distance` | Distance from received power for an LED facing down `height` metres above a receiver |
| `received_power` | Received optical power `P_t H` `(N, A)` in watts; `tx_power` scalar or `(A,)`; |
<!-- /catalog:signal-functions -->

## 数据增强

数据增强只用于扰动训练数据。每次调用已拟合的增强都会生成新的扰动，`random_state` 保证序列可复现。

```python
from indoorloc.signals import APDropout, GaussianNoise

noisy = GaussianNoise(std_db=2.0, random_state=0)(train)       # 听到的读数 + N(0, 2^2) dB
dropped = APDropout(p=0.1, random_state=0)(train)              # 10 % 的读数变为 NaN
print(np.isnan(train.X).sum(), np.isnan(noisy.X).sum(), np.isnan(dropped.X).sum())
# 1 1 709
```

## CSI

`csi` 表保存复数 CSI `(N, n_rx, n_tx, n_sub)`。`CSIAmplitude` 输出 `csi_amp` 表（线性幅度，或 `db=True` 时的 dB），
`CSIPhaseSanitize(output="phase")` 输出 `csi_phase` 表：在 `meta["subcarriers"]` 列出的子载波上去掉相位的线性趋势
（采样时刻与频率偏移）；`SubcarrierSelect` 保留选定的子载波。

```python
# data: haloc
from indoorloc.signals import CSIAmplitude, CSIPhaseSanitize, SubcarrierSelect

csi = iloc.load_dataset("haloc", split="test")                 # ESP32 CSI，52 个 L-LTF 子载波
phase = CSIPhaseSanitize(output="phase").fit_transform(csi)
amp = Compose([CSIAmplitude(), SubcarrierSelect(np.arange(0, 52, 2))]).fit_transform(csi)
print(csi.X.shape, csi.X.dtype, phase.meta["modality"], amp.X.shape, amp.meta["modality"])
# (14277, 1, 1, 52) complex64 csi_phase (14277, 1, 1, 26) csi_amp
```

对 CSI，方法需要实数特征。基准中的 HALOC 与 H-WILD 单元格就是以 `CSIAmplitude` 作为 `preprocess=` 运行的：

```python
from indoorloc.signals import CSIAmplitude

csi_train, csi_test = iloc.load_dataset("synthetic_office", modality="csi")   # 仿真 CSI
model = iloc.create_model("wknn", preprocess=CSIAmplitude(db=True)).fit(csi_train)
print(csi_train.X.shape, round(model.evaluate(csi_test).mean_error, 3))
# (830, 12, 1, 56) 3.667
```

## 测距、IMU、磁力计与可见光

这些都是数组函数；对应的方法在 [L3](methods.md)，IMU 应用（步数、航向、PDR）在 [L5](apps.md)。

```python
from indoorloc.signals import magnetic, ranging, vlc

print(ranging.rtt_to_distance(100e-9).round(3))                        # 由往返时间得到的距离（米）
# 14.99
print(ranging.rssi_to_distance(-70.0, p0_dbm=-40.0, n=3.0).round(2))  # 对数距离路径损耗
# 10.0
print(ranging.speed_of_sound(20.0).round(2), ranging.ultrasound_tof_to_distance(0.01, 20.0).round(4))
# 343.21 3.4321

mag = np.array([[0.0, 20.0, -40.0]])        # uT，设备水平（z 朝上），y 轴指向磁北
print(magnetic.magnetic_features(mag), magnetic.heading(mag).round(4))   # [B, B_h, B_v]；从正东逆时针的弧度
# [[ 44.72135955  20.         -40.        ]] [1.5708]

leds = np.array([[1.0, 1.0, 3.0], [3.0, 1.0, 3.0], [2.0, 3.0, 3.0]])    # 朝下的 LED，高 3 m
power = vlc.received_power(np.array([[2.0, 1.5, 0.8]]), leds)           # 0.8 m 高、朝上的接收器
print(vlc.power_to_distance(power, 2.2, horizontal=True).round(4))      # 反算水平距离
# [[1.118 1.118 1.5  ]]
```
