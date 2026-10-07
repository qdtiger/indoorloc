# L1 datasets

`indoorloc.datasets` turns public datasets and simulators into `SampleTable`s of numpy arrays.
It imports only numpy and the standard library. The loaders for `.mat` and `.h5` files use
scipy or h5py, which are imported inside the loader (extra `[datasets]`).

[Guide home](index.md) · [中文](../zh/datasets.md)

## Loading

```text
load_dataset(name, split=None, *, root=None, download=True, verify=True, **options)
```

* `split=None` returns the official `(train, test)` pair when the dataset has both splits, and
  otherwise its single table (`"all"`). Pass `split="test"` for one table or a tuple of split
  names for several. Aliases such as `"validation"` resolve to a dataset's own split name.
* Files are cached under `$INDOORLOC_DATA/<name>`, which defaults to
  `~/.cache/indoorloc/datasets/<name>`. Pass `root=` to use another folder. A missing file is
  downloaded unless `download=False`. Every file is checked against the sha256 recorded in the
  loader unless `verify=False`, and each table records the digests in `meta["sha256"]`.
* `options` are dataset-specific (a building, a month, a room, a modality, a seed); see the
  options table below.
* `list_datasets()` returns the registry names, including those added with `register_dataset(name,
  cls)` ([Extending](extending.md#a-dataset)); `load_dataset` also takes a `"package.module:Class"`
  path or a `Dataset` subclass. `dataset_info(name)` returns the class-level facts
  (modality, frame, license, DOI, splits) without loading anything. `indoorloc info <name>` also
  lists the files and whether they are on disk (`--json` adds their sha256).

```python
import indoorloc as iloc
from indoorloc.datasets import dataset_info, list_datasets

print(len(list_datasets()), "ujiindoorloc" in list_datasets())
# 15 True
info = dataset_info("tuji1")
print(info["modality"], info["crs"], info["license"], info["splits"])
# wifi_rssi local CC BY 4.0 ('train', 'test')
```

## What a table holds

| Field | Content |
| --- | --- |
| `X` | `(N, ...)` observations in physical units: dBm for RSSI, complex CSI, metres for ranges, radians for angles. A missing reading is **NaN**. The loaders convert each file's sentinel (100 in UJIIndoorLoc, -200 in the UCI BLE file) to NaN and record the original value in `meta["raw_missing_value"]`. Nothing is normalized. |
| `pos` | float64 `(N, D)` coordinates in the dataset's own frame, never rescaled. `meta["crs"]` names the frame and `meta["pos_units"]` the unit. |
| `floor`, `building` | int64 `(N,)` or `None` when the dataset has no such labels. Negative floors are real floors (`B1` = -1). |
| `groups` | columns for grouped splits: `user`, `device`, `time`, `trajectory`, `month`, `point`, `room`, ... A column whose values are unknown for a split is left out and listed in `meta["unknown_groups"]`. |
| `ids` | stable sample ids. |
| `meta` | `name`, `split`, `sha256`, `source_files`, `modality`, `units`, `crs`, `pos_units`, `feature_names`, `license`, `doi`, `citation` and dataset-specific facts such as `ground_scale`, `anchors` or `floor_plan`. |

The `X` layout of every modality (RSSI, CSI, ranges, TDoA, AoA, IMU) is specified in
[CONTRACTS.md §2](../architecture/CONTRACTS.md#2-the-two-data-types-indoorloccore).

## Catalog

Generated from the registry by `python docs/catalog.py`. Row counts are those recorded when
the benchmark matrix loaded the real files ([docs/benchmarks.md](../benchmarks.md)). The
simulator's counts come from generating its default tables, and the ILC 2020 count from its
file manifest. `ilc2020` reads the sample traces that the organizers published with the
competition's code (two shopping malls), not the full competition data.

<!-- catalog:datasets -->
**WiFi RSSI**

| Name | Modality | Rows | X units | Positions | License | Source |
| --- | --- | --- | --- | --- | --- | --- |
| [`longtermwifi`](../../indoorloc/datasets/longtermwifi.py) | `wifi_rssi` | train 23,040 · test 81,120 | dBm | m | CC BY 4.0 (data, Readme.txt); MIT (scripts) | Mendoza-Silva et al., Long-Term WiFi Fingerprinting Dataset for Research on Robust Indoor Positioning, Data 3(1):3, 2018, doi:10.3390/data3010003. [doi:10.5281/zenodo.3748719](https://doi.org/10.5281/zenodo.3748719) |
| [`sodindoorloc`](../../indoorloc/datasets/sodindoorloc.py)<br>alias: `sod` | `wifi_rssi` | train 21,205 · test 2,720 | dBm | m; crs `local-per-building` | not stated in the repository; cite the paper | Bi et al., Supplementary open dataset for WiFi indoor localization based on received signal strength, Satellite Navigation 3:25, 2022. [doi:10.1186/s43020-022-00086-y](https://doi.org/10.1186/s43020-022-00086-y) |
| [`tampere`](../../indoorloc/datasets/tampere.py) | `wifi_rssi` | train 697 · test 3,951 | dBm | m | CC BY 4.0 (data, FINGERPRINTING_DB/README.txt); MIT (software) | Lohan et al., Wi-Fi Crowdsourced Fingerprinting Dataset for Indoor Positioning, Data 2(4):32, 2017, doi:10.3390/data2040032. [doi:10.5281/zenodo.889798](https://doi.org/10.5281/zenodo.889798) |
| [`tuji1`](../../indoorloc/datasets/tuji1.py) | `wifi_rssi` | train 6,752 · test 2,147 | dBm | m | CC BY 4.0 | Klus et al., TUJI1 Dataset: Multi-device dataset for indoor localization with high measurement density, Data in Brief 54:110356, 2024, doi:10.1016/j.dib.2024.110356. [doi:10.5281/zenodo.7641701](https://doi.org/10.5281/zenodo.7641701) |
| [`ujiindoorloc`](../../indoorloc/datasets/ujiindoorloc.py)<br>alias: `uji` | `wifi_rssi` | train 19,937 · test 1,111 | dBm | m (Web Mercator, not ground metres); crs `EPSG:3857` | CC BY 4.0 | Torres-Sospedra et al., UJIIndoorLoc, IPIN 2014. [doi:10.24432/C5MS59](https://doi.org/10.24432/C5MS59) |
| [`wlanrssi`](../../indoorloc/datasets/wlanrssi.py) | `wifi_rssi` | all 2,000 | dBm | — (room_classification) | CC BY 4.0 | Bhatt, Wireless Indoor Localization, UCI Machine Learning Repository, 2017, doi:10.24432/C51880. [doi:10.24432/C51880](https://doi.org/10.24432/C51880) |

**BLE RSSI**

| Name | Modality | Rows | X units | Positions | License | Source |
| --- | --- | --- | --- | --- | --- | --- |
| [`ble_indoor`](../../indoorloc/datasets/ble_indoor.py) | `ble_rssi` | room=office: train 22,237 · valid 3,598 · test 5,110<br>room=lab: train 13,238 · valid 2,686 · test 3,194 | dBm | m; crs `local (one frame per room; see building)` | MIT | Kennedy, Spachos, Taylor, BLE beacon indoor localization dataset, Scholars Portal Dataverse, 2019. [doi:10.5683/SP2/UTZTFT](https://doi.org/10.5683/SP2/UTZTFT) |
| [`ble_rssi_uci`](../../indoorloc/datasets/ble_rssi_uci.py) | `ble_rssi` | all 1,420 · unlabeled 5,191 | dBm | grid cells of the source map (column letter A=1, row number counted downward); cell size not stated | CC BY 4.0 | Mohammadi, Al-Fuqaha, Guizani, Oh, Semisupervised Deep Reinforcement Learning in Support of IoT and Smart City Services, IEEE Internet of Things Journal 5(2), 2018. [doi:10.24432/C54G80](https://doi.org/10.24432/C54G80) |
| [`ibeacon_rssi`](../../indoorloc/datasets/ibeacon_rssi.py) | `ble_rssi` | train 2,748 · test 1,860 · all 4,752 | dBm | m; crs `local (one frame per zone; see building)` | CC BY 4.0 (data), MIT (scripts) | Mendoza-Silva, Matey-Sanz, Torres-Sospedra, Huerta, BLE RSS Measurements Dataset for Research on Accurate Indoor Positioning, Data 4(1), 12, 2019. [doi:10.5281/zenodo.1618692](https://doi.org/10.5281/zenodo.1618692) |

**WiFi CSI**

| Name | Modality | Rows | X units | Positions | License | Source |
| --- | --- | --- | --- | --- | --- | --- |
| [`csi_fingerprint`](../../indoorloc/datasets/csi_fingerprint.py) | `csi_amp` | area=lab, packets=50: all 15,850<br>area=meeting, packets=50: all 8,800<br>area=conference, packets=50: all 8,000<br>area=minilab, packets=50: all 1,750 | dB | grid steps of the area's reference-point grid (spacing not stated by the source); crs `local grid (one per area; see building)` | MIT | Zhu, Qiu, Qu, Zhou, Atiquzzaman, Wu, BLS-Location: A Wireless Fingerprint Localization Algorithm Based on Broad Learning, IEEE TMC 22(1), 2023. [doi:10.1109/TMC.2021.3073005](https://doi.org/10.1109/TMC.2021.3073005) |
| [`haloc`](../../indoorloc/datasets/haloc.py) | `csi` | train 96,491 · valid 28,111 · test 14,277 · all 138,879 | raw int8 I/Q of ESP-IDF (not calibrated) | m | CC BY 4.0 (Zenodo record; the description asks for non-commercial research use) | Strohmayer, Kampel, WiFi CSI-based Long-Range Person Localization Using Directional Antennas, ICLR 2024 Tiny Papers. [doi:10.5281/zenodo.10715595](https://doi.org/10.5281/zenodo.10715595) |
| [`hwild`](../../indoorloc/datasets/hwild.py) | `csi` | environment=conference: all 22,970<br>environment=laboratory: all 26,833<br>environment=office: all 26,935<br>environment=lounge: all 42,554 | Intel 5300 scaled CSI as stored (not calibrated) | m; crs `local (one frame per room; see building)` | not stated by the repository (cite the RLoc paper) | Zhang, Zhang, Wang, Li, Hu, Sun, Chen, RLoc: Towards Robust Indoor Localization by Quantifying Uncertainty, Proc. ACM IMWUT 7(4), 2023. [doi:10.1145/3631437](https://doi.org/10.1145/3631437) |

**Multi-sensor traces**

| Name | Modality | Rows | X units | Positions | License | Source |
| --- | --- | --- | --- | --- | --- | --- |
| [`ilc2020`](../../indoorloc/datasets/ilc2020.py) | **`wifi_rssi`**, `ble_rssi`, `imu`, `waypoints` | 1,095 traces on 14 floors | dBm | m | MIT | Hu, Fan, Yin, Qian, Ji, Shu, Han, Xu, Liu, Bahl, The Wisdom of 1,170 Teams: Lessons and Experiences from a Large Indoor Localization Competition, ACM MobiCom 2023. [doi:10.1145/3570361.3592507](https://doi.org/10.1145/3570361.3592507) |

**Simulated**

| Name | Modality | Rows | X units | Positions | License | Source |
| --- | --- | --- | --- | --- | --- | --- |
| [`deepmimo`](../../indoorloc/datasets/simulated/deepmimo.py) | **`rssi`**, `ranges`, `tdoa`, `aoa`, `csi` | per scenario | — | m | per scenario (see deepmimo.net) | Alkhateeb, DeepMIMO: A generic deep learning dataset for millimeter wave and massive MIMO applications, ITA 2019. [arXiv:1902.06435](https://arxiv.org/abs/1902.06435) |
| [`synthetic_office`](../../indoorloc/datasets/simulated/office.py) | **`wifi_rssi`**, `ble_rssi`, `ranges`, `tdoa`, `aoa`, `csi`, `imu`, `vlc`, `magnetic` | default (seed 0): train 830 · test 200 · trajectory 240 | dBm | m | CC0-1.0 (generated data) | IndoorLoc SyntheticOffice simulator (indoorloc.datasets.simulated). <https://github.com/qdtiger/indoorloc> |

bold = the default `modality`. Rows as recorded by the benchmark runs (`benchmarks/results/*.json`).
<!-- /catalog:datasets -->

Splits and constructor options:

<!-- catalog:dataset-options -->
| Name | Splits (aliases) | Options (`load_dataset(name, **options)`) |
| --- | --- | --- |
| `ble_indoor` | `train`, `valid`, `test` (`validation`→`valid`, `val`→`valid`) | `room='all'` |
| `ble_rssi_uci` | `all`, `unlabeled` (`labeled`→`all`, `labelled`→`all`, `unlabelled`→`unlabeled`) | — |
| `csi_fingerprint` | `all` | `area='all'`, `packets=None` |
| `deepmimo` | `all` | `scenario='asu_campus_3p5'`, `modality='rssi'`, `tx_sets='all'`, `rx_sets='rx_only'`, `max_paths=25`, `dim=3`, `power_offset_db=0.0`, `sensitivity_dbm=None`, `coherent=False`, `n_antennas=8`, `spacing_wavelengths=0.5`, `orientation=0.0`, `subcarriers=<56 values>`, `subcarrier_spacing_hz=312500.0` |
| `haloc` | `train`, `valid`, `test`, `all` (`validation`→`valid`, `val`→`valid`) | `sequences=None`, `subcarriers='lltf'` |
| `hwild` | `all` | `environment='all'`, `users=None`, `interference=None`, `features='csi'` |
| `ibeacon_rssi` | `all`, `train`, `test` | `zone='all'`, `protocol=None` |
| `ilc2020` | `all` | `site='site1'`, `floor='F1'`, `modality='wifi'`, `outside_waypoints='drop'`, `wifi_max_age=2.0`, `ble_window=None` |
| `longtermwifi` | `train`, `test` | `month=None` |
| `sodindoorloc` | `train`, `test` | `building=None`, `macs='all'`, `averaged=False` |
| `synthetic_office` | `train`, `test`, `trajectory` (`trajectories`→`trajectory`, `walk`→`trajectory`) | `seed=0`, `modality='wifi_rssi'`, `n_floors=1`, `n_aps=None`, `grid_spacing=2.0`, `samples_per_point=5`, `n_test=200`, `n_trajectories=4`, `trajectory_duration=60.0`, `dim=2`, `size=(40.0, 20.0)`, `path_loss='multiwall'`, `noise_std=None`, `shadowing_std_db=None`, `n_antennas=4`, `n_scatterers=0`, `scan_rate_hz=None`, `imu_rate_hz=50.0`, `physics=None` |
| `tampere` | `train`, `test` | — |
| `tuji1` | `train`, `test` | — |
| `ujiindoorloc` | `train`, `test` (`validation`→`test`, `val`→`test`) | — |
| `wlanrssi` | `all` | — |
<!-- /catalog:dataset-options -->

## Coordinate frames

Every loader keeps the coordinates of the source and states their frame. Four cases need
attention:

* **UJIIndoorLoc** stores Web Mercator (EPSG:3857) easting and northing. One Mercator metre is
  `meta["ground_scale"]` = 0.7661 ground metres at the campus. Results on this dataset are
  normally published in Mercator metres; `evaluate(..., scale=meta["ground_scale"])` converts
  them to ground metres.
* **Per-building or per-room frames** (SODIndoorLoc, BBIL `ble_indoor`, iBeacon RSSI, H-WILD,
  the CSI fingerprint rooms, ILC 2020 floors). Positions from two buildings or rooms cannot be
  compared, so a position error means something only when the building (or room) is right.
* **Grid units.** `ble_rssi_uci` positions are map cells and `csi_fingerprint` positions are
  reference-grid steps; the sources do not state the cell size.
* **No coordinates.** `wlanrssi` holds room labels only (`groups["room"]`). `pos` has shape
  `(N, 0)`, so score the room label instead of a position.

```python
# data: ujiindoorloc
import indoorloc as iloc

train, test = iloc.load_dataset("ujiindoorloc")
print(train.X.shape, train.X.dtype, test.meta["crs"], round(test.meta["ground_scale"], 4))
# (19937, 520) float32 EPSG:3857 0.7661
print(sorted(test.groups), test.meta["unknown_groups"])
# ['device', 'time'] ('user', 'space', 'relative_position')
```

## Options and single-table datasets

```python
# data: sodindoorloc
train, test = iloc.load_dataset("sodindoorloc", building="HCXY")    # one of three buildings
print(len(train), len(test), train.X.shape[1], train.meta["crs"])
# 11370 860 347 local-per-building
```

A dataset without an official split loads as one table. The recommended split protocol is
given in its docstring and applied with [L4 protocols](evaluation.md#protocols):

```python
# data: ble_rssi_uci
from indoorloc.evaluation import get_protocol

table = iloc.load_dataset("ble_rssi_uci")       # the labelled file, split "all"
fold = get_protocol("random-80-20").folds(table, random_state=0)[0]
train, test = table[fold.train], table[fold.test]
print(len(table), len(train), len(test), table.meta["pos_units"].split(" (")[0])
# 1420 1136 284 grid cells of the source map
```

The ILC 2020 loader reads one site, one or more floors and one modality per table
(`"wifi"`, `"ble"`, `"imu"` or `"waypoints"`). Each table carries the floor plan in
`meta["floor_plan"]`:

```python
# data: ilc2020
wifi = iloc.load_dataset("ilc2020", site="site1", floor="F1", modality="wifi")
print(wifi.X.shape, sorted(wifi.groups), sorted(wifi.meta["floor_plan"])[:3])
# (2223, 2330) ['device', 'time', 'trajectory'] ['bounds', 'materials', 'wall_floor']
```

## The simulated office

`synthetic_office` builds a corridor office with walls, anchors on every storey and walks
through it, all from `seed` (license CC0). It produces every modality the library handles:
`wifi_rssi`, `ble_rssi`, `ranges` (UWB ToA), `tdoa`, `aoa`, `csi`, `imu`, `vlc` and `magnetic`.
The anchor geometry is in `meta["anchors"]` and the floor plan in `meta["floor_plan"]`.
It is meant for tests, tutorials and controlled experiments, and its results describe the
simulation, not a real building. With `modality="imu"` the only split is `trajectory`, which is
also what `split=None` loads.

```python
ranges = iloc.load_dataset("synthetic_office", split="test", modality="ranges", n_floors=2, seed=1)
print(ranges.X.shape, ranges.meta["anchors"].shape, ranges.meta["units"], sorted(set(ranges.floor.tolist())))
# (200, 8) (8, 2) m [0, 1]
walk = iloc.load_dataset("synthetic_office", modality="imu")     # split "trajectory"
print(walk.X.shape, walk.meta["rate_hz"], walk.meta["channels"][:3])
# (12000, 6) 50.0 ('acc_x', 'acc_y', 'acc_z')
```

`deepmimo` reads DeepMIMO v4 ray-tracing scenarios as RSSI, ranging, AoA or CSI tables. It needs
the `deepmimo` package (extra `[sim]`, Python 3.11 or later) and the scenario files.

## Using the data elsewhere

```python
import numpy as np

train, test = iloc.load_dataset("synthetic_office")
X, pos = train.to_numpy()                          # read-only views, no copy
first_floor = train[train.floor == 0]              # row subset: a mask, indices or a slice
filled = train.replace(X=np.nan_to_num(train.X, nan=-104.0))
other = iloc.load_dataset("synthetic_office", split="train", seed=1)
other = other.replace(ids=np.char.add("seed1-", other.ids.astype(str)))   # ids must stay unique
both = iloc.SampleTable.concat([train, other])
print(X.shape, len(first_floor), np.isnan(filled.X).any(), len(both))
# (830, 8) 830 False 1665
```

```python
# requires: pandas
df = train.to_dataframe()                          # features, coordinates, labels and groups
print(df.shape, list(df.columns[-5:]))
# (830, 14) ['y', 'floor', 'source', 'point', 'room']
```

```python
# requires: torch
from indoorloc.datasets.torch_adapter import make_dataloader

loader = make_dataloader(train, batch_size=256, shuffle=True, seed=0)
batch = next(iter(loader))
print(sorted(batch), tuple(batch["X"].shape), batch["X"].dtype)
# ['X', 'floor', 'groups.point', 'groups.room', 'groups.source', 'ids', 'pos'] (256, 8) torch.float32
```

## Plots

`indoorloc.datasets.plot` draws where the samples of one or more tables lie: one panel per floor
(or per building frame), a 3-D view with the floors stacked, or a density map, with the walls from
`meta["floor_plan"]` underneath. It needs matplotlib (extra `[plot]`). `distribution_html` writes
an interactive HTML page with plotly, which the same extra installs.

```python
# requires: matplotlib
from indoorloc.datasets.plot import plot_distribution

fig = plot_distribution({"train": train, "test": test}, color_by="split")
fig.savefig("office_distribution.png", dpi=120)
```

`examples/dataset_distribution.py --dataset ujiindoorloc` writes the standard set of figures for a
dataset: 2-D and 3-D distributions, a density map, and the error CDF and error map of k-NN and WKNN.
