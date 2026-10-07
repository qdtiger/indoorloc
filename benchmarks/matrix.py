"""The benchmark matrix: which methods run on which dataset, under which protocol and preprocessing.

One :class:`Table` per result table of ``docs/benchmarks.md``. A table fixes the dataset (and its
constructor options), the protocol and the units; its ``runs`` list preprocessing specs with the
method specs run under each (strings exactly as given to ``indoorloc benchmark --preprocess`` and
``--method``). Every (preprocessing, method) pair is one *cell*, run in its own process by
``run.py``. A method that is not run appears in ``skips`` with the reason: no cell is dropped
silently.

Method specs may contain ``{anchors}``: ``run.py`` replaces it with the dataset's
``meta["anchors"]`` (receiver or beacon positions) as a JSON list, so the recorded command line
holds the exact geometry.

Choices that apply everywhere (see ``README.md``):

* seed 0 for every protocol and every seeded method; OpenBLAS/OpenMP/MKL threads = 8; forests
  ``n_jobs=8`` (results do not depend on it: trees are averaged in index order);
* library defaults for every method unless the spec says otherwise (k-NN ``k=5``; the MLP trains
  at most 100 epochs with early stopping on a 10 % split of the training data);
* on RSSI tables every method runs after the -104 dBm fill, and Horus also runs on the raw
  readings (``--preprocess none``: NaN = not heard), the input it is designed for: it models
  whether an AP is heard at a location instead of treating the fill value as a reading;
* every cell must finish within ``run.py``'s limits (30 minutes, 2,500 MB by default). SVM
  (libsvm: training time roughly quadratic in the training rows) and the GP radio map (an n x n
  kernel matrix over the n distinct training positions) are left out, with the reason, where a
  measurement or a size argument shows they would not; a cell that was run and hit a limit is
  kept as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BASELINE = "benchmarks.baselines:TrainingCentroid"
RF = "rf(n_jobs=8)"
ET = "extratrees(n_jobs=8)"
ENSEMBLE = 'ensemble(localizers=["wknn","rf","horus"],combine="median")'
ENSEMBLE_TREES = 'ensemble(localizers=["wknn","rf","extratrees"],combine="median")'
GP = "gp_radiomap"
GP_HIER = "hierarchical(position_model=gp_radiomap)"
PATHLOSS_KNOWN = "pathloss(anchors={anchors})"
PATHLOSS_FREE = "pathloss"
CENTROID_KNOWN = "centroid(anchors={anchors})"

# method spec -> (English, Chinese) name shown in the tables; the JSON keeps the spec itself
DISPLAY = {
    BASELINE: ("training centroid (baseline)", "训练集质心（基线）"),
    "knn(k=1)": ("1-NN", "1-NN"),
    "knn": ("k-NN (k=5)", "k-NN (k=5)"),
    "wknn": ("WKNN (k=5)", "WKNN (k=5)"),
    "horus": ("Horus", "Horus"),
    RF: ("random forest", "随机森林"),
    ET: ("extra trees", "极端随机树"),
    ENSEMBLE: ("ensemble: median of WKNN, RF, Horus", "集成：WKNN/RF/Horus 中位数"),
    ENSEMBLE_TREES: ("ensemble: median of WKNN, RF, extra trees", "集成：WKNN/RF/极端随机树 中位数"),
    "mlp": ("MLP", "MLP"),
    "svm": ("SVM (RBF)", "SVM (RBF)"),
    GP: ("GP radio map", "GP 无线电地图"),
    GP_HIER: ("GP radio map, per floor", "GP 无线电地图（逐楼层）"),
    PATHLOSS_KNOWN: ("path-loss ML, known receivers", "路径损耗最大似然（已知接收机位置）"),
    PATHLOSS_FREE: ("path-loss ML, receivers estimated", "路径损耗最大似然（估计接收机位置）"),
    CENTROID_KNOWN: ("weighted centroid, known receivers", "加权质心（已知接收机位置）"),
}

# preprocessing spec -> (English, Chinese) short description
PREPROCESS = {
    "fill": ("fill -104 dBm", "缺失填 -104 dBm"),
    "positive": ("positive", "positive 表示"),
    "exponential": ("exponential (α=24)", "exponential 表示 (α=24)"),
    "powed": ("powed (β=e)", "powed 表示 (β=e)"),
    "none": ("raw RSSI, NaN = not heard", "原始 RSSI，NaN = 未听到"),
    "CSIAmplitude": ("|CSI| (linear)", "|CSI| 幅度（线性）"),
    "fill(value=-15)": ("dB amplitude, NaN → -15 dB", "dB 幅度，NaN 填 -15 dB"),
    "fill(value=-110)": ("fill -110 dBm", "缺失填 -110 dBm"),
}


@dataclass(frozen=True)
class Skip:
    """A method that is not run on a table, and why (English, Chinese)."""

    method: str
    en: str
    zh: str


@dataclass(frozen=True)
class Table:
    """One result table: a dataset selection, a protocol and the cells run on it."""

    dataset: str
    id: str
    title: tuple[str, str]
    protocol: str
    runs: tuple[tuple[str, tuple[str, ...]], ...]
    options: dict = field(default_factory=dict)
    skips: tuple[Skip, ...] = ()
    notes: tuple[tuple[str, str], ...] = ()
    runner: str = "cli"  # "cli": indoorloc benchmark; "labels": benchmarks.labels (label accuracy)
    label: str | None = None  # the groups column scored by the labels runner

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.id}"

    def cells(self):
        for pre, methods in self.runs:
            for method in methods:
                yield pre, method


def _rssi(fill_extra=(), reps=True, raw_extra=()):
    """The RSSI fingerprinting set: fill -104 dBm for every method, the Torres-Sospedra
    representations with 1-NN and WKNN, and Horus on the raw readings (NaN = not heard), which
    its detection model scores directly; ``raw_extra``: further methods run on the raw readings."""
    runs = [("fill", (BASELINE, "knn(k=1)", "knn", "wknn", "horus", RF, ET, ENSEMBLE, "mlp", *fill_extra))]
    if reps:
        runs += [("positive", ("knn(k=1)", "wknn")), ("exponential", ("wknn",)), ("powed", ("wknn",))]
    runs += [("none", ("horus", *raw_extra))]
    return tuple(runs)


# ------------------------------------------------------------------ reasons used more than once
NO_ANCHORS = Skip("pathloss, centroid",
                  "model-based: the dataset does not publish the access-point positions",
                  "基于模型的方法：数据集未公开 AP 位置")
# The SVM estimates below scale the measured H-WILD conference cell (5 folds, 17,596-19,503 training
# packets, one epsilon-SVR per position axis): 79-103 s per two-axis fit, i.e. 2.55-2.70e-7 s x n^2,
# libsvm's training time growing roughly quadratically with the training rows. Estimates, not runs.
SVM_LOUNGE = Skip("svm",
                  "estimated, not run: 8 fits (one per left-out user) on 37,156-37,398 training packets. Scaled "
                  "quadratically from the conference room's measured SVM cell (79-103 s per fit on 17,596-19,503 "
                  "packets), the fits alone would take about 47-50 minutes, beyond the 30-minute limit per cell",
                  "估计值，未运行：8 次训练（每个留出用户一次），每次 37,156-37,398 个数据包。按会议室实测的 SVM 单元格"
                  "（在 17,596-19,503 个数据包上每次训练 79-103 s）以训练时间随行数平方增长推算，仅训练就需约 47-50 分钟，"
                  "超出每个单元格 30 分钟的限制")
SVM_HALOC = Skip("svm",
                 "estimated, not run: one fit on 96,491 training packets with three position axes (one SVR each). "
                 "Scaled quadratically from the measured H-WILD conference SVM fits (79-103 s for two axes on "
                 "17,596-19,503 packets), it would take about an hour, beyond the 30-minute limit per cell",
                 "估计值，未运行：在 96,491 个训练数据包上训练一次，三个位置坐标轴（每轴一个 SVR）。按 H-WILD 会议室实测的 "
                 "SVM 训练（两个坐标轴、17,596-19,503 个数据包，每次 79-103 s）以训练时间随行数平方增长推算，约需一小时，"
                 "超出每个单元格 30 分钟的限制")
GP_CONTINUOUS = lambda n, mb: Skip("gp_radiomap",  # noqa: E731
                                   f"positions are continuous ({n} distinct training positions): the fit runs an "
                                   f"O(n^3) Cholesky of an n x n kernel matrix ({mb} in float64) at each of the 144 "
                                   "hyper-parameter grid points",
                                   f"位置连续（训练集 {n} 个不同位置）：拟合需在 144 个超参数网格点上各做一次 n×n 核矩阵"
                                   f"（float64 为 {mb}）的 O(n^3) Cholesky 分解")
HORUS_CONTINUOUS = lambda n, en, zh: Skip("horus, ensemble with Horus",  # noqa: E731
                                          f"every training packet has its own position ({n}), so every "
                                          f"per-location Gaussian would be fitted to a single packet; {en}",
                                          f"每个训练数据包的位置都不同（{n}），每个逐位置高斯模型只有一个数据包；{zh}")
NOT_RSSI = Skip("positive / exponential / powed",
                "the Torres-Sospedra representations are defined for RSSI in dBm, not CSI",
                "Torres-Sospedra 表示针对 dBm 的 RSSI，不适用于 CSI")
NO_GEOMETRY_CSI = Skip("pathloss, centroid",
                       "model-based RSSI methods; the CSI amplitude is not a received power per anchor",
                       "基于 RSSI 模型的方法；CSI 幅度不是每个锚点的接收功率")

TABLES: tuple[Table, ...] = (
    # ------------------------------------------------------------------ UJIIndoorLoc
    Table("ujiindoorloc", "official",
          ("UJIIndoorLoc, official split (19,937 train / 1,111 validation scans)",
           "UJIIndoorLoc，官方划分（训练 19,937 / 验证 1,111 条）"),
          "official", _rssi(("svm", GP_HIER)),
          skips=(NO_ANCHORS,),
          notes=(("Errors are in EPSG:3857 (Web Mercator) metres, the dataset's own coordinates, in which "
                  "UJIIndoorLoc results are usually reported; multiply by meta['ground_scale'] = 0.7661 "
                  "(cos 39.99 deg) for ground metres (e.g. WKNN 8.794 -> 6.737 m). The IPIN score adds 15 m per floor and 50 m for a wrong building (75th percentile).",
                  "误差单位为 EPSG:3857（Web Mercator）米，即数据集自身的坐标，UJIIndoorLoc 的结果通常以此报告；乘以 "
                  "meta['ground_scale'] = 0.7661（cos 39.99°）得到地面米（例如 WKNN 8.794 → 6.737 m）。IPIN 分数：每错一层加 "
                  "15 m、楼栋错加 50 m 后的第 75 百分位。"),)),
    # ------------------------------------------------------------------ SODIndoorLoc
    Table("sodindoorloc", "official-all",
          ("SODIndoorLoc, all three buildings, official split (21,205 train / 2,720 test scans)",
           "SODIndoorLoc，三栋楼合并，官方划分（训练 21,205 / 测试 2,720 条）"),
          "official", _rssi(("svm", GP_HIER)), skips=(NO_ANCHORS,),
          notes=(("Each building has its own local frame: a position error is meaningful only when the "
                  "building is right (see the building accuracy); the per-building tables below avoid the issue.",
                  "每栋楼使用各自的局部坐标系：只有楼栋判断正确时位置误差才有意义（见楼栋准确率）；下面的逐楼表格"
                  "不存在此问题。"),)),
    *(Table("sodindoorloc", f"official-{b}",
            (f"SODIndoorLoc, building {b}, official split", f"SODIndoorLoc，{b} 楼，官方划分"),
            "official", _rssi(("svm", GP_HIER)), options={"building": b}, skips=(NO_ANCHORS,))
      for b in ("CETC331", "HCXY", "SYL")),
    # ------------------------------------------------------------------ Tampere
    Table("tampere", "official",
          ("Tampere crowdsourced, official split (697 train / 3,951 test scans), 3-D error",
           "Tampere 众包数据，官方划分（训练 697 / 测试 3,951 条），三维误差"),
          "official", _rssi(("svm", GP_HIER)), skips=(NO_ANCHORS,),
          notes=(("Positions are (x, y, z) with z the floor height (3.7 m per floor), so every error is the 3-D "
                  "error of the dataset's benchmark software; floor accuracy is on floor = round(z / 3.7).",
                  "位置为 (x, y, z)，z 为楼层高度（每层 3.7 m），因此误差均为数据集基准软件使用的三维误差；楼层准确率"
                  "按 floor = round(z / 3.7) 计算。"),)),
    # ------------------------------------------------------------------ TUJI1
    Table("tuji1", "official",
          ("TUJI1, official split (6,752 train / 2,147 test scans, five devices, one floor)",
           "TUJI1，官方划分（训练 6,752 / 测试 2,147 条，五台设备，单层）"),
          "official", _rssi(("svm", GP)), skips=(NO_ANCHORS,),
          notes=(("One floor with constant z, so the 2-D errors equal the paper's 3-D errors. The paper's 1-NN "
                  "baseline (positive representation, Euclidean) is 3.34 m; here 1-NN + positive learns the "
                  "missing value from the training data only (min - 1 dBm).",
                  "单层且 z 恒定，二维误差即论文中的三维误差。论文 1-NN 基线（positive 表示、欧氏距离）为 3.34 m；"
                  "此处 1-NN + positive 仅从训练集学习缺失值（最小值 - 1 dBm）。"),)),
    # ------------------------------------------------------------------ LongTermWiFi
    Table("longtermwifi", "official",
          ("LongTermWiFi, all 25 months pooled: every training set vs every test set (23,040 / 81,120 scans)",
           "LongTermWiFi，25 个月合并：全部训练集对全部测试集（23,040 / 81,120 条）"),
          "official", _rssi(("svm", GP_HIER)), skips=(NO_ANCHORS,),
          notes=(("Training positions (24) and test positions (106) are fixed and repeated every month; pooling "
                  "months mixes 25 months of signal drift into one model.",
                  "训练位置（24 个）和测试位置（106 个）每月固定重复；合并各月会把 25 个月的信号漂移混入同一个模型。"),
                 ("The SVM cell is the slowest of the matrix (fit 4.3 minutes, prediction of the 81,120 test scans 17 "
                  "minutes). Two earlier attempts hit the 30-minute limit while the shared cgroup was throttled on "
                  "memory; the recorded run had no memory stall.",
                  "SVM 单元格是整个矩阵中最慢的（训练 4.3 分钟，预测 81,120 条测试样本 17 分钟）。此前两次尝试在共享控制组"
                  "内存受限流时触及 30 分钟限制；记录下的这次运行没有内存停顿。"),)),
    Table("longtermwifi", "within-month",
          ("LongTermWiFi, the authors' protocol: train and test within each month (25 folds)",
           "LongTermWiFi，作者协议：每月内训练与测试（25 折）"),
          "benchmarks.protocols:WITHIN_MONTH", _rssi(("svm", GP_HIER)), skips=(NO_ANCHORS,),
          notes=(("Pooled over the 25 monthly folds (every test scan once). Month 1 has 15 training sets and "
                  "months 2-24 one each (576 scans per set); month 25 repeats its training set and its five test sets "
                  "with a second phone (Galaxy A5), so that fold trains on both phones (1,152 scans) and tests on "
                  "ten test sets (6,240 scans).",
                  "对 25 个月度折叠合并统计（每条测试样本一次）。第 1 个月有 15 个训练集，第 2-24 个月各 1 个（每个 576 条）；"
                  "第 25 个月用第二部手机（Galaxy A5）重复采集了训练集和五个测试集，因此该折用两部手机的训练集（1,152 条）"
                  "训练、在十个测试集（6,240 条）上测试。"),)),
    # ------------------------------------------------------------------ UJI BLE (iBeacon)
    Table("ibeacon_rssi", "official-all",
          ("UJI iBeacon RSS, both zones, the authors' default train/test configuration",
           "UJI iBeacon RSS，两个区域合并，作者默认训练/测试配置"),
          "official", _rssi(("svm", GP_HIER)),
          skips=(Skip("pathloss, centroid",
                      "19-27 % of the test scans hear fewer than 3 beacons (11-12 % hear none): a model-based "
                      "method cannot place them, and `indoorloc benchmark` requires an estimate for every row",
                      "19-27% 的测试样本听到的信标少于 3 个（11-12% 一个也没有）：基于模型的方法无法定位，而 "
                      "`indoorloc benchmark` 要求每条样本都有估计"),),
          notes=(("The two zones (geo = building 1, lib = 2) have separate frames and beacons; see the "
                  "per-zone tables. Scans that heard no beacon (fill: all -104 dBm) are kept, as in the dataset.",
                  "两个区域（geo = 楼栋 1，lib = 2）坐标系与信标各自独立，见逐区域表格。未听到任何信标的样本（填充后"
                  "全为 -104 dBm）按数据集原样保留。"),)),
    *(Table("ibeacon_rssi", f"official-{z}",
            (f"UJI iBeacon RSS, zone {z}, the authors' default train/test configuration",
             f"UJI iBeacon RSS，{z} 区域，作者默认训练/测试配置"),
            "official", _rssi(("svm", GP)), options={"zone": z},
            skips=(Skip("pathloss, centroid", "see the table of both zones", "见两区域合并表格"),))
      for z in ("lib", "geo")),
    # ------------------------------------------------------------------ BBIL (ble_indoor)
    *(Table("ble_indoor", f"official-{room}",
            (f"BBIL {room} ({exp}), official split: train vs test recordings (valid unused)",
             f"BBIL {zh}（{exp}），官方划分：train 对 test 录制（未使用 valid）"),
            "official",
            _rssi(("svm", GP), raw_extra=(PATHLOSS_KNOWN, PATHLOSS_FREE, CENTROID_KNOWN))
            + ((("fill(value=-110)", ("wknn",)),) if room == "office" else ()),
            options={"room": room},
            notes=(("Positions are continuous along walks (interpolated between landmark presses): most training "
                    "positions (69 % in the lab, 78 % in the office) hold one or two scans, so most of Horus's "
                    "per-location Gaussians rest on the variance floor and it acts much like a Gaussian-kernel "
                    "nearest-neighbour rule. The authors report the mean and P90 error on the test recordings. "
                    "Model-based rows use the receivers' positions (meta['anchors'], 2-D; receivers 1.6 m high) "
                    "and raw RSSI with missing readings left missing.",
                    "位置沿行走轨迹连续（在地标按键之间插值）：多数训练位置（实验室 69%，办公室 78%）只有一两条样本，"
                    "因此 Horus 的逐位置高斯大多取方差下限，其行为近似高斯核近邻。作者报告测试录制上的平均误差和 P90。"
                    "基于模型的行使用接收机位置（meta['anchors']，二维；"
                    "接收机高 1.6 m），并直接使用原始 RSSI（缺失保持缺失）。"),
                   (f"The GP radio map is fitted over the {n_pos} distinct training positions and scores them as "
                    "candidates.",
                    f"GP 无线电地图在 {n_pos} 个不同的训练位置上拟合，并以它们作为候选位置打分。"),))
      for room, exp, zh, n_pos in (("office", "experiment1", "办公室", "7,534"), ("lab", "experiment2", "实验室", "3,311"))),
    # ------------------------------------------------------------------ UCI BLE RSSI (Waldo library)
    Table("ble_rssi_uci", "random-80-20",
          ("UCI BLE RSSI (Waldo library), random 80/20 split of the 1,420 labelled scans, seed 0; grid cells",
           "UCI BLE RSSI（Waldo 图书馆），1,420 条有标签样本随机 80/20 划分，种子 0；单位为网格"),
          "random-80-20", _rssi(("svm", GP)),
          skips=(Skip("pathloss, centroid", "the beacon positions are given only as a map image, not as coordinates",
                      "信标位置只以地图图片给出，没有坐标"),),
          notes=(("Errors are in grid cells of the map shipped with the data (the cell size is not stated). "
                  "Consecutive scans at one cell are near duplicates, so a random split is optimistic. The "
                  "5,191 unlabelled scans are not used.",
                  "误差单位为数据附带地图的网格（未给出网格尺寸）。同一网格的连续扫描几乎重复，随机划分偏乐观。未使用 "
                  "5,191 条无标签样本。"),)),
    # ------------------------------------------------------------------ UCI WLAN RSSI (rooms)
    Table("wlanrssi", "stratified-5-fold",
          ("UCI Wireless Indoor Localization: room accuracy, stratified 5-fold cross-validation, seed 0",
           "UCI Wireless Indoor Localization：房间准确率，分层 5 折交叉验证，种子 0"),
          "stratified-kfold-5",
          ((("fill", (BASELINE, "knn(k=1)", "knn", "wknn", "horus", RF, ET, ENSEMBLE, "mlp", "svm")),
            ("positive", ("knn(k=1)", "wknn")), ("exponential", ("wknn",)), ("powed", ("wknn",)),
            ("none", ("horus",)))),
          runner="labels", label="room",
          skips=(Skip("gp_radiomap", "the dataset has no coordinates: a radio map over position cannot be fitted",
                      "数据集没有坐标，无法拟合基于位置的无线电地图"), NO_ANCHORS),
          notes=(("Only room labels (1-4) exist. Each localizer gets the room as its floor label and a constant "
                  "placeholder position (never scored); the table reports room accuracy. The file has no missing "
                  "readings, so 'fill' changes nothing, and Horus gets the same input with and without it.",
                  "数据只有房间标签（1-4）。每个定位器把房间当作楼层标签，并使用恒定占位位置（不计分）；表格报告房间"
                  "准确率。文件无缺失读数，'fill' 不改变数据，Horus 填充与否输入相同。"),)),
    # ------------------------------------------------------------------ HALOC (CSI)
    Table("haloc", "official",
          ("HALOC, official split: sequences 0-3 train, 5 test (96,491 / 14,277 packets); CSI amplitude",
           "HALOC，官方划分：序列 0-3 训练、5 测试（96,491 / 14,277 个数据包）；CSI 幅度"),
          "official",
          (("CSIAmplitude", (BASELINE, "knn(k=1)", "knn", "wknn", RF, ET, ENSEMBLE_TREES, "mlp")),),
          skips=(HORUS_CONTINUOUS("96,491 distinct",
                                  "the test x location log-likelihood matrix alone would be 14,277 x 96,491 float64 = "
                                  "11 GB, far above the 2,500 MB per-cell limit",
                                  "仅测试×位置的对数似然矩阵就有 14,277 × 96,491 个 float64 = 11 GB，远超每个单元格 "
                                  "2,500 MB 的内存限制"),
                 SVM_HALOC, GP_CONTINUOUS("96,491", "74 GB"), NOT_RSSI,
                 NO_GEOMETRY_CSI),
          notes=(("Features: |H| of the 52 L-LTF subcarriers (raw ESP32 I/Q, not calibrated). Sequence 4 "
                  "(validation) is not used. Positions are 3-D (z about 1.2-1.3 m).",
                  "特征：52 个 L-LTF 子载波的 |H|（ESP32 原始 I/Q，未校准）。未使用序列 4（验证集）。位置为三维"
                  "（z 约 1.2-1.3 m）。"),)),
    # ------------------------------------------------------------------ H-WILD (CSI)
    *(Table("hwild", f"louo-{env}",
            (f"H-WILD {env}: leave one user out; CSI amplitude of 4 APs x 3 antennas x 30 subcarriers",
             f"H-WILD {zh}：留一用户交叉验证；4 个 AP × 3 天线 × 30 子载波的 CSI 幅度"),
            "benchmarks.protocols:LEAVE_ONE_USER_OUT",
            (("CSIAmplitude", (BASELINE, "knn(k=1)", "knn", "wknn", RF, ET, ENSEMBLE_TREES, "mlp",
                               *(("svm",) if svm else ()))),),
            options={"environment": env},
            skips=(HORUS_CONTINUOUS(f"{n} packets, {n} distinct positions",
                                    "a probe in the conference room (fold user=1: 17,663 training packets, 200 test "
                                    "packets) took 170 ms per test packet, about 65 minutes for that room's 22,970 "
                                    "test packets and more for the larger rooms, beyond the 30-minute limit",
                                    "在会议室的探测（留出用户 1 的一折：训练 17,663 个、测试 200 个数据包）每个测试数据包"
                                    "耗时 170 ms，该房间 22,970 个测试数据包约需 65 分钟，更大的房间更久，超出 30 分钟限制"),
                   *(() if svm else (SVM_LOUNGE,)),
                   GP_CONTINUOUS(fit, gb), NOT_RSSI, NO_GEOMETRY_CSI),
            notes=(("Pooled over the leave-one-user-out folds (every packet tested once, by a model that never "
                    "saw that user). Walks with and without interference are both included.",
                    "对留一用户各折合并统计（每个数据包由从未见过该用户的模型测试一次）。包含有干扰与无干扰两类行走。"),))
      for env, zh, n, folds, fit, gb, svm in (("conference", "会议室", "22,970", 5, "17,596-19,503", "3.0 GB", True),
                                              ("laboratory", "实验室", "26,833", 5, "21,434-21,535", "3.7 GB", True),
                                              ("office", "办公室", "26,935", 5, "21,537-21,564", "3.7 GB", True),
                                              ("lounge", "休息室", "42,554", 8, "37,156-37,398", "11 GB", False))),
    # ------------------------------------------------------------------ CSI fingerprint (Zhu et al.)
    *(Table("csi_fingerprint", f"points-{area}",
            (f"CSI fingerprint dataset, area {area}: 5 folds over reference points; first 50 packets per point",
             f"CSI 指纹数据集，{zh}：按参考点 5 折；每点前 50 个数据包"),
            "benchmarks.protocols:POINT_KFOLD_5",
            (("fill(value=-15)", (BASELINE, "knn(k=1)", "knn", "wknn", "horus", RF, ET, ENSEMBLE, "mlp", "svm", GP)),),
            options={"area": area, "packets": 50},
            skips=(NOT_RSSI, NO_GEOMETRY_CSI),
            notes=(("Errors are in grid steps of the area's reference-point grid (spacing not stated). Every test "
                    "point is absent from training, so this measures interpolation to new positions; a random "
                    "packet split would score almost 0 (point recognition). Features: 3 x 30 amplitudes in dB as "
                    "stored; the few zero amplitudes (-inf dB, NaN after loading: 48 values in the lab, none in "
                    "the other three areas) are set to -15 dB, below every stored value.",
                    "误差单位为该区域参考点网格的步长（未给出间距）。测试点都不在训练集中，因此衡量的是向新位置的插值；"
                    "随机按包划分几乎为 0（只是识别参考点）。特征：3 × 30 个 dB 幅度（原样）；极少数零幅度（-inf dB，"
                    "加载后为 NaN：实验室 48 个，其他三个区域没有）置为 -15 dB，低于所有存储值。"),))
      for area, zh in (("lab", "实验室"), ("meeting", "会议室"), ("conference", "报告厅"), ("minilab", "小实验室"))),
)

DATASET_ORDER = ("ujiindoorloc", "sodindoorloc", "tampere", "tuji1", "longtermwifi", "ibeacon_rssi", "ble_indoor",
                 "ble_rssi_uci", "wlanrssi", "haloc", "hwild", "csi_fingerprint")


def tables(datasets=None, ids=None) -> list[Table]:
    """The tables of the selected datasets (all by default) and table ids, in document order."""
    out = [t for t in TABLES if (not datasets or t.dataset in datasets) and (not ids or t.id in ids)]
    return sorted(out, key=lambda t: DATASET_ORDER.index(t.dataset))


def display(method: str, lang: int = 0) -> str:
    """Table name of a method spec (lang 0 English, 1 Chinese); unknown specs are shown as given."""
    return DISPLAY[method][lang] if method in DISPLAY else method


def display_preprocess(spec: str, lang: int = 0) -> str:
    return PREPROCESS[spec][lang] if spec in PREPROCESS else spec
