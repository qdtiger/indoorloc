# IndoorLoc 真实数据基准测试

由 `benchmarks/render.py` 根据 `benchmarks/results/*.json` 生成，请勿手工编辑。English version: [benchmarks.md](benchmarks.md)。

下列结果表中的每个数字，都是在 *实验环境* 所述机器上，用 IndoorLoc 的公开命令行（`indoorloc benchmark`；只有房间标签的数据集用 `benchmarks/labels.py`）在经 sha256 校验的真实数据文件上运行得到的，每个单元格一个独立进程。结果表中没有任何数字抄自论文。其他作者发表的数字只出现在单独的 *已发表结果* 表中，按原文报告并注明核查状态；它们使用了不同的代码、预处理，往往还有不同的评测协议，只作参考，不作排名。

## 目录

- [总览](#总览)
- [UJIIndoorLoc](#ujiindoorloc)
- [SODIndoorLoc](#sodindoorloc)
- [Tampere (Wi-Fi crowdsourced fingerprints)](#tampere)
- [TUJI1](#tuji1)
- [Long-Term WiFi (UJI library)](#longtermwifi)
- [iBeacon RSSI (UJI BLE RSS database)](#ibeacon-rssi)
- [BLE Indoor (BBIL)](#ble-indoor)
- [BLE RSSI (UCI, Western Michigan University library)](#ble-rssi-uci)
- [Wireless Indoor Localization (UCI)](#wlanrssi)
- [HALOC](#haloc)
- [H-WILD](#hwild)
- [CSI fingerprint dataset (Zhu et al., four rooms)](#csi-fingerprint)

## 总览

每个结果表一行。*基线* 对所有查询都给出训练位置的平均值（不能超过它的方法没有学到任何位置信息）；*最佳* 为该表各方法中合并平均误差最小者（房间标签则为准确率最高者）。

| 表 | 协议 | 训练 / 测试行数 | 单位 | 基线 | WKNN (k=5) | 最佳（方法：数值） |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| [ujiindoorloc / official](#ujiindoorloc-official) | `official` | 19,937 / 1,111 | 米（EPSG:3857） | 143.330 | 8.794 | WKNN (k=5) (exponential 表示 (α=24)): 7.988 |
| [sodindoorloc / official-all](#sodindoorloc-official-all) | `official` | 21,205 / 2,720 | 米 | 636.083 | 3.350 | Horus (原始 RSSI，NaN = 未听到): 2.852 |
| [sodindoorloc / official-CETC331](#sodindoorloc-official-cetc331) | `official` | 955 / 840 | 米 | 15.245 | 2.817 | Horus (原始 RSSI，NaN = 未听到): 2.425 |
| [sodindoorloc / official-HCXY](#sodindoorloc-official-hcxy) | `official` | 11,370 / 860 | 米 | 36.493 | 2.320 | Horus (原始 RSSI，NaN = 未听到): 1.962 |
| [sodindoorloc / official-SYL](#sodindoorloc-official-syl) | `official` | 8,880 / 1,020 | 米 | 19.982 | 4.656 | MLP (缺失填 -104 dBm): 3.460 |
| [tampere / official](#tampere-official) | `official` | 697 / 3,951 | 米 | 33.208 | 9.300 | 极端随机树 (缺失填 -104 dBm): 8.081 |
| [tuji1 / official](#tuji1-official) | `official` | 6,752 / 2,147 | 米 | 9.261 | 2.733 | Horus (原始 RSSI，NaN = 未听到): 2.126 |
| [longtermwifi / official](#longtermwifi-official) | `official` | 23,040 / 81,120 | 米 | 5.004 | 2.288 | 集成：WKNN/RF/Horus 中位数 (缺失填 -104 dBm): 1.950 |
| [longtermwifi / within-month](#longtermwifi-within-month) | `benchmarks.protocols:WITHIN_MONTH` | 25 × (576–8,640 / 3,120–6,240) | 米 | 5.004 | 2.529 | 集成：WKNN/RF/Horus 中位数 (缺失填 -104 dBm): 2.259 |
| [ibeacon_rssi / official-all](#ibeacon-rssi-official-all) | `official` | 2,748 / 1,860 | 米 | 8.699 | 3.510 | 集成：WKNN/RF/Horus 中位数 (缺失填 -104 dBm): 3.271 |
| [ibeacon_rssi / official-lib](#ibeacon-rssi-official-lib) | `official` | 876 / 1,080 | 米 | 4.183 | 2.753 | SVM (RBF) (缺失填 -104 dBm): 2.403 |
| [ibeacon_rssi / official-geo](#ibeacon-rssi-official-geo) | `official` | 1,872 / 780 | 米 | 4.072 | 3.984 | 随机森林 (缺失填 -104 dBm): 2.990 |
| [ble_indoor / official-office](#ble-indoor-official-office) | `official` | 22,237 / 5,110 | 米 | 7.416 | 3.517 | MLP (缺失填 -104 dBm): 3.235 |
| [ble_indoor / official-lab](#ble-indoor-official-lab) | `official` | 13,238 / 3,194 | 米 | 3.613 | 2.180 | MLP (缺失填 -104 dBm): 2.014 |
| [ble_rssi_uci / random-80-20](#ble-rssi-uci-random-80-20) | `random-80-20` | 1,136 / 284 | 网格 | 4.907 | 1.549 | 极端随机树 (缺失填 -104 dBm): 1.474 |
| [wlanrssi / stratified-5-fold](#wlanrssi-stratified-5-fold) | `stratified-kfold-5` | 5 × (1,600–1,600 / 400–400) | % 房间 | 25.00% | 98.40% | MLP (缺失填 -104 dBm): 98.60% |
| [haloc / official](#haloc-official) | `official` | 96,491 / 14,277 | 米 | 4.897 | 3.669 | MLP (\|CSI\| 幅度（线性）): 2.964 |
| [hwild / louo-conference](#hwild-louo-conference) | `benchmarks.protocols:LEAVE_ONE_USER_OUT` | 5 × (17,596–19,503 / 3,467–5,374) | 米 | 2.408 | 1.056 | MLP (\|CSI\| 幅度（线性）): 0.806 |
| [hwild / louo-laboratory](#hwild-louo-laboratory) | `benchmarks.protocols:LEAVE_ONE_USER_OUT` | 5 × (21,434–21,535 / 5,298–5,399) | 米 | 2.501 | 1.544 | MLP (\|CSI\| 幅度（线性）): 1.168 |
| [hwild / louo-office](#hwild-louo-office) | `benchmarks.protocols:LEAVE_ONE_USER_OUT` | 5 × (21,537–21,564 / 5,371–5,398) | 米 | 2.727 | 2.003 | MLP (\|CSI\| 幅度（线性）): 1.601 |
| [hwild / louo-lounge](#hwild-louo-lounge) | `benchmarks.protocols:LEAVE_ONE_USER_OUT` | 8 × (37,156–37,398 / 5,156–5,398) | 米 | 3.297 | 2.330 | MLP (\|CSI\| 幅度（线性）): 1.648 |
| [csi_fingerprint / points-lab](#csi-fingerprint-points-lab) | `benchmarks.protocols:POINT_KFOLD_5` | 5 × (12,650–12,700 / 3,150–3,200) | 网格步长 | 8.485 | 10.320 | 随机森林 (dB 幅度，NaN 填 -15 dB): 7.884 |
| [csi_fingerprint / points-meeting](#csi-fingerprint-points-meeting) | `benchmarks.protocols:POINT_KFOLD_5` | 5 × (7,000–7,050 / 1,750–1,800) | 网格步长 | 5.268 | 6.342 | 随机森林 (dB 幅度，NaN 填 -15 dB): 4.682 |
| [csi_fingerprint / points-conference](#csi-fingerprint-points-conference) | `benchmarks.protocols:POINT_KFOLD_5` | 5 × (6,400–6,400 / 1,600–1,600) | 网格步长 | 5.094 | 5.409 | MLP (dB 幅度，NaN 填 -15 dB): 4.826 |
| [csi_fingerprint / points-minilab](#csi-fingerprint-points-minilab) | `benchmarks.protocols:POINT_KFOLD_5` | 5 × (1,400–1,400 / 350–350) | 网格步长 | 2.423 | 2.092 | MLP (dB 幅度，NaN 填 -15 dB): 1.787 |

## 观察

由下方位置误差结果表统计（合并平均误差，各表采用其自身协议）。统计时任何差异都算作胜出，无论多小；具体幅度见各表。

- 使用 positive 表示的 WKNN (k=5) 平均误差低于缺失填 -104 dBm 的情形：7 / 15 个表。
- 使用 exponential 表示的 WKNN (k=5) 平均误差低于缺失填 -104 dBm 的情形：12 / 15 个表。
- 使用 powed 表示的 WKNN (k=5) 平均误差低于缺失填 -104 dBm 的情形：10 / 15 个表。
- WKNN、随机森林与 Horus 的中位数集成优于其全部三个成员：11 / 19 个表。
- WKNN、随机森林与极端随机树的中位数集成（CSI 表）优于其全部三个成员：1 / 4 个表。
- Horus 平均误差低于 WKNN：3 / 19 个表。
- 在原始读数上（NaN = 未听到，即其检测模型所针对的输入）运行的 Horus 平均误差低于缺失填 -104 dBm 后的 Horus：13 / 15 个表；低于 WKNN：8 / 15 个表。
- 未能超过训练集质心基线（平均误差不低于基线）的方法：
  - [ibeacon_rssi / official-geo](#ibeacon-rssi-official-geo): 1-NN; Horus; GP 无线电地图; 1-NN (positive 表示); Horus (原始 RSSI，NaN = 未听到)
  - [ble_indoor / official-lab](#ble-indoor-official-lab): 路径损耗最大似然（已知接收机位置） (原始 RSSI，NaN = 未听到); 路径损耗最大似然（估计接收机位置） (原始 RSSI，NaN = 未听到)
  - [csi_fingerprint / points-lab](#csi-fingerprint-points-lab): 1-NN; k-NN (k=5); WKNN (k=5); Horus; 集成：WKNN/RF/Horus 中位数; MLP; SVM (RBF); GP 无线电地图
  - [csi_fingerprint / points-meeting](#csi-fingerprint-points-meeting): 1-NN; k-NN (k=5); WKNN (k=5); Horus; 集成：WKNN/RF/Horus 中位数; GP 无线电地图
  - [csi_fingerprint / points-conference](#csi-fingerprint-points-conference): 1-NN; k-NN (k=5); WKNN (k=5); Horus; 集成：WKNN/RF/Horus 中位数; SVM (RBF); GP 无线电地图
  - [csi_fingerprint / points-minilab](#csi-fingerprint-points-minilab): Horus; GP 无线电地图

## 实验环境

- 硬件：13th Gen Intel(R) Core(TM) i9-13900K，32 个逻辑 CPU，39.2 GB 内存；Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.43。该工作站为共享（同时运行着一个 GPU 训练任务和其他开发进程），因此时间仅供参考；所有单元格只用 CPU。
- 线程：OpenBLAS / OpenMP / MKL = 8（PyTorch 的算子内线程数取自 OMP_NUM_THREADS）；森林 `n_jobs=8`。一次只运行一个单元格。
- 软件：Python 3.14.4，numpy 2.4.5（scipy-openblas 0.3.31.188.0），pandas 3.0.3, scipy 1.17.1, sklearn 1.8.0, torch 2.11.0+cu128；indoorloc 0.2.0.dev0。
- 代码：每个单元格都记录其导入的库的源码摘要（对包内 .py/.json 文件的 sha256）。已完成的单元格运行了 3 个版本：`84a6dc45715c3b84`（328 个单元格）：2026-09-28 10:04 UTC 用 `run.py --freeze` 冻结的副本，来自 `refactor/five-layer-architecture` 分支的 git 提交 `ff90d76a9dd9`，含未提交的修改；`34d0d1b7013ddf9b`（16 个单元格）：2026-09-29 01:00 UTC 用 `run.py --freeze` 冻结的副本，来自 `refactor/five-layer-architecture` 分支的 git 提交 `ff90d76a9dd9`，含未提交的修改；`ea8891eee6b1a6da`（3 个单元格）：2026-09-28 14:19 UTC 用 `run.py --freeze` 冻结的副本，来自 `refactor/five-layer-architecture` 分支的 git 提交 `ff90d76a9dd9`，含未提交的修改。
- 每个单元格的限制：30 分钟、常驻内存 2,500 MB；超出者被终止并如实标注（不会被删除）。
- 记录在案的运行开销：349 个单元格（完成 347 个，因限制中止 2 个，失败 0 个），墙钟时间共 2.2 小时、CPU 时间共 6.2 小时；已完成单元格的最大峰值内存 2,290 MB。被替换的运行（重复运行，以及内存限流后重做的尝试）不计入。
- 计时：*训练 s* 与 *预测 s* 为命令行自身围绕 `fit` 与 `localize` 的计时（含预处理），对各折求和。*峰值 MB* 为该单元格进程的最大常驻内存（`ru_maxrss`），包含 Python、导入的模块和已加载的数据集。
- 内存限流：单元格运行在与其他进程共享的 Linux 控制组中（memory.high 4,096 MB，memory.max 6,144 MB，swap 4,096 MB）。超过 memory.high 时内核会减慢每次内存分配，这会拉长墙钟时间，但不改变任何结果。每个单元格都记录了运行期间该控制组的内存压力停顿（`memory_stall_s`，Linux PSI）及自身 CPU 时间；标 † 的时间来自这样的单元格：运行期间控制组中有任务等待内存的时间超过其墙钟时间的 10%（PSI `some`，两种统计中更严格的一种；已完成的 347 个单元格中有 0 个）。
- 随机种子：所有协议和带随机性的方法均为 0；`PYTHONHASHSEED=0`。
- 可复现性：124 个单元格在不同进程中运行了两次（第二次替换第一次并记录 `rerun_check`）；其合并及各折指标的最大差异：0（完全相同）。
- 确定性检查：82 个单元格用 `run.py --verify` 再次运行（2026-09-29，库源码摘要 `34d0d1b7013ddf9b`, `ea8891eee6b1a6da`，比单元格记录的代码更新：这也说明此后的代码修改没有改变这些数字）；其合并及各折指标的最大差异：0（完全相同）。

## 与已知数值的交叉核对

*预期* 为本矩阵运行之前已知的数值：TUJI1 论文（Klus 等，2024，表 3）中的 1-NN 基线，以及本库的四个回归数值（UJIIndoorLoc 的两个均值由 `tests/cli/test_cli_main.py` 中的真实数据测试固定；HALOC 和 BBIL 的均值是在审查这两个加载器时测得的）。回归数值只能说明结果没有变化，因此 *库外重算* 用一个简短的纯 numpy k-NN（与 `indoorloc.methods`、`indoorloc.signals` 不共享任何代码）从加载器给出的数组重新计算每个单元格（`python -m benchmarks.crosscheck`，`results/crosscheck.json`）。

| 单元格 | 预期 | 本次测得 | 库外重算 | 一致 |
| :--- | ---: | ---: | ---: | :--- |
| UJIIndoorLoc k-NN (k=5) | 8.8084 | 8.8084 | 8.8084 | 是 |
| UJIIndoorLoc WKNN (k=5) | 8.7937 | 8.7937 | 8.7937 | 是 |
| TUJI1 1-NN，positive 表示（论文表 3） | 3.34 | 3.3430 | 3.3430 | 是 |
| HALOC WKNN (k=5)，\|CSI\| | 3.6685 | 3.6685 | 3.6685 | 是 |
| BBIL 办公室 WKNN (k=5)，缺失填 -110 dBm | 3.5189 | 3.5189 | 3.5189 | 是 |

以整数或平均 dBm 表示的读数常出现距离完全相等的情形。本库和 numpy 重算都按训练样本序号排列等距邻居；scikit-learn 1.8.0 的暴力搜索不这样做，因此在第 k 个邻居处出现并列的测试行上得到略有不同的均值（UJIIndoorLoc k-NN (k=5)：19 条测试行，8.8314；UJIIndoorLoc WKNN (k=5)：19 条测试行，8.8166；BBIL 办公室 WKNN (k=5)，缺失填 -110 dBm：34 条测试行，3.5183）。

BBIL 参考值是把缺失读数设为 -110 dBm（低于最弱的训练读数 -106 dBm）得到的；使用库默认的 -104 dBm 填充时，同一 WKNN 为 3.5173 m（见下表）。只有缺少某个接收机读数的 2,192 条训练样本受影响；测试样本没有缺失。

## 方法与预处理

规格与传给 `--method` / `--preprocess` 的字符串完全相同；未写出的参数均为库默认值，每个结果文件都保存了完整解析后的参数（`model_params`）。

| 方法 | spec | 说明 |
| :--- | :--- | :--- |
| 训练集质心（基线） | `benchmarks.baselines:TrainingCentroid` | benchmarks/baselines.py：训练位置均值，楼层/楼栋取多数 |
| 1-NN | `knn(k=1)` | Bahl & Padmanabhan，RADAR，INFOCOM 2000；距离相同时按训练样本序号排序 |
| k-NN (k=5) | `knn` | k-NN，等权，楼层/楼栋投票 |
| WKNN (k=5) | `wknn` | 反距离加权，楼层/楼栋加权投票 |
| Horus | `horus` | Youssef & Agrawala，MobiSys 2005：每个位置、每个 AP 一个高斯，取最可能 3 个位置的质心（缺失 AP 的处理见类文档） |
| 随机森林 | `rf(n_jobs=8)` | Breiman 2001；100 棵树，max_features='sqrt'，自助采样 |
| 极端随机树 | `extratrees(n_jobs=8)` | Geurts 等 2006；100 棵树，max_features='sqrt'，不自助采样 |
| 集成：WKNN/RF/Horus 中位数 | `ensemble(localizers=["wknn","rf","horus"],combine="median")` | 三个估计逐坐标取中位数，标签投票；成员由注册表按默认参数创建（森林的预测与 n_jobs 无关，但成员森林单线程运行，因此集成的时间不等于上面各行之和） |
| 集成：WKNN/RF/极端随机树 中位数 | `ensemble(localizers=["wknn","rf","extratrees"],combine="median")` | 同上；用于不运行 Horus 的 CSI 表 |
| MLP | `mlp` | 512-256-128，ReLU，BatchNorm，dropout 0.3，输入标准化；AdamW（学习率 1e-3，权重衰减 1e-4），批大小 256，最多 100 轮，在 10% 训练数据上早停（耐心 10）；CPU |
| SVM (RBF) | `svm` | 每个坐标轴一个 RBF epsilon-SVR（C=1，标准化单位下 epsilon=0.01），标签用 SVC |
| GP 无线电地图 | `gp_radiomap` | Ferris 等，RSS 2006；每个 AP 在 16 × 9 网格上选超参数；取最可能的训练位置 |
| GP 无线电地图（逐楼层） | `hierarchical(position_model=gp_radiomap)` | 先用 k-NN 投票定楼栋和楼层，再在每层各用一个 GP 无线电地图 |
| 路径损耗最大似然（已知接收机位置） | `pathloss(anchors={anchors})` | 每个接收机拟合对数距离模型，最大似然定位 |
| 路径损耗最大似然（估计接收机位置） | `pathloss` | 由有标签的训练样本联合估计接收机位置与模型参数 |
| 加权质心（已知接收机位置） | `centroid(anchors={anchors})` | 按线性接收功率加权的接收机质心（加权质心定位，Blumenthal 等 2007） |

| 预处理 | spec | 说明 |
| :--- | :--- | :--- |
| 缺失填 -104 dBm | `fill` | FillMissing(value=-104)：缺失读数设为 -104 dBm（UJIIndoorLoc 的最弱读数） |
| positive / exponential / powed 表示 | `positive` / `exponential` / `powed` | Torres-Sospedra 等，Expert Syst. Appl. 42(23)，2015；min = 训练集最低读数 - 1 dBm；α=24，β=e |
| \|CSI\| 幅度（线性） | `CSIAmplitude` | 每个天线、每个子载波的 \|H\|，展平 |
| dB 幅度，NaN 填 -15 dB | `fill(value=-15)` | 仅 csi_fingerprint：其零幅度（dB 下为 NaN） |
| 缺失填 -110 dBm | `fill(value=-110)` | 仅 BBIL 办公室：参考值所用设置（见交叉核对） |
| 原始 RSSI，NaN = 未听到 | `none` | 原始数值；缺失读数保持为 NaN（未听到）：用于 Horus（其检测模型直接为缺失读数打分）和基于模型的方法 |

<a id="ujiindoorloc"></a>

## UJIIndoorLoc (`ujiindoorloc`)

Torres-Sospedra et al., UJIIndoorLoc, IPIN 2014; doi:10.24432/C5MS59; 许可：CC BY 4.0.

<a id="ujiindoorloc-official"></a>
### UJIIndoorLoc，官方划分（训练 19,937 / 验证 1,111 条）

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 19,937 / 测试 1,111)
- 数据：`ujiindoorloc`: 训练 19,937, 测试 1,111 行; sha256 训练 `45ca0128bd12`; 测试 `5f90c536648c`
- 单位：米（Web Mercator 坐标，非地面米）
- 手动运行一个单元格：`indoorloc benchmark --dataset ujiindoorloc --protocol official --preprocess fill --method wknn --seed 0 --no-download`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 楼层 % | 楼栋 % | IPIN 分数 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 143.330 | 166.968 | 195.126 | 218.860 | 15.48 | 24.12 | 267.070 | 0.02 | <0.01 | 237 |
| 1-NN | 缺失填 -104 dBm | 9.750 | 5.955 | 11.812 | 20.610 | 90.01 | 99.55 | 15.032 | 0.05 | 0.27 | 349 |
| k-NN (k=5) | 缺失填 -104 dBm | 8.808 | 5.358 | 10.354 | 19.120 | 90.28 | 99.73 | 12.223 | 0.04 | 0.24 | 348 |
| WKNN (k=5) | 缺失填 -104 dBm | 8.794 | 5.355 | 10.329 | 19.154 | 90.46 | 99.73 | 12.201 | 0.04 | 0.24 | 349 |
| Horus | 缺失填 -104 dBm | 9.726 | 5.998 | 12.470 | 21.659 | 87.22 | 99.91 | 16.768 | 0.35 | 6.98 | 589 |
| 随机森林 | 缺失填 -104 dBm | 10.695 | 7.982 | 14.211 | 22.296 | 91.00 | 99.91 | 16.025 | 1.65 | 0.04 | 494 |
| 极端随机树 | 缺失填 -104 dBm | 10.281 | 7.627 | 13.842 | 21.486 | 90.64 | 100.00 | 15.504 | 1.89 | 0.04 | 590 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 8.331 | 5.455 | 10.784 | 18.834 | 90.46 | 99.91 | 12.645 | 8.36 | 6.60 | 976 |
| MLP | 缺失填 -104 dBm | 11.294 | 8.946 | 13.886 | 19.779 | 93.70 | 99.46 | 14.549 | 17.9 | <0.01 | 920 |
| SVM (RBF) | 缺失填 -104 dBm | 19.704 | 12.895 | 26.032 | 45.681 | 92.35 | 99.82 | 28.923 | 230 | 7.79 | 956 |
| GP 无线电地图（逐楼层） | 缺失填 -104 dBm | 10.403 | 6.750 | 13.638 | 22.376 | 90.19 | 99.73 | 16.218 | 0.67 | 0.41 | 445 |
| 1-NN | positive 表示 | 9.850 | 6.003 | 11.865 | 20.813 | 89.74 | 99.55 | 15.282 | 0.04 | 0.23 | 348 |
| WKNN (k=5) | positive 表示 | 8.858 | 5.355 | 10.313 | 19.244 | 90.10 | 99.73 | 12.297 | 0.04 | 0.22 | 348 |
| WKNN (k=5) | exponential 表示 (α=24) | **7.988** | 4.680 | 9.029 | 15.997 | 92.44 | 99.55 | 10.135 | 0.10 | 0.23 | 351 |
| WKNN (k=5) | powed 表示 (β=e) | 8.250 | 4.840 | 9.895 | 18.089 | 93.52 | 99.73 | 10.808 | 0.12 | 0.22 | 349 |
| Horus | 原始 RSSI，NaN = 未听到 | 7.994 | 4.743 | 10.181 | 18.156 | 92.17 | 99.91 | 11.409 | 0.32 | 0.23 | 550 |

**未运行**:
- pathloss, centroid: 基于模型的方法：数据集未公开 AP 位置

**说明**:
- 误差单位为 EPSG:3857（Web Mercator）米，即数据集自身的坐标，UJIIndoorLoc 的结果通常以此报告；乘以 meta['ground_scale'] = 0.7661（cos 39.99°）得到地面米（例如 WKNN 8.794 → 6.737 m）。IPIN 分数：每错一层加 15 m、楼栋错加 50 m 后的第 75 百分位。

#### 已发表结果

按作者报告、存于 `indoorloc.evaluation.literature`（IndoorLoc 未重新运行）。*核查*：`verified` = 已与论文全文核对；`corrected` = 0.1 版数值有误，此处为论文数值；`unchecked` = 来源已确认，但数值未与论文核对。

| 方法 | 报告值 | 协议 | 核查 | 出处位置 | 来源 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GBDT + sample differences | 平均 2.45 m; 楼层 99.14 % | unspecified | `unchecked` | – | Cao et al. (2021), Satellite Navigation 2:27, doi:10.1186/s43020-021-00058-8 |
| XGBoost + sample differences | 平均 3.42 m; 楼层 99.4 % | unspecified | `unchecked` | – | Cao et al. (2021), Satellite Navigation 2:27, doi:10.1186/s43020-021-00058-8 |
| P-MIMO LSTM | 平均 4.2 m; 标准差 3.2 m | subset: phones 13 and 14 in buildings 0 and 1, sequential... | `verified` | Table VI, 'All buildings' row (arXiv:1903.11703) | Hoang et al. (2019), IEEE Internet of Things Journal 6(6):10639-10651, doi:10.1109/JIOT.2019.2940368 |
| WKNN (k=3) | 平均 7.3 m; 楼层 92.5 % | unspecified | `unchecked` | – | Torres-Sospedra et al. (2014), 2014 International Conference on Indoor Positioning and Indoor Navigation (IPIN), pp. 261-270, doi:10.1109/IPIN.2014.7275492 |
| MLNN (as re-implemented by Hoang et al.) | 平均 7.5 m; 标准差 3.8 m | subset: phones 13 and 14 in buildings 0 and 1, sequential... | `verified` | Table VI, 'All buildings' row (arXiv:1903.11703) | Hoang et al. (2019), IEEE Internet of Things Journal 6(6):10639-10651, doi:10.1109/JIOT.2019.2940368 |
| DNN-WKNN hybrid | 平均 7.82 m; 楼层 95 % | unspecified | `unchecked` | – | Mao et al. (2025), Algorithms 18(1):17, doi:10.3390/a18010017 |
| k-NN (k=1), dataset baseline | 平均 7.9 m; 楼层 91.6 %; 楼栋 99.2 % | official | `unchecked` | – | Torres-Sospedra et al. (2014), 2014 International Conference on Indoor Positioning and Indoor Navigation (IPIN), pp. 261-270, doi:10.1109/IPIN.2014.7275492 |
| RADAR k-NN (as re-implemented by Hoang et al.) | 平均 8.1 m; 标准差 4.9 m | subset: phones 13 and 14 in buildings 0 and 1, sequential... | `verified` | Table VI, 'All buildings' row (arXiv:1903.11703) | Hoang et al. (2019), IEEE Internet of Things Journal 6(6):10639-10651, doi:10.1109/JIOT.2019.2940368 |
| Scalable DNN (hierarchical) | 平均 9.29 m; 楼层 91.27 %; 楼栋 99.82 % | official (inferred: the rates are multiples of 1/1111, th... | `corrected` | Table 3 (kappa=8, sigma=0.2, weighted centroid) and the text below it (arXiv:1712.01990) | Kim et al. (2018), Big Data Analytics 3:4, doi:10.1186/s41044-018-0031-2 |
| CNN (lightweight) | 平均 9.5 m; 楼层 90 %; 楼栋 99 % | unspecified | `unchecked` | – | Sinha and Hwang (2019), Electronics 8(9):989, doi:10.3390/electronics8090989 |
| CNNLoc | 平均 11.78 m; 楼层 91.35 %; 楼栋 99.91 % | unspecified | `unchecked` | – | Song et al. (2019), 2019 IEEE SmartWorld/UIC/ATC/SCALCOM/IOP/SCI, pp. 589-595, doi:10.1109/SmartWorld-UIC-ATC-SCALCOM-IOP-SCI.2019.00139 |
| CCpos (CDAE-CNN) | 平均 12.4 m; 楼层 95.3 %; 楼栋 99.6 % | unspecified | `verified` | Section 4.3 (text) and the abstract | Qin et al. (2021), Sensors 21(4):1114, doi:10.3390/s21041114 |
| 1-NN baseline (positive representation, Manhattan distance) | EvAAL 平均 8.46 m; EvAAL P75 11.72 m; 楼栋 100 %; 楼层 85.34 % | EvAAL-ETRI 2015 off-site track: trained on the public UJI... | `verified` | Table 2 | Torres-Sospedra et al. (2017), Journal of Ambient Intelligence and Smart Environments 9(2):263-279, doi:10.3233/AIS-170421 |
| Ensemble of the competitors | EvAAL 平均 6.1 m; EvAAL P75 8.24 m; 楼栋 100 %; 楼层 96.43 % | EvAAL-ETRI 2015 off-site track: trained on the public UJI... | `verified` | Table 3 | Torres-Sospedra et al. (2017), Journal of Ambient Intelligence and Smart Environments 9(2):263-279, doi:10.3233/AIS-170421 |
| HFTS | EvAAL 平均 8.49 m; EvAAL P75 11.6 m; 楼栋 100 %; 楼层 96.25 % | EvAAL-ETRI 2015 off-site track: trained on the public UJI... | `verified` | Table 3 | Torres-Sospedra et al. (2017), Journal of Ambient Intelligence and Smart Environments 9(2):263-279, doi:10.3233/AIS-170421 |
| ICSL | EvAAL 平均 7.67 m; EvAAL P75 10.87 m; 楼栋 100 %; 楼层 86.93 % | EvAAL-ETRI 2015 off-site track: trained on the public UJI... | `verified` | Table 3 | Torres-Sospedra et al. (2017), Journal of Ambient Intelligence and Smart Environments 9(2):263-279, doi:10.3233/AIS-170421 |
| MOSAIC | EvAAL 平均 11.64 m; EvAAL P75 12.12 m; 楼栋 98.65 %; 楼层 93.86 % | EvAAL-ETRI 2015 off-site track: trained on the public UJI... | `verified` | Table 3 | Torres-Sospedra et al. (2017), Journal of Ambient Intelligence and Smart Environments 9(2):263-279, doi:10.3233/AIS-170421 |
| RTLS@UM | EvAAL 平均 6.2 m; EvAAL P75 8.34 m; 楼栋 100 %; 楼层 93.74 % | EvAAL-ETRI 2015 off-site track: trained on the public UJI... | `verified` | Table 3 | Torres-Sospedra et al. (2017), Journal of Ambient Intelligence and Smart Environments 9(2):263-279, doi:10.3233/AIS-170421 |

2 条其他记录未显示 (literature/unidentified 2; `indoorloc literature <dataset> --all`).

<a id="sodindoorloc"></a>

## SODIndoorLoc (`sodindoorloc`)

Bi et al., Supplementary open dataset for WiFi indoor localization based on received signal strength, Satellite Navigation 3:25, 2022; doi:10.1186/s43020-022-00086-y; 许可：not stated in the repository; cite the paper.

<a id="sodindoorloc-official-all"></a>
### SODIndoorLoc，三栋楼合并，官方划分（训练 21,205 / 测试 2,720 条）

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 21,205 / 测试 2,720)
- 数据：`sodindoorloc`: 训练 21,205, 测试 2,720 行; sha256 训练/测试 `3 个文件`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset sodindoorloc --protocol official --preprocess fill --method wknn --seed 0 --no-download`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 楼层 % | 楼栋 % | IPIN 分数 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 636.083 | 652.511 | 667.519 | 683.519 | 69.12 | 31.62 | 731.091 | 0.03 | <0.01 | 296 |
| 1-NN | 缺失填 -104 dBm | 3.532 | 2.625 | 4.700 | 7.215 | 100.00 | 100.00 | 4.700 | 0.08 | 0.63 | 491 |
| k-NN (k=5) | 缺失填 -104 dBm | 3.355 | 2.410 | 4.454 | 6.627 | 100.00 | 100.00 | 4.454 | 0.05 | 0.68 | 491 |
| WKNN (k=5) | 缺失填 -104 dBm | 3.350 | 2.413 | 4.482 | 6.627 | 100.00 | 100.00 | 4.482 | 0.05 | 0.65 | 491 |
| Horus | 缺失填 -104 dBm | 3.360 | 2.542 | 4.517 | 6.627 | 100.00 | 100.00 | 4.517 | 0.53 | 43.8 | 912 |
| 随机森林 | 缺失填 -104 dBm | 5.674 | 2.636 | 4.586 | 11.767 | 100.00 | 100.00 | 4.586 | 1.55 | 0.04 | 479 |
| 极端随机树 | 缺失填 -104 dBm | 5.659 | 2.679 | 4.550 | 10.279 | 100.00 | 100.00 | 4.550 | 1.89 | 0.04 | 489 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 2.990 | 2.272 | 3.815 | 5.450 | 100.00 | 100.00 | 3.815 | 7.51 | 44.9 | 1,235 |
| MLP | 缺失填 -104 dBm | 16.608 | 10.649 | 18.659 | 39.026 | 100.00 | 100.00 | 18.659 | 15.7 | 0.01 | 1,090 |
| SVM (RBF) | 缺失填 -104 dBm | 24.078 | 13.745 | 31.186 | 59.210 | 100.00 | 100.00 | 31.186 | 37.4 | 3.61 | 788 |
| GP 无线电地图（逐楼层） | 缺失填 -104 dBm | 3.854 | 2.476 | 5.126 | 8.159 | 100.00 | 100.00 | 5.126 | 2.10 | 1.86 | 724 |
| 1-NN | positive 表示 | 3.535 | 2.663 | 4.722 | 7.116 | 100.00 | 100.00 | 4.722 | 0.07 | 0.64 | 491 |
| WKNN (k=5) | positive 表示 | 3.382 | 2.473 | 4.518 | 6.627 | 100.00 | 100.00 | 4.518 | 0.08 | 0.64 | 491 |
| WKNN (k=5) | exponential 表示 (α=24) | 2.900 | 2.061 | 3.783 | 5.435 | 100.00 | 100.00 | 3.783 | 0.16 | 0.67 | 542 |
| WKNN (k=5) | powed 表示 (β=e) | 2.905 | 2.074 | 3.767 | 5.553 | 100.00 | 100.00 | 3.767 | 0.16 | 0.67 | 542 |
| Horus | 原始 RSSI，NaN = 未听到 | **2.852** | 2.007 | 3.848 | 5.433 | 100.00 | 100.00 | 3.848 | 0.45 | 1.79 | 850 |

**未运行**:
- pathloss, centroid: 基于模型的方法：数据集未公开 AP 位置

**说明**:
- 每栋楼使用各自的局部坐标系：只有楼栋判断正确时位置误差才有意义（见楼栋准确率）；下面的逐楼表格不存在此问题。

<a id="sodindoorloc-official-cetc331"></a>
### SODIndoorLoc，CETC331 楼，官方划分

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 955 / 测试 840)
- 数据：`sodindoorloc` (building=CETC331): 训练 955, 测试 840 行; sha256 训练 `c5975a9e2c8f`; 测试 `3832048128c8`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset sodindoorloc --protocol official --preprocess fill --method wknn --seed 0 --no-download --dataset-option building=CETC331`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 楼层 % | IPIN 分数 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 15.245 | 12.995 | 17.759 | 33.150 | 38.10 | 31.619 | <0.01 | <0.01 | 48 |
| 1-NN | 缺失填 -104 dBm | 3.364 | 2.846 | 4.401 | 6.447 | 100.00 | 4.401 | <0.01 | 0.01 | 50 |
| k-NN (k=5) | 缺失填 -104 dBm | 2.820 | 2.149 | 3.740 | 5.168 | 100.00 | 3.740 | <0.01 | 0.02 | 50 |
| WKNN (k=5) | 缺失填 -104 dBm | 2.817 | 2.195 | 3.769 | 5.157 | 100.00 | 3.769 | <0.01 | 0.02 | 50 |
| Horus | 缺失填 -104 dBm | 3.185 | 2.661 | 4.226 | 5.946 | 100.00 | 4.226 | <0.01 | 0.34 | 65 |
| 随机森林 | 缺失填 -104 dBm | 3.574 | 2.569 | 4.212 | 7.345 | 100.00 | 4.212 | 0.71 | <0.01 | 209 |
| 极端随机树 | 缺失填 -104 dBm | 3.274 | 2.531 | 3.657 | 5.413 | 100.00 | 3.657 | 0.65 | <0.01 | 215 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 2.789 | 2.153 | 3.548 | 5.170 | 100.00 | 3.548 | 0.69 | 0.36 | 228 |
| MLP | 缺失填 -104 dBm | 3.147 | 2.774 | 4.163 | 5.670 | 100.00 | 4.163 | 2.11 | <0.01 | 815 |
| SVM (RBF) | 缺失填 -104 dBm | 3.413 | 2.782 | 4.178 | 6.178 | 100.00 | 4.178 | 0.50 | 0.04 | 194 |
| GP 无线电地图（逐楼层） | 缺失填 -104 dBm | 3.412 | 2.548 | 4.940 | 6.748 | 100.00 | 4.940 | 0.96 | 0.10 | 111 |
| 1-NN | positive 表示 | 3.368 | 2.846 | 4.449 | 6.351 | 100.00 | 4.449 | <0.01 | 0.01 | 50 |
| WKNN (k=5) | positive 表示 | 2.876 | 2.315 | 3.804 | 5.414 | 100.00 | 3.804 | <0.01 | 0.02 | 50 |
| WKNN (k=5) | exponential 表示 (α=24) | 2.486 | 1.968 | 3.343 | 4.848 | 100.00 | 3.343 | <0.01 | 0.02 | 50 |
| WKNN (k=5) | powed 表示 (β=e) | 2.458 | 1.939 | 3.228 | 4.565 | 100.00 | 3.228 | <0.01 | 0.02 | 50 |
| Horus | 原始 RSSI，NaN = 未听到 | **2.425** | 1.766 | 3.412 | 4.700 | 100.00 | 3.412 | <0.01 | 0.17 | 64 |

**未运行**:
- pathloss, centroid: 基于模型的方法：数据集未公开 AP 位置

<a id="sodindoorloc-official-hcxy"></a>
### SODIndoorLoc，HCXY 楼，官方划分

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 11,370 / 测试 860)
- 数据：`sodindoorloc` (building=HCXY): 训练 11,370, 测试 860 行; sha256 训练 `82c8ad73a382`; 测试 `76cbd00ce110`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset sodindoorloc --protocol official --preprocess fill --method wknn --seed 0 --no-download --dataset-option building=HCXY`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 36.493 | 36.078 | 42.161 | 59.860 | <0.01 | <0.01 | 113 |
| 1-NN | 缺失填 -104 dBm | 2.285 | 1.791 | 3.059 | 5.403 | 0.01 | 0.09 | 195 |
| k-NN (k=5) | 缺失填 -104 dBm | 2.325 | 1.346 | 3.116 | 5.410 | 0.02 | 0.08 | 195 |
| WKNN (k=5) | 缺失填 -104 dBm | 2.320 | 1.346 | 3.116 | 5.406 | 0.01 | 0.08 | 195 |
| Horus | 缺失填 -104 dBm | 2.660 | 1.635 | 4.209 | 5.983 | 0.09 | 0.88 | 252 |
| 随机森林 | 缺失填 -104 dBm | 2.001 | 1.576 | 2.459 | 3.757 | 0.66 | <0.01 | 283 |
| 极端随机树 | 缺失填 -104 dBm | 2.236 | 1.672 | 2.707 | 4.663 | 0.66 | <0.01 | 292 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 2.052 | 1.308 | 2.767 | 5.160 | 1.64 | 1.00 | 470 |
| MLP | 缺失填 -104 dBm | 2.079 | 1.567 | 2.794 | 4.268 | 4.92 | <0.01 | 918 |
| SVM (RBF) | 缺失填 -104 dBm | 2.682 | 2.011 | 3.674 | 5.160 | 10.8 | 0.61 | 459 |
| GP 无线电地图（逐楼层） | 缺失填 -104 dBm | 3.117 | 1.342 | 3.062 | 5.522 | 0.65 | 0.63 | 238 |
| 1-NN | positive 表示 | 2.238 | 1.791 | 3.050 | 5.403 | 0.02 | 0.08 | 190 |
| WKNN (k=5) | positive 表示 | 2.198 | 1.342 | 3.000 | 5.432 | 0.02 | 0.08 | 190 |
| WKNN (k=5) | exponential 表示 (α=24) | 1.993 | 1.410 | 2.297 | 3.762 | 0.04 | 0.09 | 196 |
| WKNN (k=5) | powed 表示 (β=e) | 1.963 | 1.458 | 2.273 | 4.243 | 0.04 | 0.08 | 196 |
| Horus | 原始 RSSI，NaN = 未听到 | **1.962** | 1.330 | 2.999 | 4.376 | 0.08 | 0.13 | 237 |

**未运行**:
- pathloss, centroid: 基于模型的方法：数据集未公开 AP 位置

<a id="sodindoorloc-official-syl"></a>
### SODIndoorLoc，SYL 楼，官方划分

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 8,880 / 测试 1,020)
- 数据：`sodindoorloc` (building=SYL): 训练 8,880, 测试 1,020 行; sha256 训练 `5a6c238eb75c`; 测试 `46c71543530e`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset sodindoorloc --protocol official --preprocess fill --method wknn --seed 0 --no-download --dataset-option building=SYL`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 19.982 | 20.026 | 26.424 | 30.747 | <0.01 | <0.01 | 100 |
| 1-NN | 缺失填 -104 dBm | 4.723 | 4.031 | 5.692 | 7.823 | 0.01 | 0.08 | 168 |
| k-NN (k=5) | 缺失填 -104 dBm | 4.665 | 3.842 | 6.121 | 7.823 | 0.01 | 0.09 | 167 |
| WKNN (k=5) | 缺失填 -104 dBm | 4.656 | 3.819 | 6.107 | 7.823 | 0.01 | 0.09 | 168 |
| Horus | 缺失填 -104 dBm | 4.096 | 3.059 | 5.161 | 7.678 | 0.08 | 0.83 | 215 |
| 随机森林 | 缺失填 -104 dBm | 3.508 | 2.891 | 3.835 | 5.849 | 0.63 | <0.01 | 255 |
| 极端随机树 | 缺失填 -104 dBm | 3.684 | 3.028 | 4.198 | 6.219 | 0.63 | <0.01 | 255 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 3.834 | 3.028 | 4.351 | 7.215 | 1.27 | 0.90 | 407 |
| MLP | 缺失填 -104 dBm | **3.460** | 2.946 | 3.943 | 5.977 | 7.46 | <0.01 | 912 |
| SVM (RBF) | 缺失填 -104 dBm | 3.851 | 3.061 | 4.668 | 6.362 | 4.36 | 0.48 | 356 |
| GP 无线电地图（逐楼层） | 缺失填 -104 dBm | 4.840 | 3.499 | 5.532 | 9.619 | 0.44 | 0.53 | 204 |
| 1-NN | positive 表示 | 4.285 | 3.650 | 5.433 | 7.542 | 0.02 | 0.08 | 165 |
| WKNN (k=5) | positive 表示 | 4.263 | 3.499 | 5.433 | 7.337 | 0.01 | 0.08 | 165 |
| WKNN (k=5) | exponential 表示 (α=24) | 4.044 | 3.059 | 5.367 | 7.083 | 0.03 | 0.09 | 168 |
| WKNN (k=5) | powed 表示 (β=e) | 3.928 | 3.059 | 5.258 | 6.841 | 0.03 | 0.09 | 167 |
| Horus | 原始 RSSI，NaN = 未听到 | 3.568 | 2.892 | 4.575 | 5.453 | 0.06 | 0.13 | 203 |

**未运行**:
- pathloss, centroid: 基于模型的方法：数据集未公开 AP 位置

#### 已发表结果

按作者报告、存于 `indoorloc.evaluation.literature`（IndoorLoc 未重新运行）。*核查*：`verified` = 已与论文全文核对；`corrected` = 0.1 版数值有误，此处为论文数值；`unchecked` = 来源已确认，但数值未与论文核对。

| 方法 | 报告值 | 协议 | 核查 | 出处位置 | 来源 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MLP regression | 平均 2.3 m | unspecified | `unchecked` | – | Bi et al. (2022), Satellite Navigation 3:25, doi:10.1186/s43020-022-00086-y |
| k-NN regression | 平均 2.8 m | unspecified | `unchecked` | – | Bi et al. (2022), Satellite Navigation 3:25, doi:10.1186/s43020-022-00086-y |
| Random forest regression | 平均 3.1 m | unspecified | `unchecked` | – | Bi et al. (2022), Satellite Navigation 3:25, doi:10.1186/s43020-022-00086-y |
| SVM regression | 平均 3.5 m | unspecified | `unchecked` | – | Bi et al. (2022), Satellite Navigation 3:25, doi:10.1186/s43020-022-00086-y |
| FasterKAN | 平均 3.56 m; 楼层 99 %; 楼栋 99 % | unspecified | `unchecked` | – | Feng et al. (2024), Engineered Science, doi:10.30919/es1289 |
| WKNN (k=5) | 平均 4.2 m | unspecified | `unchecked` | – | Bi et al. (2022), Satellite Navigation 3:25, doi:10.1186/s43020-022-00086-y |
| k-NN (k=1) | 平均 5.1 m | unspecified | `unchecked` | – | Bi et al. (2022), Satellite Navigation 3:25, doi:10.1186/s43020-022-00086-y |

<a id="tampere"></a>

## Tampere (Wi-Fi crowdsourced fingerprints) (`tampere`)

Lohan et al., Wi-Fi Crowdsourced Fingerprinting Dataset for Indoor Positioning, Data 2(4):32, 2017, doi:10.3390/data2040032; doi:10.5281/zenodo.889798; 许可：CC BY 4.0 (data, FINGERPRINTING_DB/README.txt); MIT (software).

<a id="tampere-official"></a>
### Tampere 众包数据，官方划分（训练 697 / 测试 3,951 条），三维误差

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 697 / 测试 3,951)
- 数据：`tampere`: 训练 697, 测试 3,951 行; sha256 训练/测试 `4 个文件`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset tampere --protocol official --preprocess fill --method wknn --seed 0 --no-download`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 楼层 % | IPIN 分数 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 33.208 | 30.363 | 44.807 | 53.790 | 32.02 | 68.976 | <0.01 | <0.01 | 116 |
| 1-NN | 缺失填 -104 dBm | 9.436 | 6.474 | 11.637 | 20.689 | 91.60 | 13.783 | <0.01 | 0.10 | 153 |
| k-NN (k=5) | 缺失填 -104 dBm | 9.816 | 6.858 | 12.206 | 20.463 | 88.89 | 14.671 | <0.01 | 0.13 | 154 |
| WKNN (k=5) | 缺失填 -104 dBm | 9.300 | 6.592 | 11.512 | 18.897 | 90.03 | 13.439 | <0.01 | 0.14 | 154 |
| Horus | 缺失填 -104 dBm | 9.382 | 6.457 | 11.598 | 20.463 | 91.57 | 13.673 | 0.04 | 36.9 | 226 |
| 随机森林 | 缺失填 -104 dBm | 8.941 | 6.772 | 10.855 | 17.019 | 94.20 | 11.677 | 0.69 | 0.07 | 282 |
| 极端随机树 | 缺失填 -104 dBm | **8.081** | 6.025 | 9.767 | 15.871 | 94.31 | 10.572 | 0.66 | 0.08 | 289 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 8.191 | 5.840 | 9.744 | 16.309 | 93.07 | 10.777 | 0.83 | 37.9 | 405 |
| MLP | 缺失填 -104 dBm | 8.985 | 7.277 | 11.433 | 16.463 | 95.24 | 12.063 | 1.58 | 0.03 | 950 |
| SVM (RBF) | 缺失填 -104 dBm | 9.711 | 7.007 | 11.899 | 19.199 | 92.99 | 12.883 | 0.68 | 1.91 | 311 |
| GP 无线电地图（逐楼层） | 缺失填 -104 dBm | 13.004 | 7.822 | 15.909 | 29.406 | 88.89 | 18.820 | 0.79 | 2.14 | 211 |
| 1-NN | positive 表示 | 9.323 | 6.257 | 11.366 | 20.521 | 92.03 | 13.306 | <0.01 | 0.10 | 165 |
| WKNN (k=5) | positive 表示 | 9.363 | 6.538 | 11.617 | 19.258 | 90.53 | 13.389 | <0.01 | 0.14 | 166 |
| WKNN (k=5) | exponential 表示 (α=24) | 9.652 | 6.667 | 11.414 | 19.608 | 91.04 | 12.940 | <0.01 | 0.16 | 193 |
| WKNN (k=5) | powed 表示 (β=e) | 10.591 | 7.186 | 12.452 | 21.470 | 89.07 | 14.651 | <0.01 | 0.17 | 193 |
| Horus | 原始 RSSI，NaN = 未听到 | 13.067 | 7.277 | 16.567 | 33.313 | 85.50 | 20.636 | 0.03 | 1.85 | 214 |

**未运行**:
- pathloss, centroid: 基于模型的方法：数据集未公开 AP 位置

**说明**:
- 位置为 (x, y, z)，z 为楼层高度（每层 3.7 m），因此误差均为数据集基准软件使用的三维误差；楼层准确率按 floor = round(z / 3.7) 计算。

#### 已发表结果

该数据集没有存储可追溯来源且有数值的已发表结果 (8 条其他记录未显示: literature/unidentified 8).

<a id="tuji1"></a>

## TUJI1 (`tuji1`)

Klus et al., TUJI1 Dataset: Multi-device dataset for indoor localization with high measurement density, Data in Brief 54:110356, 2024, doi:10.1016/j.dib.2024.110356; doi:10.5281/zenodo.7641701; 许可：CC BY 4.0.

<a id="tuji1-official"></a>
### TUJI1，官方划分（训练 6,752 / 测试 2,147 条，五台设备，单层）

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 6,752 / 测试 2,147)
- 数据：`tuji1`: 训练 6,752, 测试 2,147 行; sha256 训练/测试 `3 个文件`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset tuji1 --protocol official --preprocess fill --method wknn --seed 0 --no-download`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 9.261 | 9.371 | 12.297 | 15.048 | <0.01 | <0.01 | 79 |
| 1-NN | 缺失填 -104 dBm | 3.424 | 2.825 | 4.787 | 6.916 | <0.01 | 0.13 | 134 |
| k-NN (k=5) | 缺失填 -104 dBm | 2.757 | 2.379 | 3.662 | 5.285 | <0.01 | 0.13 | 135 |
| WKNN (k=5) | 缺失填 -104 dBm | 2.733 | 2.349 | 3.630 | 5.264 | <0.01 | 0.13 | 135 |
| Horus | 缺失填 -104 dBm | 2.960 | 2.165 | 3.934 | 6.137 | 0.06 | 9.34 | 187 |
| 随机森林 | 缺失填 -104 dBm | 2.348 | 2.136 | 3.073 | 4.154 | 0.64 | 0.03 | 312 |
| 极端随机树 | 缺失填 -104 dBm | 2.451 | 2.215 | 3.223 | 4.360 | 0.66 | 0.03 | 355 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 2.304 | 1.977 | 3.056 | 4.254 | 1.39 | 9.59 | 430 |
| MLP | 缺失填 -104 dBm | 2.156 | 1.821 | 2.804 | 4.061 | 6.62 | <0.01 | 880 |
| SVM (RBF) | 缺失填 -104 dBm | 2.368 | 2.093 | 3.134 | 4.281 | 5.66 | 2.67 | 419 |
| GP 无线电地图 | 缺失填 -104 dBm | 3.402 | 2.696 | 4.713 | 7.359 | 10.5 | 4.00 | 279 |
| 1-NN | positive 表示 | 3.343 | 2.698 | 4.557 | 6.794 | <0.01 | 0.11 | 143 |
| WKNN (k=5) | positive 表示 | 2.656 | 2.257 | 3.580 | 5.107 | <0.01 | 0.13 | 142 |
| WKNN (k=5) | exponential 表示 (α=24) | 2.394 | 2.021 | 3.180 | 4.635 | 0.02 | 0.13 | 134 |
| WKNN (k=5) | powed 表示 (β=e) | 2.488 | 2.117 | 3.313 | 4.675 | 0.02 | 0.13 | 134 |
| Horus | 原始 RSSI，NaN = 未听到 | **2.126** | 1.614 | 2.856 | 4.334 | 0.05 | 0.65 | 179 |

**未运行**:
- pathloss, centroid: 基于模型的方法：数据集未公开 AP 位置

**说明**:
- 单层且 z 恒定，二维误差即论文中的三维误差。论文 1-NN 基线（positive 表示、欧氏距离）为 3.34 m；此处 1-NN + positive 仅从训练集学习缺失值（最小值 - 1 dBm）。

#### 已发表结果

按作者报告、存于 `indoorloc.evaluation.literature`（IndoorLoc 未重新运行）。*核查*：`verified` = 已与论文全文核对；`corrected` = 0.1 版数值有误，此处为论文数值；`unchecked` = 来源已确认，但数值未与论文核对。

| 方法 | 报告值 | 协议 | 核查 | 出处位置 | 来源 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1-NN (positive representation, Euclidean distance) | 三维平均 3.34 m | official | `corrected` | Table 3 ('1 NN') | Klus et al. (2024), Data in Brief 54:110356, doi:10.1016/j.dib.2024.110356 |
| k-NN, tuned (exponential representation, Sørensen distance, k=7) | 三维平均 2.27 m | official | `corrected` | Table 3 ('Best Coef.') | Klus et al. (2024), Data in Brief 54:110356, doi:10.1016/j.dib.2024.110356 |

2 条其他记录未显示 (indoorloc-0.1/not-literature 2; `indoorloc literature <dataset> --all`).

<a id="longtermwifi"></a>

## Long-Term WiFi (UJI library) (`longtermwifi`)

Mendoza-Silva et al., Long-Term WiFi Fingerprinting Dataset for Research on Robust Indoor Positioning, Data 3(1):3, 2018, doi:10.3390/data3010003; doi:10.5281/zenodo.3748719; 许可：CC BY 4.0 (data, Readme.txt); MIT (scripts).

<a id="longtermwifi-official"></a>
### LongTermWiFi，25 个月合并：全部训练集对全部测试集（23,040 / 81,120 条）

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 23,040 / 测试 81,120)
- 数据：`longtermwifi`: 训练 23,040, 测试 81,120 行; sha256 训练/测试 `0a74d814be48`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset longtermwifi --protocol official --preprocess fill --method wknn --seed 0 --no-download`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 楼层 % | IPIN 分数 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 5.004 | 5.144 | 6.265 | 7.482 | 50.00 | 35.144 | 0.02 | 0.11 | 1,175 |
| 1-NN | 缺失填 -104 dBm | 2.705 | 2.195 | 4.196 | 5.390 | 99.95 | 4.196 | 0.06 | 18.5 | 1,663 |
| k-NN (k=5) | 缺失填 -104 dBm | 2.290 | 1.912 | 3.155 | 4.447 | 99.97 | 3.155 | 0.06 | 18.6 | 1,663 |
| WKNN (k=5) | 缺失填 -104 dBm | 2.288 | 1.933 | 3.157 | 4.456 | 99.97 | 3.160 | 0.05 | 18.6 | 1,663 |
| Horus | 缺失填 -104 dBm | 2.400 | 2.195 | 3.507 | 4.712 | 99.96 | 3.509 | 0.43 | 18.2 | 1,501 |
| 随机森林 | 缺失填 -104 dBm | 2.017 | 1.805 | 2.716 | 3.682 | 99.99 | 2.716 | 1.36 | 1.69 | 1,494 |
| 极端随机树 | 缺失填 -104 dBm | 2.061 | 1.842 | 2.784 | 3.756 | 99.99 | 2.784 | 1.73 | 1.94 | 1,643 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | **1.950** | 1.737 | 2.692 | 3.724 | 99.99 | 2.693 | 6.85 | 38.7 | 1,928 |
| MLP | 缺失填 -104 dBm | 2.040 | 1.584 | 2.833 | 4.229 | 99.95 | 2.836 | 31.4 | 0.50 | 2,290 |
| SVM (RBF) | 缺失填 -104 dBm | 2.086 | 1.802 | 2.860 | 3.991 | 99.98 | 2.861 | 258 | 1,037 | 1,809 |
| GP 无线电地图（逐楼层） | 缺失填 -104 dBm | 2.554 | 2.195 | 3.832 | 4.980 | 99.97 | 3.832 | 0.40 | 21.7 | 1,857 |
| 1-NN | positive 表示 | 2.611 | 2.195 | 3.832 | 5.390 | 99.96 | 3.832 | 0.08 | 18.6 | 1,663 |
| WKNN (k=5) | positive 表示 | 2.240 | 1.865 | 3.084 | 4.455 | 99.98 | 3.087 | 0.08 | 19.0 | 1,663 |
| WKNN (k=5) | exponential 表示 (α=24) | 2.332 | 1.892 | 3.379 | 4.740 | 99.97 | 3.385 | 0.17 | 19.2 | 1,992 |
| WKNN (k=5) | powed 表示 (β=e) | 2.589 | 2.193 | 3.595 | 5.373 | 99.73 | 3.606 | 0.17 | 19.1 | 1,992 |
| Horus | 原始 RSSI，NaN = 未听到 | 2.155 | 1.936 | 3.383 | 4.407 | 100.00 | 3.383 | 0.42 | 1.87 | 1,470 |

**未运行**:
- pathloss, centroid: 基于模型的方法：数据集未公开 AP 位置

**说明**:
- 训练位置（24 个）和测试位置（106 个）每月固定重复；合并各月会把 25 个月的信号漂移混入同一个模型。
- SVM 单元格是整个矩阵中最慢的（训练 4.3 分钟，预测 81,120 条测试样本 17 分钟）。此前两次尝试在共享控制组内存受限流时触及 30 分钟限制；记录下的这次运行没有内存停顿。

<a id="longtermwifi-within-month"></a>
### LongTermWiFi，作者协议：每月内训练与测试（25 折）

- 协议：`benchmarks.protocols:WITHIN_MONTH`: 数据集作者的协议：每个月 m 用该月的训练集训练、该月的测试集测试 (25 折，共 81,120 条测试行；每条测试行只测试一次；指标对所有折合并)
- 数据：`longtermwifi`: 训练 23,040, 测试 81,120 行; sha256 训练/测试 `0a74d814be48`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset longtermwifi --protocol benchmarks.protocols:WITHIN_MONTH --preprocess fill --method wknn --seed 0 --no-download`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 楼层 % | IPIN 分数 | 各折平均（最小–最大） | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 5.004 | 5.144 | 6.265 | 7.482 | 50.00 | 35.144 | 5.00–5.00 | 0.02 | 0.06 | 983 |
| 1-NN | 缺失填 -104 dBm | 2.972 | 2.370 | 4.390 | 5.678 | 99.81 | 4.390 | 2.62–4.12 | 0.04 | 2.24 | 992 |
| k-NN (k=5) | 缺失填 -104 dBm | 2.533 | 2.202 | 3.512 | 4.734 | 99.85 | 3.512 | 2.22–3.69 | 0.04 | 2.75 | 991 |
| WKNN (k=5) | 缺失填 -104 dBm | 2.529 | 2.200 | 3.470 | 4.738 | 99.85 | 3.476 | 2.22–3.69 | 0.04 | 2.65 | 992 |
| Horus | 缺失填 -104 dBm | 2.597 | 2.216 | 3.829 | 4.980 | 99.91 | 3.830 | 2.20–3.71 | 0.26 | 18.3 | 983 |
| 随机森林 | 缺失填 -104 dBm | 2.444 | 2.274 | 3.283 | 4.232 | 99.95 | 3.284 | 1.98–3.43 | 6.29 | 0.79 | 1,144 |
| 极端随机树 | 缺失填 -104 dBm | 2.427 | 2.257 | 3.269 | 4.200 | 99.94 | 3.271 | 2.02–3.24 | 4.16 | 0.88 | 1,209 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | **2.259** | 2.029 | 3.097 | 4.241 | 99.95 | 3.100 | 1.91–3.36 | 6.08 | 23.1 | 1,088 |
| MLP | 缺失填 -104 dBm | 2.453 | 2.244 | 3.320 | 4.394 | 99.88 | 3.325 | 1.97–3.38 | 21.0 | 0.34 | 1,578 |
| SVM (RBF) | 缺失填 -104 dBm | 2.447 | 2.192 | 3.348 | 4.510 | 99.94 | 3.351 | 2.01–3.99 | 34.6 | 31.3 | 1,084 |
| GP 无线电地图（逐楼层） | 缺失填 -104 dBm | 2.623 | 2.370 | 3.832 | 4.980 | 99.85 | 3.832 | 2.27–3.74 | 0.69 | 4.40 | 994 |
| 1-NN | positive 表示 | 2.864 | 2.370 | 4.390 | 5.678 | 99.83 | 4.390 | 2.51–4.00 | 0.05 | 2.57 | 992 |
| WKNN (k=5) | positive 表示 | 2.469 | 2.083 | 3.442 | 4.725 | 99.85 | 3.453 | 2.15–3.54 | 0.05 | 3.09 | 991 |
| WKNN (k=5) | exponential 表示 (α=24) | 2.623 | 2.202 | 3.613 | 5.282 | 99.76 | 3.625 | 2.30–3.31 | 0.13 | 3.52 | 992 |
| WKNN (k=5) | powed 表示 (β=e) | 2.985 | 2.559 | 4.171 | 5.963 | 99.05 | 4.277 | 2.53–3.44 | 0.12 | 3.00 | 992 |
| Horus | 原始 RSSI，NaN = 未听到 | 2.374 | 2.105 | 3.466 | 4.740 | 99.95 | 3.466 | 2.09–2.71 | 0.21 | 1.82 | 984 |

**未运行**:
- pathloss, centroid: 基于模型的方法：数据集未公开 AP 位置

**说明**:
- 对 25 个月度折叠合并统计（每条测试样本一次）。第 1 个月有 15 个训练集，第 2-24 个月各 1 个（每个 576 条）；第 25 个月用第二部手机（Galaxy A5）重复采集了训练集和五个测试集，因此该折用两部手机的训练集（1,152 条）训练、在十个测试集（6,240 条）上测试。

#### 已发表结果

按作者报告、存于 `indoorloc.evaluation.literature`（IndoorLoc 未重新运行）。*核查*：`verified` = 已与论文全文核对；`corrected` = 0.1 版数值有误，此处为论文数值；`unchecked` = 来源已确认，但数值未与论文核对。

| 方法 | 报告值 | 协议 | 核查 | 出处位置 | 来源 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| KD-3D CNN (student) | 平均 0.84 m | unspecified | `unchecked` | – | Rizwan et al. (2025), IEEE Access 13:157764-157779, doi:10.1109/ACCESS.2025.3602462 |
| 3D separable CNN | 平均 1.04 m | unspecified | `unchecked` | – | Rizwan et al. (2025), IEEE Access 13:30274-30286, doi:10.1109/ACCESS.2025.3535948 |
| 2D-CNN with knowledge distillation | 平均 1.33 m | unspecified | `unchecked` | – | Rizwan et al. (2025), Scientific Reports 15:39078, doi:10.1038/s41598-025-25589-x |

2 条其他记录未显示 (indoorloc-0.1/not-literature 2; `indoorloc literature <dataset> --all`).

<a id="ibeacon-rssi"></a>

## iBeacon RSSI (UJI BLE RSS database) (`ibeacon_rssi`)

Mendoza-Silva, Matey-Sanz, Torres-Sospedra, Huerta, BLE RSS Measurements Dataset for Research on Accurate Indoor Positioning, Data 4(1), 12, 2019; doi:10.5281/zenodo.1618692; 许可：CC BY 4.0 (data), MIT (scripts).

<a id="ibeacon-rssi-official-all"></a>
### UJI iBeacon RSS，两个区域合并，作者默认训练/测试配置

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 2,748 / 测试 1,860，144 行未使用)
- 数据：`ibeacon_rssi`: 训练 2,748, 测试 1,860, 全部 4,752 行; sha256 训练/测试/全部 `8 个文件`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset ibeacon_rssi --protocol official --preprocess fill --method wknn --seed 0 --no-download`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 楼栋 % | IPIN 分数 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 8.699 | 8.014 | 12.014 | 14.778 | 41.94 | 62.014 | <0.01 | <0.01 | 58 |
| 1-NN | 缺失填 -104 dBm | 4.591 | 3.461 | 6.590 | 8.745 | 93.33 | 6.829 | <0.01 | 0.07 | 64 |
| k-NN (k=5) | 缺失填 -104 dBm | 3.494 | 2.617 | 4.478 | 7.171 | 92.10 | 4.575 | <0.01 | 0.05 | 64 |
| WKNN (k=5) | 缺失填 -104 dBm | 3.510 | 2.633 | 4.612 | 7.242 | 92.63 | 4.685 | <0.01 | 0.07 | 64 |
| Horus | 缺失填 -104 dBm | 3.999 | 2.765 | 5.186 | 7.774 | 95.05 | 5.186 | <0.01 | 0.05 | 58 |
| 随机森林 | 缺失填 -104 dBm | 3.280 | 2.397 | 4.033 | 6.613 | 93.28 | 4.035 | 0.71 | 0.02 | 228 |
| 极端随机树 | 缺失填 -104 dBm | 3.317 | 2.440 | 3.891 | 6.810 | 93.71 | 3.891 | 0.63 | 0.03 | 247 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | **3.271** | 2.366 | 4.126 | 6.843 | 93.28 | 4.179 | 0.86 | 0.16 | 238 |
| MLP | 缺失填 -104 dBm | 3.466 | 2.461 | 4.368 | 7.069 | 93.01 | 4.380 | 3.60 | <0.01 | 823 |
| SVM (RBF) | 缺失填 -104 dBm | 3.544 | 2.513 | 4.336 | 7.282 | 91.61 | 4.366 | 0.86 | 0.33 | 222 |
| GP 无线电地图（逐楼层） | 缺失填 -104 dBm | 4.613 | 2.976 | 6.590 | 8.677 | 92.10 | 6.973 | 0.02 | 0.08 | 66 |
| 1-NN | positive 表示 | 4.591 | 3.461 | 6.590 | 8.745 | 93.33 | 6.829 | <0.01 | 0.05 | 64 |
| WKNN (k=5) | positive 表示 | 3.510 | 2.633 | 4.612 | 7.242 | 92.63 | 4.685 | <0.01 | 0.06 | 64 |
| WKNN (k=5) | exponential 表示 (α=24) | 3.357 | 2.373 | 4.172 | 7.100 | 92.63 | 4.247 | <0.01 | 0.07 | 64 |
| WKNN (k=5) | powed 表示 (β=e) | 3.352 | 2.373 | 4.072 | 7.242 | 92.15 | 4.112 | <0.01 | 0.07 | 64 |
| Horus | 原始 RSSI，NaN = 未听到 | 3.745 | 2.658 | 4.744 | 7.740 | 92.85 | 4.753 | <0.01 | 0.03 | 58 |

**未运行**:
- pathloss, centroid: 19-27% 的测试样本听到的信标少于 3 个（11-12% 一个也没有）：基于模型的方法无法定位，而 `indoorloc benchmark` 要求每条样本都有估计

**说明**:
- 两个区域（geo = 楼栋 1，lib = 2）坐标系与信标各自独立，见逐区域表格。未听到任何信标的样本（填充后全为 -104 dBm）按数据集原样保留。

<a id="ibeacon-rssi-official-lib"></a>
### UJI iBeacon RSS，lib 区域，作者默认训练/测试配置

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 876 / 测试 1,080，144 行未使用)
- 数据：`ibeacon_rssi` (zone=lib): 训练 876, 测试 1,080, 全部 2,100 行; sha256 训练/测试/全部 `4 个文件`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset ibeacon_rssi --protocol official --preprocess fill --method wknn --seed 0 --no-download --dataset-option zone=lib`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 4.183 | 4.167 | 5.753 | 6.589 | <0.01 | <0.01 | 50 |
| 1-NN | 缺失填 -104 dBm | 3.827 | 2.377 | 4.197 | 7.154 | <0.01 | 0.03 | 51 |
| k-NN (k=5) | 缺失填 -104 dBm | 2.757 | 2.207 | 3.328 | 5.716 | <0.01 | 0.02 | 51 |
| WKNN (k=5) | 缺失填 -104 dBm | 2.753 | 2.205 | 3.345 | 5.567 | <0.01 | 0.02 | 51 |
| Horus | 缺失填 -104 dBm | 3.055 | 2.352 | 3.592 | 6.073 | <0.01 | 0.02 | 50 |
| 随机森林 | 缺失填 -104 dBm | 2.445 | 2.046 | 3.161 | 4.655 | 0.57 | <0.01 | 202 |
| 极端随机树 | 缺失填 -104 dBm | 2.447 | 2.002 | 3.171 | 4.727 | 0.54 | <0.01 | 208 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 2.420 | 2.040 | 3.089 | 4.687 | 0.59 | 0.05 | 205 |
| MLP | 缺失填 -104 dBm | 2.487 | 2.038 | 3.286 | 4.901 | 2.24 | <0.01 | 813 |
| SVM (RBF) | 缺失填 -104 dBm | **2.403** | 1.979 | 3.226 | 4.584 | 0.50 | 0.05 | 193 |
| GP 无线电地图 | 缺失填 -104 dBm | 3.627 | 2.377 | 3.475 | 6.819 | 0.01 | <0.01 | 56 |
| 1-NN | positive 表示 | 3.827 | 2.377 | 4.197 | 7.154 | <0.01 | 0.02 | 51 |
| WKNN (k=5) | positive 表示 | 2.753 | 2.205 | 3.345 | 5.567 | <0.01 | 0.03 | 50 |
| WKNN (k=5) | exponential 表示 (α=24) | 2.692 | 2.200 | 2.945 | 5.673 | <0.01 | 0.02 | 51 |
| WKNN (k=5) | powed 表示 (β=e) | 2.744 | 2.218 | 2.987 | 5.445 | <0.01 | 0.02 | 51 |
| Horus | 原始 RSSI，NaN = 未听到 | 3.064 | 2.315 | 3.364 | 6.279 | <0.01 | 0.02 | 50 |

**未运行**:
- pathloss, centroid: 见两区域合并表格

<a id="ibeacon-rssi-official-geo"></a>
### UJI iBeacon RSS，geo 区域，作者默认训练/测试配置

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 1,872 / 测试 780)
- 数据：`ibeacon_rssi` (zone=geo): 训练 1,872, 测试 780, 全部 2,652 行; sha256 训练/测试/全部 `4 个文件`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset ibeacon_rssi --protocol official --preprocess fill --method wknn --seed 0 --no-download --dataset-option zone=geo`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 4.072 | 3.739 | 5.332 | 5.898 | <0.01 | <0.01 | 48 |
| 1-NN | 缺失填 -104 dBm | 5.814 | 5.328 | 7.981 | 10.132 | <0.01 | 0.02 | 53 |
| k-NN (k=5) | 缺失填 -104 dBm | 3.924 | 3.520 | 5.150 | 7.203 | <0.01 | 0.02 | 53 |
| WKNN (k=5) | 缺失填 -104 dBm | 3.984 | 3.475 | 5.193 | 7.317 | <0.01 | 0.02 | 53 |
| Horus | 缺失填 -104 dBm | 4.322 | 3.794 | 5.877 | 7.566 | <0.01 | 0.01 | 48 |
| 随机森林 | 缺失填 -104 dBm | **2.990** | 2.659 | 3.975 | 5.505 | 0.56 | <0.01 | 214 |
| 极端随机树 | 缺失填 -104 dBm | 3.057 | 2.648 | 3.893 | 5.714 | 0.53 | <0.01 | 221 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 3.312 | 2.943 | 4.416 | 6.207 | 0.65 | 0.05 | 220 |
| MLP | 缺失填 -104 dBm | 3.316 | 3.058 | 4.402 | 5.872 | 2.07 | <0.01 | 822 |
| SVM (RBF) | 缺失填 -104 dBm | 3.266 | 2.947 | 4.313 | 5.658 | 0.60 | 0.07 | 201 |
| GP 无线电地图 | 缺失填 -104 dBm | 5.708 | 5.328 | 7.734 | 9.704 | <0.01 | <0.01 | 52 |
| 1-NN | positive 表示 | 5.451 | 4.296 | 7.356 | 10.132 | <0.01 | 0.02 | 53 |
| WKNN (k=5) | positive 表示 | 3.774 | 3.283 | 4.947 | 7.242 | <0.01 | 0.02 | 53 |
| WKNN (k=5) | exponential 表示 (α=24) | 3.630 | 3.025 | 4.707 | 7.242 | <0.01 | 0.02 | 53 |
| WKNN (k=5) | powed 表示 (β=e) | 3.618 | 2.953 | 4.698 | 7.242 | <0.01 | 0.02 | 53 |
| Horus | 原始 RSSI，NaN = 未听到 | 4.107 | 3.411 | 5.484 | 7.669 | <0.01 | <0.01 | 48 |

**未运行**:
- pathloss, centroid: 见两区域合并表格

#### 已发表结果

该数据集没有存储可追溯来源且有数值的已发表结果 (2 条其他记录未显示: literature/missing 2).

<a id="ble-indoor"></a>

## BLE Indoor (BBIL) (`ble_indoor`)

Kennedy, Spachos, Taylor, BLE beacon indoor localization dataset, Scholars Portal Dataverse, 2019; doi:10.5683/SP2/UTZTFT; 许可：MIT.

<a id="ble-indoor-official-office"></a>
### BBIL 办公室（experiment1），官方划分：train 对 test 录制（未使用 valid）

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 22,237 / 测试 5,110，3,598 行未使用)
- 数据：`ble_indoor` (room=office): 训练 22,237, 验证 3,598, 测试 5,110 行; sha256 训练/验证/测试 `32fd58a94030`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset ble_indoor --protocol official --preprocess fill --method wknn --seed 0 --no-download --dataset-option room=office`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 7.416 | 7.355 | 9.839 | 11.310 | <0.01 | <0.01 | 88 |
| 1-NN | 缺失填 -104 dBm | 4.331 | 3.796 | 6.240 | 8.196 | <0.01 | 0.69 | 198 |
| k-NN (k=5) | 缺失填 -104 dBm | 3.517 | 3.134 | 4.748 | 6.359 | <0.01 | 0.66 | 197 |
| WKNN (k=5) | 缺失填 -104 dBm | 3.517 | 3.134 | 4.749 | 6.365 | <0.01 | 0.65 | 198 |
| Horus | 缺失填 -104 dBm | 3.855 | 3.391 | 5.330 | 7.116 | 0.01 | 4.30 | 957 |
| 随机森林 | 缺失填 -104 dBm | 3.327 | 3.013 | 4.345 | 5.713 | 0.89 | 0.07 | 472 |
| 极端随机树 | 缺失填 -104 dBm | 3.360 | 3.080 | 4.344 | 5.646 | 0.76 | 0.09 | 592 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 3.389 | 3.016 | 4.508 | 6.036 | 3.11 | 5.05 | 1,348 |
| MLP | 缺失填 -104 dBm | **3.235** | 2.855 | 4.273 | 5.783 | 21.8 | <0.01 | 862 |
| SVM (RBF) | 缺失填 -104 dBm | 3.271 | 2.896 | 4.412 | 5.908 | 18.0 | 2.97 | 716 |
| GP 无线电地图 | 缺失填 -104 dBm | 超出内存 | – | – | – | – | – | – |
| 1-NN | positive 表示 | 4.330 | 3.788 | 6.252 | 8.196 | <0.01 | 0.69 | 198 |
| WKNN (k=5) | positive 表示 | 3.518 | 3.134 | 4.749 | 6.371 | <0.01 | 0.67 | 197 |
| WKNN (k=5) | exponential 表示 (α=24) | 3.460 | 3.038 | 4.685 | 6.340 | <0.01 | 0.66 | 199 |
| WKNN (k=5) | powed 表示 (β=e) | 3.478 | 3.036 | 4.723 | 6.387 | <0.01 | 0.70 | 200 |
| Horus | 原始 RSSI，NaN = 未听到 | 3.661 | 3.170 | 5.092 | 6.959 | 0.01 | 4.10 | 955 |
| 路径损耗最大似然（已知接收机位置） | 原始 RSSI，NaN = 未听到 | 5.263 | 4.086 | 6.748 | 10.457 | 0.01 | 1.26 | 93 |
| 路径损耗最大似然（估计接收机位置） | 原始 RSSI，NaN = 未听到 | 5.661 | 4.227 | 6.948 | 11.490 | 1.26 | 1.10 | 131 |
| 加权质心（已知接收机位置） | 原始 RSSI，NaN = 未听到 | 3.688 | 3.307 | 4.971 | 6.558 | <0.01 | <0.01 | 89 |
| WKNN (k=5) | 缺失填 -110 dBm | 3.519 | 3.138 | 4.757 | 6.372 | <0.01 | 0.67 | 197 |

**未运行**:
- GP 无线电地图 (缺失填 -104 dBm): 常驻内存达到 2555 MB，超过每进程 2500 MB 的限制，已终止（已用 CPU 时间 3,270 s；在 688 s 中控制组因内存停顿 34 s）

**说明**:
- 位置沿行走轨迹连续（在地标按键之间插值）：多数训练位置（实验室 69%，办公室 78%）只有一两条样本，因此 Horus 的逐位置高斯大多取方差下限，其行为近似高斯核近邻。作者报告测试录制上的平均误差和 P90。基于模型的行使用接收机位置（meta['anchors']，二维；接收机高 1.6 m），并直接使用原始 RSSI（缺失保持缺失）。
- GP 无线电地图在 7,534 个不同的训练位置上拟合，并以它们作为候选位置打分。

<a id="ble-indoor-official-lab"></a>
### BBIL 实验室（experiment2），官方划分：train 对 test 录制（未使用 valid）

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 13,238 / 测试 3,194，2,686 行未使用)
- 数据：`ble_indoor` (room=lab): 训练 13,238, 验证 2,686, 测试 3,194 行; sha256 训练/验证/测试 `32fd58a94030`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset ble_indoor --protocol official --preprocess fill --method wknn --seed 0 --no-download --dataset-option room=lab`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 3.613 | 3.586 | 4.380 | 4.888 | <0.01 | <0.01 | 70 |
| 1-NN | 缺失填 -104 dBm | 2.738 | 2.297 | 3.966 | 5.772 | <0.01 | 0.22 | 133 |
| k-NN (k=5) | 缺失填 -104 dBm | 2.182 | 1.921 | 2.933 | 4.048 | <0.01 | 0.22 | 133 |
| WKNN (k=5) | 缺失填 -104 dBm | 2.180 | 1.913 | 2.928 | 4.051 | <0.01 | 0.21 | 134 |
| Horus | 缺失填 -104 dBm | 2.479 | 2.193 | 3.423 | 4.669 | <0.01 | 1.12 | 302 |
| 随机森林 | 缺失填 -104 dBm | 2.200 | 1.995 | 2.828 | 3.838 | 0.95 | 0.05 | 361 |
| 极端随机树 | 缺失填 -104 dBm | 2.228 | 2.053 | 2.859 | 3.808 | 0.63 | 0.06 | 434 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 2.172 | 1.973 | 2.841 | 3.904 | 2.20 | 1.41 | 579 |
| MLP | 缺失填 -104 dBm | **2.014** | 1.754 | 2.612 | 3.782 | 8.41 | <0.01 | 850 |
| SVM (RBF) | 缺失填 -104 dBm | 2.122 | 1.902 | 2.857 | 3.938 | 6.67 | 1.10 | 703 |
| GP 无线电地图 | 缺失填 -104 dBm | 2.622 | 2.271 | 3.703 | 5.200 | 80.9 | 1.22 | 913 |
| 1-NN | positive 表示 | 2.738 | 2.297 | 3.966 | 5.772 | <0.01 | 0.22 | 134 |
| WKNN (k=5) | positive 表示 | 2.180 | 1.913 | 2.928 | 4.051 | <0.01 | 0.25 | 134 |
| WKNN (k=5) | exponential 表示 (α=24) | 2.173 | 1.885 | 2.901 | 4.136 | <0.01 | 0.22 | 134 |
| WKNN (k=5) | powed 表示 (β=e) | 2.192 | 1.887 | 2.930 | 4.148 | <0.01 | 0.22 | 134 |
| Horus | 原始 RSSI，NaN = 未听到 | 2.407 | 2.106 | 3.308 | 4.582 | <0.01 | 1.10 | 301 |
| 路径损耗最大似然（已知接收机位置） | 原始 RSSI，NaN = 未听到 | 5.038 | 3.575 | 5.991 | 10.781 | <0.01 | 1.08 | 74 |
| 路径损耗最大似然（估计接收机位置） | 原始 RSSI，NaN = 未听到 | 5.835 | 3.875 | 7.572 | 13.909 | 1.09 | 0.89 | 120 |
| 加权质心（已知接收机位置） | 原始 RSSI，NaN = 未听到 | 2.751 | 2.576 | 3.640 | 4.613 | <0.01 | <0.01 | 70 |

**说明**:
- 位置沿行走轨迹连续（在地标按键之间插值）：多数训练位置（实验室 69%，办公室 78%）只有一两条样本，因此 Horus 的逐位置高斯大多取方差下限，其行为近似高斯核近邻。作者报告测试录制上的平均误差和 P90。基于模型的行使用接收机位置（meta['anchors']，二维；接收机高 1.6 m），并直接使用原始 RSSI（缺失保持缺失）。
- GP 无线电地图在 3,311 个不同的训练位置上拟合，并以它们作为候选位置打分。

#### 已发表结果

该数据集没有存储可追溯来源且有数值的已发表结果 (2 条其他记录未显示: literature/missing 2).

<a id="ble-rssi-uci"></a>

## BLE RSSI (UCI, Western Michigan University library) (`ble_rssi_uci`)

Mohammadi, Al-Fuqaha, Guizani, Oh, Semisupervised Deep Reinforcement Learning in Support of IoT and Smart City Services, IEEE Internet of Things Journal 5(2), 2018; doi:10.24432/C54G80; 许可：CC BY 4.0.

<a id="ble-rssi-uci-random-80-20"></a>
### UCI BLE RSSI（Waldo 图书馆），1,420 条有标签样本随机 80/20 划分，种子 0；单位为网格

- 协议：`random-80-20`: 随机 80% 训练 / 20% 测试（设种子；对指纹法偏乐观） (训练 1,136 / 测试 284)
- 数据：`ble_rssi_uci`: 无标签 5,191, 全部 1,420 行; sha256 全部 `2be36c37b2dd`
- 单位：源地图的网格（列字母 A=1，行号向下计数）；未给出网格尺寸
- 手动运行一个单元格：`indoorloc benchmark --dataset ble_rssi_uci --protocol random-80-20 --preprocess fill --method wknn --seed 0 --no-download`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 4.907 | 4.371 | 6.034 | 8.454 | <0.01 | <0.01 | 46 |
| 1-NN | 缺失填 -104 dBm | 1.673 | 1.000 | 2.871 | 4.243 | <0.01 | <0.01 | 48 |
| k-NN (k=5) | 缺失填 -104 dBm | 1.567 | 1.208 | 2.243 | 3.219 | <0.01 | 0.01 | 48 |
| WKNN (k=5) | 缺失填 -104 dBm | 1.549 | 1.211 | 2.408 | 3.308 | <0.01 | <0.01 | 48 |
| Horus | 缺失填 -104 dBm | 1.730 | 1.443 | 2.355 | 3.596 | <0.01 | <0.01 | 46 |
| 随机森林 | 缺失填 -104 dBm | 1.539 | 1.282 | 2.360 | 3.314 | 0.53 | <0.01 | 204 |
| 极端随机树 | 缺失填 -104 dBm | **1.474** | 1.169 | 2.336 | 3.186 | 0.53 | <0.01 | 205 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 1.520 | 1.244 | 2.376 | 3.254 | 0.53 | 0.02 | 204 |
| MLP | 缺失填 -104 dBm | 1.550 | 1.365 | 2.143 | 2.792 | 2.18 | <0.01 | 811 |
| SVM (RBF) | 缺失填 -104 dBm | 1.587 | 1.216 | 2.206 | 3.150 | 0.50 | <0.01 | 193 |
| GP 无线电地图 | 缺失填 -104 dBm | 1.909 | 1.414 | 2.828 | 4.207 | 0.03 | <0.01 | 55 |
| 1-NN | positive 表示 | 1.630 | 1.000 | 2.828 | 4.243 | <0.01 | <0.01 | 48 |
| WKNN (k=5) | positive 表示 | 1.521 | 1.162 | 2.408 | 3.293 | <0.01 | <0.01 | 48 |
| WKNN (k=5) | exponential 表示 (α=24) | 1.515 | 1.142 | 2.366 | 3.293 | <0.01 | <0.01 | 48 |
| WKNN (k=5) | powed 表示 (β=e) | 1.599 | 1.288 | 2.415 | 3.404 | <0.01 | <0.01 | 48 |
| Horus | 原始 RSSI，NaN = 未听到 | 1.570 | 1.201 | 2.223 | 3.267 | <0.01 | <0.01 | 46 |

**未运行**:
- pathloss, centroid: 信标位置只以地图图片给出，没有坐标

**说明**:
- 误差单位为数据附带地图的网格（未给出网格尺寸）。同一网格的连续扫描几乎重复，随机划分偏乐观。未使用 5,191 条无标签样本。

#### 已发表结果

按作者报告、存于 `indoorloc.evaluation.literature`（IndoorLoc 未重新运行）。*核查*：`verified` = 已与论文全文核对；`corrected` = 0.1 版数值有误，此处为论文数值；`unchecked` = 来源已确认，但数值未与论文核对。

| 方法 | 报告值 | 协议 | 核查 | 出处位置 | 来源 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ANN | 准确率 81.27 % | unspecified (the paper classifies 'thirteen target classe... | `verified` | Table 2 ('Average Accuracy') | Sun et al. (2021), Sensors 21(6):1995, doi:10.3390/s21061995 |
| ANN with dropout | 准确率 80.21 % | unspecified (the paper classifies 'thirteen target classe... | `verified` | Table 2 ('Average Accuracy') | Sun et al. (2021), Sensors 21(6):1995, doi:10.3390/s21061995 |
| CNN | 准确率 91.35 % | unspecified (the paper classifies 'thirteen target classe... | `verified` | Table 2 ('Average Accuracy') | Sun et al. (2021), Sensors 21(6):1995, doi:10.3390/s21061995 |
| DNN | 准确率 81.1 % | unspecified (the paper classifies 'thirteen target classe... | `verified` | Table 2 ('Average Accuracy') | Sun et al. (2021), Sensors 21(6):1995, doi:10.3390/s21061995 |
| DNN with dropout | 准确率 80.47 % | unspecified (the paper classifies 'thirteen target classe... | `verified` | Table 2 ('Average Accuracy') | Sun et al. (2021), Sensors 21(6):1995, doi:10.3390/s21061995 |
| Decision tree | 准确率 83.95 % | unspecified (the paper classifies 'thirteen target classe... | `verified` | Table 2 ('Average Accuracy') | Sun et al. (2021), Sensors 21(6):1995, doi:10.3390/s21061995 |
| Logistic regression | 准确率 86.32 % | unspecified (the paper classifies 'thirteen target classe... | `verified` | Table 2 ('Average Accuracy') | Sun et al. (2021), Sensors 21(6):1995, doi:10.3390/s21061995 |
| MLP | 准确率 90.85 % | unspecified (the paper classifies 'thirteen target classe... | `verified` | Table 2 ('Average Accuracy') | Sun et al. (2021), Sensors 21(6):1995, doi:10.3390/s21061995 |
| Optimised CNN (improved PSO) | 准确率 97.92 % | unspecified (the paper classifies 'thirteen target classe... | `verified` | Table 2 ('Average Accuracy') | Sun et al. (2021), Sensors 21(6):1995, doi:10.3390/s21061995 |
| SVM | 准确率 88.67 % | unspecified (the paper classifies 'thirteen target classe... | `verified` | Table 2 ('Average Accuracy') | Sun et al. (2021), Sensors 21(6):1995, doi:10.3390/s21061995 |
| k-NN | 准确率 75.78 % | unspecified (the paper classifies 'thirteen target classe... | `verified` | Table 2 ('Average Accuracy') | Sun et al. (2021), Sensors 21(6):1995, doi:10.3390/s21061995 |
| k-NN | 准确率 85 % | unspecified | `unchecked` | – | Maduranga et al. (2023), Signals 4(4):651-668, doi:10.3390/signals4040036 |

2 条其他记录未显示 (indoorloc-0.1/not-literature 2; `indoorloc literature <dataset> --all`).

<a id="wlanrssi"></a>

## Wireless Indoor Localization (UCI) (`wlanrssi`)

Bhatt, Wireless Indoor Localization, UCI Machine Learning Repository, 2017, doi:10.24432/C51880; 许可：CC BY 4.0.

<a id="wlanrssi-stratified-5-fold"></a>
### UCI Wireless Indoor Localization：房间准确率，分层 5 折交叉验证，种子 0

- 协议：`stratified-kfold-5`: 按 groups['room'] 分层的 5 折（设种子）；每行测试一次 (5 折，共 2,000 条测试行；每条测试行只测试一次；指标对所有折合并)
- 数据：`wlanrssi`: 全部 2,000 行; sha256 全部 `2ae62faa2807`
- 单位：标签正确的行所占百分比
- 手动运行一个单元格：`python -m benchmarks.labels --dataset wlanrssi --label room --preprocess fill --method wknn --seed 0 --no-download`

| 方法 | 预处理 | 房间准确率 % | 各折准确率（最小–最大） | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | 缺失填 -104 dBm | 25.00 | 25.00–25.00 | <0.01 | <0.01 | 44 |
| 1-NN | 缺失填 -104 dBm | 98.55 | 98.00–99.25 | <0.01 | 0.04 | 48 |
| k-NN (k=5) | 缺失填 -104 dBm | 98.20 | 97.50–99.00 | <0.01 | 0.04 | 49 |
| WKNN (k=5) | 缺失填 -104 dBm | 98.40 | 97.75–99.00 | <0.01 | 0.04 | 49 |
| Horus | 缺失填 -104 dBm | 97.75 | 96.75–98.25 | <0.01 | 0.03 | 44 |
| 随机森林 | 缺失填 -104 dBm | 98.45 | 97.75–99.25 | 1.52 | 0.01 | 200 |
| 极端随机树 | 缺失填 -104 dBm | 98.55 | 98.00–99.00 | 1.21 | 0.02 | 204 |
| 集成：WKNN/RF/Horus 中位数 | 缺失填 -104 dBm | 98.45 | 97.75–99.00 | 1.08 | 0.09 | 200 |
| MLP | 缺失填 -104 dBm | **98.60** | 98.25–98.75 | 9.49 | <0.01 | 819 |
| SVM (RBF) | 缺失填 -104 dBm | 98.30 | 97.75–99.50 | 0.46 | <0.01 | 192 |
| 1-NN | positive 表示 | 98.55 | 98.00–99.25 | <0.01 | 0.04 | 49 |
| WKNN (k=5) | positive 表示 | 98.40 | 97.75–99.00 | <0.01 | 0.03 | 49 |
| WKNN (k=5) | exponential 表示 (α=24) | 98.10 | 97.00–99.00 | <0.01 | 0.03 | 49 |
| WKNN (k=5) | powed 表示 (β=e) | 98.00 | 97.25–98.50 | <0.01 | 0.04 | 49 |
| Horus | 原始 RSSI，NaN = 未听到 | 97.75 | 96.75–98.25 | <0.01 | 0.03 | 43 |

**未运行**:
- gp_radiomap: 数据集没有坐标，无法拟合基于位置的无线电地图
- pathloss, centroid: 基于模型的方法：数据集未公开 AP 位置

**说明**:
- 数据只有房间标签（1-4）。每个定位器把房间当作楼层标签，并使用恒定占位位置（不计分）；表格报告房间准确率。文件无缺失读数，'fill' 不改变数据，Horus 填充与否输入相同。

#### 已发表结果

该数据集没有存储可追溯来源且有数值的已发表结果 (4 条其他记录未显示: indoorloc-0.1/not-literature 3, literature/unidentified 1).

<a id="haloc"></a>

## HALOC (`haloc`)

Strohmayer, Kampel, WiFi CSI-based Long-Range Person Localization Using Directional Antennas, ICLR 2024 Tiny Papers; doi:10.5281/zenodo.10715595; 许可：CC BY 4.0 (Zenodo record; the description asks for non-commercial research use).

<a id="haloc-official"></a>
### HALOC，官方划分：序列 0-3 训练、5 测试（96,491 / 14,277 个数据包）；CSI 幅度

- 协议：`official`: 数据集自带的训练/测试文件（多数已发表结果的设置） (训练 96,491 / 测试 14,277，28,111 行未使用)
- 数据：`haloc`: 训练 96,491, 验证 28,111, 测试 14,277, 全部 138,879 行; sha256 训练/验证/测试/全部 `51183ac2d1ca`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset haloc --protocol official --preprocess CSIAmplitude --method wknn --seed 0 --no-download`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | \|CSI\| 幅度（线性） | 4.897 | 4.732 | 7.304 | 8.941 | <0.01 | <0.01 | 505 |
| 1-NN | \|CSI\| 幅度（线性） | 4.389 | 2.910 | 6.768 | 10.754 | 0.01 | 7.89 | 914 |
| k-NN (k=5) | \|CSI\| 幅度（线性） | 3.671 | 2.848 | 5.275 | 7.995 | 0.01 | 7.65 | 931 |
| WKNN (k=5) | \|CSI\| 幅度（线性） | 3.669 | 2.847 | 5.264 | 7.970 | 0.01 | 7.60 | 930 |
| 随机森林 | \|CSI\| 幅度（线性） | 3.656 | 3.042 | 5.189 | 7.463 | 3.54 | 0.34 | 1,587 |
| 极端随机树 | \|CSI\| 幅度（线性） | 3.705 | 3.119 | 5.237 | 7.520 | 2.43 | 0.42 | 2,172 |
| 集成：WKNN/RF/极端随机树 中位数 | \|CSI\| 幅度（线性） | 超出内存 | – | – | – | – | – | – |
| MLP | \|CSI\| 幅度（线性） | **2.964** | 1.978 | 4.049 | 6.869 | 98.3 | 0.02 | 960 |

**未运行**:
- horus, ensemble with Horus: 每个训练数据包的位置都不同（96,491 distinct），每个逐位置高斯模型只有一个数据包；仅测试×位置的对数似然矩阵就有 14,277 × 96,491 个 float64 = 11 GB，远超每个单元格 2,500 MB 的内存限制
- svm: 估计值，未运行：在 96,491 个训练数据包上训练一次，三个位置坐标轴（每轴一个 SVR）。按 H-WILD 会议室实测的 SVM 训练（两个坐标轴、17,596-19,503 个数据包，每次 79-103 s）以训练时间随行数平方增长推算，约需一小时，超出每个单元格 30 分钟的限制
- gp_radiomap: 位置连续（训练集 96,491 个不同位置）：拟合需在 144 个超参数网格点上各做一次 n×n 核矩阵（float64 为 74 GB）的 O(n^3) Cholesky 分解
- positive / exponential / powed: Torres-Sospedra 表示针对 dBm 的 RSSI，不适用于 CSI
- pathloss, centroid: 基于 RSSI 模型的方法；CSI 幅度不是每个锚点的接收功率
- 集成：WKNN/RF/极端随机树 中位数 (|CSI| 幅度（线性）): 常驻内存达到 2505 MB，超过每进程 2500 MB 的限制，已终止（已用 CPU 时间 40 s）

**说明**:
- 特征：52 个 L-LTF 子载波的 |H|（ESP32 原始 I/Q，未校准）。未使用序列 4（验证集）。位置为三维（z 约 1.2-1.3 m）。

#### 已发表结果

该数据集没有存储可追溯来源且有数值的已发表结果.

<a id="hwild"></a>

## H-WILD (`hwild`)

Zhang, Zhang, Wang, Li, Hu, Sun, Chen, RLoc: Towards Robust Indoor Localization by Quantifying Uncertainty, Proc. ACM IMWUT 7(4), 2023; doi:10.1145/3631437; 许可：not stated by the repository (cite the RLoc paper).

<a id="hwild-louo-conference"></a>
### H-WILD 会议室：留一用户交叉验证；4 个 AP × 3 天线 × 30 子载波的 CSI 幅度

- 协议：`benchmarks.protocols:LEAVE_ONE_USER_OUT`: 每个用户（groups['user']）轮流测试，用同一选择中的其他用户训练 (5 折，共 22,970 条测试行；每条测试行只测试一次；指标对所有折合并)
- 数据：`hwild` (environment=conference): 全部 22,970 行; sha256 全部 `36 个文件`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset hwild --protocol benchmarks.protocols:LEAVE_ONE_USER_OUT --preprocess CSIAmplitude --method wknn --seed 0 --no-download --dataset-option environment=conference`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 各折平均（最小–最大） | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | \|CSI\| 幅度（线性） | 2.408 | 2.462 | 2.941 | 3.271 | 2.05–2.52 | 0.04 | <0.01 | 399 |
| 1-NN | \|CSI\| 幅度（线性） | 1.268 | 0.857 | 1.753 | 3.048 | 1.16–1.41 | 0.10 | 3.41 | 492 |
| k-NN (k=5) | \|CSI\| 幅度（线性） | 1.058 | 0.814 | 1.412 | 2.267 | 0.89–1.21 | 0.09 | 3.41 | 488 |
| WKNN (k=5) | \|CSI\| 幅度（线性） | 1.056 | 0.811 | 1.413 | 2.265 | 0.89–1.21 | 0.09 | 3.46 | 488 |
| 随机森林 | \|CSI\| 幅度（线性） | 1.122 | 0.975 | 1.485 | 2.089 | 0.93–1.26 | 10.6 | 0.40 | 845 |
| 极端随机树 | \|CSI\| 幅度（线性） | 1.149 | 1.012 | 1.513 | 2.091 | 0.95–1.29 | 4.02 | 0.47 | 1,013 |
| 集成：WKNN/RF/极端随机树 中位数 | \|CSI\| 幅度（线性） | 1.102 | 0.953 | 1.460 | 2.064 | 0.90–1.25 | 102 | 4.60 | 1,149 |
| MLP | \|CSI\| 幅度（线性） | **0.806** | 0.600 | 1.026 | 1.706 | 0.69–0.92 | 103 | 0.08 | 1,099 |
| SVM (RBF) | \|CSI\| 幅度（线性） | 0.994 | 0.824 | 1.329 | 1.958 | 0.84–1.13 | 443 | 145 | 1,047 |

**未运行**:
- horus, ensemble with Horus: 每个训练数据包的位置都不同（22,970 packets, 22,970 distinct positions），每个逐位置高斯模型只有一个数据包；在会议室的探测（留出用户 1 的一折：训练 17,663 个、测试 200 个数据包）每个测试数据包耗时 170 ms，该房间 22,970 个测试数据包约需 65 分钟，更大的房间更久，超出 30 分钟限制
- gp_radiomap: 位置连续（训练集 17,596-19,503 个不同位置）：拟合需在 144 个超参数网格点上各做一次 n×n 核矩阵（float64 为 3.0 GB）的 O(n^3) Cholesky 分解
- positive / exponential / powed: Torres-Sospedra 表示针对 dBm 的 RSSI，不适用于 CSI
- pathloss, centroid: 基于 RSSI 模型的方法；CSI 幅度不是每个锚点的接收功率

**说明**:
- 对留一用户各折合并统计（每个数据包由从未见过该用户的模型测试一次）。包含有干扰与无干扰两类行走。

<a id="hwild-louo-laboratory"></a>
### H-WILD 实验室：留一用户交叉验证；4 个 AP × 3 天线 × 30 子载波的 CSI 幅度

- 协议：`benchmarks.protocols:LEAVE_ONE_USER_OUT`: 每个用户（groups['user']）轮流测试，用同一选择中的其他用户训练 (5 折，共 26,833 条测试行；每条测试行只测试一次；指标对所有折合并)
- 数据：`hwild` (environment=laboratory): 全部 26,833 行; sha256 全部 `40 个文件`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset hwild --protocol benchmarks.protocols:LEAVE_ONE_USER_OUT --preprocess CSIAmplitude --method wknn --seed 0 --no-download --dataset-option environment=laboratory`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 各折平均（最小–最大） | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | \|CSI\| 幅度（线性） | 2.501 | 2.543 | 3.150 | 3.669 | 2.09–2.73 | 0.05 | 0.01 | 438 |
| 1-NN | \|CSI\| 幅度（线性） | 1.906 | 1.579 | 2.566 | 3.811 | 1.84–1.99 | 0.11 | 4.87 | 570 |
| k-NN (k=5) | \|CSI\| 幅度（线性） | 1.545 | 1.333 | 2.048 | 2.901 | 1.44–1.65 | 0.11 | 5.04 | 568 |
| WKNN (k=5) | \|CSI\| 幅度（线性） | 1.544 | 1.330 | 2.044 | 2.899 | 1.44–1.65 | 0.11 | 5.09 | 569 |
| 随机森林 | \|CSI\| 幅度（线性） | 1.616 | 1.488 | 2.141 | 2.849 | 1.40–1.78 | 12.4 | 0.52 | 969 |
| 极端随机树 | \|CSI\| 幅度（线性） | 1.640 | 1.530 | 2.170 | 2.855 | 1.41–1.81 | 4.76 | 0.58 | 1,142 |
| 集成：WKNN/RF/极端随机树 中位数 | \|CSI\| 幅度（线性） | 1.590 | 1.465 | 2.106 | 2.807 | 1.37–1.76 | 117 | 5.80 | 1,276 |
| MLP | \|CSI\| 幅度（线性） | **1.168** | 0.989 | 1.493 | 2.160 | 1.06–1.27 | 122 | 0.07 | 1,131 |
| SVM (RBF) | \|CSI\| 幅度（线性） | 1.354 | 1.194 | 1.797 | 2.477 | 1.24–1.50 | 679 | 204 | 1,097 |

**未运行**:
- horus, ensemble with Horus: 每个训练数据包的位置都不同（26,833 packets, 26,833 distinct positions），每个逐位置高斯模型只有一个数据包；在会议室的探测（留出用户 1 的一折：训练 17,663 个、测试 200 个数据包）每个测试数据包耗时 170 ms，该房间 22,970 个测试数据包约需 65 分钟，更大的房间更久，超出 30 分钟限制
- gp_radiomap: 位置连续（训练集 21,434-21,535 个不同位置）：拟合需在 144 个超参数网格点上各做一次 n×n 核矩阵（float64 为 3.7 GB）的 O(n^3) Cholesky 分解
- positive / exponential / powed: Torres-Sospedra 表示针对 dBm 的 RSSI，不适用于 CSI
- pathloss, centroid: 基于 RSSI 模型的方法；CSI 幅度不是每个锚点的接收功率

**说明**:
- 对留一用户各折合并统计（每个数据包由从未见过该用户的模型测试一次）。包含有干扰与无干扰两类行走。

<a id="hwild-louo-office"></a>
### H-WILD 办公室：留一用户交叉验证；4 个 AP × 3 天线 × 30 子载波的 CSI 幅度

- 协议：`benchmarks.protocols:LEAVE_ONE_USER_OUT`: 每个用户（groups['user']）轮流测试，用同一选择中的其他用户训练 (5 折，共 26,935 条测试行；每条测试行只测试一次；指标对所有折合并)
- 数据：`hwild` (environment=office): 全部 26,935 行; sha256 全部 `40 个文件`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset hwild --protocol benchmarks.protocols:LEAVE_ONE_USER_OUT --preprocess CSIAmplitude --method wknn --seed 0 --no-download --dataset-option environment=office`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 各折平均（最小–最大） | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | \|CSI\| 幅度（线性） | 2.727 | 2.577 | 3.769 | 4.707 | 2.43–3.12 | 0.04 | 0.01 | 434 |
| 1-NN | \|CSI\| 幅度（线性） | 2.442 | 2.069 | 3.669 | 4.858 | 2.32–2.53 | 0.11 | 4.63 | 545 |
| k-NN (k=5) | \|CSI\| 幅度（线性） | 2.005 | 1.747 | 2.733 | 3.803 | 1.82–2.13 | 0.11 | 4.79 | 554 |
| WKNN (k=5) | \|CSI\| 幅度（线性） | 2.003 | 1.745 | 2.732 | 3.798 | 1.82–2.13 | 0.10 | 4.75 | 554 |
| 随机森林 | \|CSI\| 幅度（线性） | 1.943 | 1.745 | 2.601 | 3.507 | 1.73–2.13 | 12.8 | 0.49 | 960 |
| 极端随机树 | \|CSI\| 幅度（线性） | 1.965 | 1.769 | 2.615 | 3.530 | 1.75–2.17 | 4.65 | 0.58 | 1,146 |
| 集成：WKNN/RF/极端随机树 中位数 | \|CSI\| 幅度（线性） | 1.933 | 1.730 | 2.590 | 3.495 | 1.72–2.12 | 124 | 5.91 | 1,280 |
| MLP | \|CSI\| 幅度（线性） | **1.601** | 1.292 | 2.149 | 3.199 | 1.51–1.65 | 112 | 0.07 | 1,120 |
| SVM (RBF) | \|CSI\| 幅度（线性） | 1.806 | 1.561 | 2.449 | 3.447 | 1.62–1.92 | 626 | 208 | 1,096 |

**未运行**:
- horus, ensemble with Horus: 每个训练数据包的位置都不同（26,935 packets, 26,935 distinct positions），每个逐位置高斯模型只有一个数据包；在会议室的探测（留出用户 1 的一折：训练 17,663 个、测试 200 个数据包）每个测试数据包耗时 170 ms，该房间 22,970 个测试数据包约需 65 分钟，更大的房间更久，超出 30 分钟限制
- gp_radiomap: 位置连续（训练集 21,537-21,564 个不同位置）：拟合需在 144 个超参数网格点上各做一次 n×n 核矩阵（float64 为 3.7 GB）的 O(n^3) Cholesky 分解
- positive / exponential / powed: Torres-Sospedra 表示针对 dBm 的 RSSI，不适用于 CSI
- pathloss, centroid: 基于 RSSI 模型的方法；CSI 幅度不是每个锚点的接收功率

**说明**:
- 对留一用户各折合并统计（每个数据包由从未见过该用户的模型测试一次）。包含有干扰与无干扰两类行走。

<a id="hwild-louo-lounge"></a>
### H-WILD 休息室：留一用户交叉验证；4 个 AP × 3 天线 × 30 子载波的 CSI 幅度

- 协议：`benchmarks.protocols:LEAVE_ONE_USER_OUT`: 每个用户（groups['user']）轮流测试，用同一选择中的其他用户训练 (8 折，共 42,554 条测试行；每条测试行只测试一次；指标对所有折合并)
- 数据：`hwild` (environment=lounge): 全部 42,554 行; sha256 全部 `56 个文件`
- 单位：米
- 手动运行一个单元格：`indoorloc benchmark --dataset hwild --protocol benchmarks.protocols:LEAVE_ONE_USER_OUT --preprocess CSIAmplitude --method wknn --seed 0 --no-download --dataset-option environment=lounge`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 各折平均（最小–最大） | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | \|CSI\| 幅度（线性） | 3.297 | 3.305 | 4.290 | 5.183 | 2.79–3.77 | 0.13 | 0.02 | 668 |
| 1-NN | \|CSI\| 幅度（线性） | 2.905 | 2.455 | 4.095 | 5.811 | 2.71–3.07 | 0.32 | 12.6 | 871 |
| k-NN (k=5) | \|CSI\| 幅度（线性） | 2.332 | 2.028 | 3.166 | 4.392 | 2.03–2.58 | 0.31 | 12.7 | 871 |
| WKNN (k=5) | \|CSI\| 幅度（线性） | 2.330 | 2.027 | 3.165 | 4.389 | 2.02–2.58 | 0.32 | 12.8 | 871 |
| 随机森林 | \|CSI\| 幅度（线性） | 2.371 | 2.178 | 3.170 | 4.214 | 2.03–2.71 | 37.3 | 0.94 | 1,430 |
| 极端随机树 | \|CSI\| 幅度（线性） | 2.423 | 2.253 | 3.230 | 4.218 | 2.07–2.77 | 13.4 | 1.12 | 1,653 |
| 集成：WKNN/RF/极端随机树 中位数 | \|CSI\| 幅度（线性） | 2.352 | 2.165 | 3.148 | 4.160 | 2.00–2.69 | 363 | 14.9 | 1,969 |
| MLP | \|CSI\| 幅度（线性） | **1.648** | 1.360 | 2.133 | 3.156 | 1.49–1.75 | 364 | 0.11 | 1,419 |

**未运行**:
- horus, ensemble with Horus: 每个训练数据包的位置都不同（42,554 packets, 42,554 distinct positions），每个逐位置高斯模型只有一个数据包；在会议室的探测（留出用户 1 的一折：训练 17,663 个、测试 200 个数据包）每个测试数据包耗时 170 ms，该房间 22,970 个测试数据包约需 65 分钟，更大的房间更久，超出 30 分钟限制
- svm: 估计值，未运行：8 次训练（每个留出用户一次），每次 37,156-37,398 个数据包。按会议室实测的 SVM 单元格（在 17,596-19,503 个数据包上每次训练 79-103 s）以训练时间随行数平方增长推算，仅训练就需约 47-50 分钟，超出每个单元格 30 分钟的限制
- gp_radiomap: 位置连续（训练集 37,156-37,398 个不同位置）：拟合需在 144 个超参数网格点上各做一次 n×n 核矩阵（float64 为 11 GB）的 O(n^3) Cholesky 分解
- positive / exponential / powed: Torres-Sospedra 表示针对 dBm 的 RSSI，不适用于 CSI
- pathloss, centroid: 基于 RSSI 模型的方法；CSI 幅度不是每个锚点的接收功率

**说明**:
- 对留一用户各折合并统计（每个数据包由从未见过该用户的模型测试一次）。包含有干扰与无干扰两类行走。

#### 已发表结果

该数据集没有存储可追溯来源且有数值的已发表结果.

<a id="csi-fingerprint"></a>

## CSI fingerprint dataset (Zhu et al., four rooms) (`csi_fingerprint`)

Zhu, Qiu, Qu, Zhou, Atiquzzaman, Wu, BLS-Location: A Wireless Fingerprint Localization Algorithm Based on Broad Learning, IEEE TMC 22(1), 2023; doi:10.1109/TMC.2021.3073005; 许可：MIT.

<a id="csi-fingerprint-points-lab"></a>
### CSI 指纹数据集，实验室：按参考点 5 折；每点前 50 个数据包

- 协议：`benchmarks.protocols:POINT_KFOLD_5`: 按参考点（groups['point']）分 5 折：每个点测试一次且从不出现在训练中（indoorloc.evaluation.kfold 的 GroupKFold 规则，设种子） (5 折，共 15,850 条测试行；每条测试行只测试一次；指标对所有折合并)
- 数据：`csi_fingerprint` (area=lab, packets=50): 全部 15,850 行; sha256 全部 `317 个文件`
- 单位：该区域参考点网格的步长（来源未给出间距）
- 手动运行一个单元格：`indoorloc benchmark --dataset csi_fingerprint --protocol benchmarks.protocols:POINT_KFOLD_5 --preprocess 'fill(value=-15)' --method wknn --seed 0 --no-download --dataset-option area=lab --dataset-option packets=50`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 各折平均（最小–最大） | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | dB 幅度，NaN 填 -15 dB | 8.485 | 8.945 | 11.008 | 12.603 | 7.77–8.86 | <0.01 | <0.01 | 157 |
| 1-NN | dB 幅度，NaN 填 -15 dB | 10.561 | 10.050 | 14.036 | 19.105 | 9.97–11.96 | 0.02 | 1.34 | 182 |
| k-NN (k=5) | dB 幅度，NaN 填 -15 dB | 10.322 | 10.050 | 13.601 | 18.439 | 9.59–11.72 | 0.02 | 1.34 | 182 |
| WKNN (k=5) | dB 幅度，NaN 填 -15 dB | 10.320 | 10.050 | 13.613 | 18.439 | 9.59–11.72 | 0.02 | 1.39 | 182 |
| Horus | dB 幅度，NaN 填 -15 dB | 10.200 | 9.849 | 13.600 | 18.279 | 9.55–11.58 | 0.13 | 2.75 | 159 |
| 随机森林 | dB 幅度，NaN 填 -15 dB | **7.884** | 7.760 | 10.520 | 12.754 | 7.64–8.12 | 3.43 | 0.08 | 319 |
| 极端随机树 | dB 幅度，NaN 填 -15 dB | 7.938 | 7.960 | 10.504 | 12.590 | 7.62–8.26 | 0.95 | 0.08 | 322 |
| 集成：WKNN/RF/Horus 中位数 | dB 幅度，NaN 填 -15 dB | 9.948 | 9.537 | 13.326 | 17.593 | 9.31–11.20 | 21.1 | 4.04 | 332 |
| MLP | dB 幅度，NaN 填 -15 dB | 8.603 | 7.933 | 11.511 | 14.805 | 8.40–8.82 | 86.1 | 0.02 | 877 |
| SVM (RBF) | dB 幅度，NaN 填 -15 dB | 9.305 | 8.789 | 12.140 | 15.808 | 8.92–9.68 | 40.3 | 12.1 | 742 |
| GP 无线电地图 | dB 幅度，NaN 填 -15 dB | 10.630 | 10.050 | 14.036 | 19.105 | 9.88–12.08 | 0.92 | 2.24 | 165 |

**未运行**:
- positive / exponential / powed: Torres-Sospedra 表示针对 dBm 的 RSSI，不适用于 CSI
- pathloss, centroid: 基于 RSSI 模型的方法；CSI 幅度不是每个锚点的接收功率

**说明**:
- 误差单位为该区域参考点网格的步长（未给出间距）。测试点都不在训练集中，因此衡量的是向新位置的插值；随机按包划分几乎为 0（只是识别参考点）。特征：3 × 30 个 dB 幅度（原样）；极少数零幅度（-inf dB，加载后为 NaN：实验室 48 个，其他三个区域没有）置为 -15 dB，低于所有存储值。

<a id="csi-fingerprint-points-meeting"></a>
### CSI 指纹数据集，会议室：按参考点 5 折；每点前 50 个数据包

- 协议：`benchmarks.protocols:POINT_KFOLD_5`: 按参考点（groups['point']）分 5 折：每个点测试一次且从不出现在训练中（indoorloc.evaluation.kfold 的 GroupKFold 规则，设种子） (5 折，共 8,800 条测试行；每条测试行只测试一次；指标对所有折合并)
- 数据：`csi_fingerprint` (area=meeting, packets=50): 全部 8,800 行; sha256 全部 `176 个文件`
- 单位：该区域参考点网格的步长（来源未给出间距）
- 手动运行一个单元格：`indoorloc benchmark --dataset csi_fingerprint --protocol benchmarks.protocols:POINT_KFOLD_5 --preprocess 'fill(value=-15)' --method wknn --seed 0 --no-download --dataset-option area=meeting --dataset-option packets=50`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 各折平均（最小–最大） | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | dB 幅度，NaN 填 -15 dB | 5.268 | 5.412 | 6.751 | 7.943 | 4.92–5.71 | <0.01 | <0.01 | 113 |
| 1-NN | dB 幅度，NaN 填 -15 dB | 6.548 | 6.000 | 9.000 | 11.662 | 6.33–6.98 | 0.01 | 0.41 | 128 |
| k-NN (k=5) | dB 幅度，NaN 填 -15 dB | 6.341 | 5.831 | 8.544 | 10.817 | 6.11–6.71 | 0.01 | 0.45 | 128 |
| WKNN (k=5) | dB 幅度，NaN 填 -15 dB | 6.342 | 5.831 | 8.544 | 10.817 | 6.11–6.71 | 0.01 | 0.43 | 130 |
| Horus | dB 幅度，NaN 填 -15 dB | 6.317 | 5.831 | 8.553 | 11.047 | 5.78–6.72 | 0.06 | 0.85 | 114 |
| 随机森林 | dB 幅度，NaN 填 -15 dB | **4.682** | 4.606 | 6.127 | 7.893 | 4.03–5.30 | 1.87 | 0.04 | 263 |
| 极端随机树 | dB 幅度，NaN 填 -15 dB | 4.727 | 4.820 | 6.264 | 7.474 | 4.08–5.27 | 0.75 | 0.05 | 265 |
| 集成：WKNN/RF/Horus 中位数 | dB 幅度，NaN 填 -15 dB | 6.094 | 5.640 | 8.433 | 10.751 | 5.61–6.28 | 10.4 | 1.35 | 274 |
| MLP | dB 幅度，NaN 填 -15 dB | 4.861 | 4.661 | 6.616 | 8.537 | 4.35–5.33 | 39.2 | 0.02 | 855 |
| SVM (RBF) | dB 幅度，NaN 填 -15 dB | 5.111 | 4.636 | 6.868 | 8.743 | 4.22–5.69 | 12.0 | 3.58 | 397 |
| GP 无线电地图 | dB 幅度，NaN 填 -15 dB | 6.612 | 6.083 | 8.602 | 10.817 | 6.11–7.08 | 0.33 | 0.71 | 136 |

**未运行**:
- positive / exponential / powed: Torres-Sospedra 表示针对 dBm 的 RSSI，不适用于 CSI
- pathloss, centroid: 基于 RSSI 模型的方法；CSI 幅度不是每个锚点的接收功率

**说明**:
- 误差单位为该区域参考点网格的步长（未给出间距）。测试点都不在训练集中，因此衡量的是向新位置的插值；随机按包划分几乎为 0（只是识别参考点）。特征：3 × 30 个 dB 幅度（原样）；极少数零幅度（-inf dB，加载后为 NaN：实验室 48 个，其他三个区域没有）置为 -15 dB，低于所有存储值。

<a id="csi-fingerprint-points-conference"></a>
### CSI 指纹数据集，报告厅：按参考点 5 折；每点前 50 个数据包

- 协议：`benchmarks.protocols:POINT_KFOLD_5`: 按参考点（groups['point']）分 5 折：每个点测试一次且从不出现在训练中（indoorloc.evaluation.kfold 的 GroupKFold 规则，设种子） (5 折，共 8,000 条测试行；每条测试行只测试一次；指标对所有折合并)
- 数据：`csi_fingerprint` (area=conference, packets=50): 全部 8,000 行; sha256 全部 `160 个文件`
- 单位：该区域参考点网格的步长（来源未给出间距）
- 手动运行一个单元格：`indoorloc benchmark --dataset csi_fingerprint --protocol benchmarks.protocols:POINT_KFOLD_5 --preprocess 'fill(value=-15)' --method wknn --seed 0 --no-download --dataset-option area=conference --dataset-option packets=50`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 各折平均（最小–最大） | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | dB 幅度，NaN 填 -15 dB | 5.094 | 5.107 | 6.698 | 7.731 | 4.52–5.37 | <0.01 | <0.01 | 109 |
| 1-NN | dB 幅度，NaN 填 -15 dB | 6.293 | 6.000 | 8.544 | 10.770 | 6.02–6.48 | <0.01 | 0.37 | 125 |
| k-NN (k=5) | dB 幅度，NaN 填 -15 dB | 5.405 | 5.099 | 7.228 | 9.051 | 5.08–5.71 | <0.01 | 0.41 | 123 |
| WKNN (k=5) | dB 幅度，NaN 填 -15 dB | 5.409 | 5.113 | 7.230 | 9.055 | 5.08–5.71 | <0.01 | 0.39 | 124 |
| Horus | dB 幅度，NaN 填 -15 dB | 5.679 | 5.307 | 7.553 | 9.829 | 5.51–5.79 | 0.06 | 0.71 | 110 |
| 随机森林 | dB 幅度，NaN 填 -15 dB | 4.964 | 4.859 | 6.412 | 7.790 | 4.53–5.24 | 1.20 | 0.07 | 297 |
| 极端随机树 | dB 幅度，NaN 填 -15 dB | 4.970 | 4.863 | 6.394 | 7.811 | 4.51–5.28 | 0.80 | 0.10 | 314 |
| 集成：WKNN/RF/Horus 中位数 | dB 幅度，NaN 填 -15 dB | 5.142 | 4.933 | 6.698 | 8.461 | 4.73–5.42 | 5.25 | 1.22 | 288 |
| MLP | dB 幅度，NaN 填 -15 dB | **4.826** | 4.581 | 6.238 | 8.004 | 4.61–5.17 | 18.0 | 0.02 | 854 |
| SVM (RBF) | dB 幅度，NaN 填 -15 dB | 5.108 | 4.999 | 6.351 | 7.808 | 4.57–5.41 | 8.96 | 3.30 | 372 |
| GP 无线电地图 | dB 幅度，NaN 填 -15 dB | 6.374 | 6.000 | 8.944 | 10.817 | 6.18–6.62 | 0.26 | 0.21 | 133 |

**未运行**:
- positive / exponential / powed: Torres-Sospedra 表示针对 dBm 的 RSSI，不适用于 CSI
- pathloss, centroid: 基于 RSSI 模型的方法；CSI 幅度不是每个锚点的接收功率

**说明**:
- 误差单位为该区域参考点网格的步长（未给出间距）。测试点都不在训练集中，因此衡量的是向新位置的插值；随机按包划分几乎为 0（只是识别参考点）。特征：3 × 30 个 dB 幅度（原样）；极少数零幅度（-inf dB，加载后为 NaN：实验室 48 个，其他三个区域没有）置为 -15 dB，低于所有存储值。

<a id="csi-fingerprint-points-minilab"></a>
### CSI 指纹数据集，小实验室：按参考点 5 折；每点前 50 个数据包

- 协议：`benchmarks.protocols:POINT_KFOLD_5`: 按参考点（groups['point']）分 5 折：每个点测试一次且从不出现在训练中（indoorloc.evaluation.kfold 的 GroupKFold 规则，设种子） (5 折，共 1,750 条测试行；每条测试行只测试一次；指标对所有折合并)
- 数据：`csi_fingerprint` (area=minilab, packets=50): 全部 1,750 行; sha256 全部 `35 个文件`
- 单位：该区域参考点网格的步长（来源未给出间距）
- 手动运行一个单元格：`indoorloc benchmark --dataset csi_fingerprint --protocol benchmarks.protocols:POINT_KFOLD_5 --preprocess 'fill(value=-15)' --method wknn --seed 0 --no-download --dataset-option area=minilab --dataset-option packets=50`

| 方法 | 预处理 | 平均 | 中位数 | P75 | P90 | 各折平均（最小–最大） | 训练 s | 预测 s | 峰值 MB |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练集质心（基线） | dB 幅度，NaN 填 -15 dB | 2.423 | 2.463 | 3.212 | 3.681 | 2.14–2.65 | <0.01 | <0.01 | 68 |
| 1-NN | dB 幅度，NaN 填 -15 dB | 2.322 | 2.000 | 3.162 | 5.000 | 1.72–2.73 | <0.01 | 0.04 | 70 |
| k-NN (k=5) | dB 幅度，NaN 填 -15 dB | 2.096 | 1.897 | 2.764 | 4.045 | 1.57–2.39 | <0.01 | 0.04 | 70 |
| WKNN (k=5) | dB 幅度，NaN 填 -15 dB | 2.092 | 1.908 | 2.729 | 4.053 | 1.57–2.39 | <0.01 | 0.05 | 71 |
| Horus | dB 幅度，NaN 填 -15 dB | 2.841 | 2.254 | 4.117 | 5.099 | 2.56–3.06 | 0.01 | 0.06 | 68 |
| 随机森林 | dB 幅度，NaN 填 -15 dB | 1.832 | 1.742 | 2.356 | 3.111 | 1.48–2.01 | 0.86 | 0.02 | 210 |
| 极端随机树 | dB 幅度，NaN 填 -15 dB | 1.811 | 1.692 | 2.420 | 3.115 | 1.44–2.00 | 0.65 | 0.02 | 216 |
| 集成：WKNN/RF/Horus 中位数 | dB 幅度，NaN 填 -15 dB | 2.016 | 1.848 | 2.613 | 3.630 | 1.68–2.23 | 2.20 | 0.16 | 211 |
| MLP | dB 幅度，NaN 填 -15 dB | **1.787** | 1.645 | 2.286 | 3.015 | 1.49–2.00 | 9.92 | <0.01 | 843 |
| SVM (RBF) | dB 幅度，NaN 填 -15 dB | 1.960 | 1.739 | 2.597 | 3.530 | 1.65–2.22 | 0.88 | 0.15 | 201 |
| GP 无线电地图 | dB 幅度，NaN 填 -15 dB | 2.902 | 2.236 | 4.123 | 5.099 | 2.62–3.20 | 0.04 | 0.03 | 86 |

**未运行**:
- positive / exponential / powed: Torres-Sospedra 表示针对 dBm 的 RSSI，不适用于 CSI
- pathloss, centroid: 基于 RSSI 模型的方法；CSI 幅度不是每个锚点的接收功率

**说明**:
- 误差单位为该区域参考点网格的步长（未给出间距）。测试点都不在训练集中，因此衡量的是向新位置的插值；随机按包划分几乎为 0（只是识别参考点）。特征：3 × 30 个 dB 幅度（原样）；极少数零幅度（-inf dB，加载后为 NaN：实验室 48 个，其他三个区域没有）置为 -15 dB，低于所有存储值。

#### 已发表结果

该数据集没有存储可追溯来源且有数值的已发表结果 (2 条其他记录未显示: literature/missing 2).

## 复现

数据集从 `$INDOORLOC_DATA`（默认 `~/.cache/indoorloc/datasets`）读取；`indoorloc info <dataset>` 列出文件及其 sha256。详见 [benchmarks/README.md](../benchmarks/README.md)。

```bash
python benchmarks/run.py                          # 整个矩阵，然后生成本文档
python benchmarks/run.py --dataset tuji1          # 一个数据集
python benchmarks/run.py --dataset sodindoorloc --table official-HCXY   # 一个表
python benchmarks/run.py --list                   # 列出每个单元格的命令
python benchmarks/run.py --dataset tuji1 --verify # 重新运行并与已存数值比较
python -m benchmarks.crosscheck                   # 在库外重算交叉核对的单元格
indoorloc benchmark --dataset tuji1 --protocol official --preprocess positive --method 'knn(k=1)'
```
