<div align="center">

<img src="assets/logo.png" alt="IndoorLoc" width="320">

<p><strong>无线室内定位，从数据到应用，每个数字都能重跑。</strong></p>
<p>Datasets · Signals · Methods · Evaluation · Applications</p>

<a href="pyproject.toml"><img src="https://img.shields.io/badge/version-0.2.0.dev0-147dd1?style=flat-square" alt="IndoorLoc 0.2.0.dev0"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-52657b?style=flat-square" alt="Apache 2.0 license"></a>
<a href="examples/readme_case/results.json"><img src="https://img.shields.io/badge/UJI_baselines-recorded_run-17846c?style=flat-square" alt="Recorded UJIIndoorLoc KNN/WKNN run"></a>
<a href="docs/benchmarks_zh.md"><img src="https://img.shields.io/badge/benchmark-349_cells-7c3aed?style=flat-square" alt="349-cell benchmark matrix"></a>

<p>
<a href="#quickstart">快速运行</a> &nbsp;·&nbsp;
<a href="#benchmarks">基准</a> &nbsp;·&nbsp;
<a href="#wireless">无线定位约定</a> &nbsp;·&nbsp;
<a href="#your-experiment">接入自己的实验</a> &nbsp;·&nbsp;
<a href="#architecture">架构与范围</a>
</p>

</div>

<p align="center">
  <a href="assets/readme/localization-zh.webp">
    <img src="assets/readme/localization-zh.webp" width="100%" alt="IndoorLoc 五层架构三维动画：数据、信号、方法、评测与应用五层叠成可展开的堆栈。一条真实的 UJIIndoorLoc 留出扫描自下而上经过前四层；应用层回放一条 ILC 2020 留出手机轨迹，显示 WiFi 定位点与贴合楼层地图的 PDR + WiFi 粒子滤波。画面中的数字均为运行记录。">
  </a>
</p>

IndoorLoc 把室内定位的数据、信号处理、方法、评测与应用连成五层，每一层都可单独使用。你可以复现公开数据上的结果、替换任一层，或把自己的数据和预测接进来。

动画把五层画成一座可展开的三维堆栈，自下而上是 **L1 数据 → L2 信号 → L3 方法 → L4 评测 → L5 应用**。镜头逐层上升，左下角的代码同步高亮对应的一行：

- **L1**：按信号类型分组的 15 个加载器（6 WiFi、3 BLE、3 CSI、1 个 IMU + WiFi 手机轨迹、2 个仿真），13 个实测数据集均带 SHA-256 校验；旁边是 UJIIndoorLoc 的 933 个测量位置。
- **L2**：留出扫描 810 的 520 维指纹，它听到 11 个 AP；未听到的读数为 NaN，由 `FillMissing(-104)` 补为 −104 dBm。
- **L3**：同一接口下的多种定位器；WKNN 找到 5 个最近指纹（来自 3 个测量位置），这条扫描误差 5.40 m，楼层正确。
- **L4**：全部 1,111 条留出扫描的误差直方图与累计曲线：平均 8.79 m、中位 5.35 m、P90 19.15 m，楼层准确率 90.46%。
- **L5**：一条 ILC 2020 留出手机轨迹画在楼层地图上：真值路点、WiFi WKNN 定位点，以及粒子不穿墙的 PDR + WiFi 粒子滤波。10 条留出轨迹、67 个路点上的平均误差：WiFi WKNN 6.95 m，离线的卡尔曼 RTS 平滑 5.20 m，在线的 PDR + WiFi + 地图 5.85 m。后两者相对 WiFi 定位的改进都超出轨迹间的波动（按轨迹 bootstrap 的 95% 区间不含 0），但只有 10 条轨迹，还分不出两者的高下。

数据使用 **UJIIndoorLoc 官方切分**（19,937 条训练、1,111 条留出）与 **ILC 2020 site1/F1**（120 条轨迹，留一轨迹）。示例扫描按固定规则选取：第 5、6 近邻距离之差大于 1e-4，5 个近邻来自至少 3 个位置，两种方法的楼层与建筑都正确，其中 KNN/WKNN 平均误差最接近全体中位数的一条。

**[示例代码](examples/readme_demo.py) · [实验协议](examples/readme_case/protocol.json) · [运行记录](examples/readme_case/results.json) · [L5 记录](examples/readme_case/apps_ilc2020.json) · [逐样本预测](examples/readme_case/predictions.npz) · [静态预览](assets/readme/localization-zh.png)**

<a id="quickstart"></a>

## 先运行一次定位

仓库附带已拟合的 KNN/WKNN（`config.json` + `arrays.npz`，不用 pickle）和完整留出集，CPU 即可运行。在 **Python 3.10+** 环境中，从仓库根目录执行：

```bash
python3 -m pip install -e .
python3 -m examples.readme_demo
```

```text
KNN   mean 8.8084  median 5.3578  P90 19.1195  floor 90.28 %  building 99.73 %  (n=1111)
WKNN  mean 8.7937  median 5.3546  P90 19.1537  floor 90.46 %  building 99.73 %  (n=1111)
```

运行后，`work_dirs/readme_demo/` 包含 `localization.html` 三维演示、逐条预测 CSV 和指标 JSON。支持 WebGL 的浏览器可以离线播放、暂停、旋转和缩放；地址后加 `?theme=dark` 切换深色，`?lang=zh` 切换中文。

<details>
<summary>重新生成 README 动画</summary>

```bash
python3 -m examples.readme_figure                     # 由运行记录生成 assets/readme/localization.html
python3 -m pip install -r examples/readme_case/render_requirements.txt
python3 -m examples.render_readme_animation assets/readme/localization.html   # 中英文 PNG 与 22 秒 WebP
```

可用 `--chromium /path/to/chromium` 指定浏览器，`--lang zh` 只导出中文，`--theme both` 另导出深色，`--preview 3 17.9` 只输出指定时刻的单帧。Three.js 0.160.1（MIT）与 Inter、JetBrains Mono、Noto Sans SC 字体子集（SIL OFL 1.1）随仓库交付。

</details>

也可以直接调用 Python：

```python
import indoorloc as iloc
from examples.readme_demo import load_example

model, test = load_example("wknn")     # 记录中的模型：FillMissing(-104) + WKNN(k=5)
pred = model.localize(test.X[[810]])   # 动画中的扫描（dBm，NaN = 未听到），形状 (1, 520)
print(pred.pos, pred.floor)
print(model.evaluate(test))            # mean 8.7937 ...，与线程数无关
```

<a id="benchmarks"></a>

## 基准

**UJIIndoorLoc · 官方训练 / 留出切分 · k = 5**

| 方法 | 平均误差 ↓ | 中位误差 ↓ | P90 误差 ↓ | 楼层准确率 ↑ | 建筑准确率 ↑ | 模型 |
|:---|---:|---:|---:|---:|---:|:---|
| KNN | 8.81 m | 5.36 m | 19.12 m | 90.28% | 99.73% | [config](examples/readme_case/knn/config.json) |
| WKNN | 8.79 m | 5.35 m | 19.15 m | 90.46% | 99.73% | [config](examples/readme_case/wknn/config.json) |

误差单位是数据集自带的 Web Mercator（EPSG:3857）米，文献按此报告；乘以 `meta["ground_scale"]` = 0.7661 得到地面米（WKNN 6.74 m）。`k = 5` 事先固定，未用留出集调参。

近邻距离相等时按训练样本序号取舍，因此结果与 BLAS/OpenMP 线程数无关。0.1 版用 scikit-learn 暴力搜索，并列取舍随线程数变化（2 个线程下 8.89 / 8.86 m）；[MIGRATION.md](MIGRATION.md) 同时复现了两版的数字。

全部数据集与方法的结果见 **[docs/benchmarks_zh.md](docs/benchmarks_zh.md)**：12 个实测数据集、349 个实验单元，每个单元都通过公开命令行在校验过的文件上运行，记录种子、库摘要、耗时与峰值内存；124 个单元在独立进程中重跑，结果完全一致。文献数字单独成表，并标注出处与核对状态。

| 实验组成 | 可查产物 |
|:---|:---|
| 数据版本、切分、预处理和模型参数 | [protocol.json](examples/readme_case/protocol.json) |
| 各项指标、耗时、运行环境 | [results.json](examples/readme_case/results.json) |
| 完整留出观测与位置真值 | [samples.npz](examples/readme_case/samples.npz) |
| 两个模型的逐样本预测 | [predictions.npz](examples/readme_case/predictions.npz) |
| L5 的 ILC 2020 轨迹与结果 | [apps_ilc2020.json](examples/readme_case/apps_ilc2020.json) |
| 全部产物的 SHA-256 | [checksums.json](examples/readme_case/checksums.json) |

<details>
<summary><strong>从原始数据重新拟合，得到同一份结果</strong></summary>

```bash
python3 -m examples.readme_demo --rebuild --output work_dirs/uji_rebuild
```

命令下载 UCI 原始文件并核对 SHA-256，拟合两个基线，保存并重新加载模型、检查预测逐位一致，再用 ILC 2020 site1/F1 重跑 L5 的留一轨迹实验。已有原始 CSV 时加 `--data-root /path/to/ujiindoorloc`。

</details>

<a id="wireless"></a>

## 无线定位，保留观测的含义

| 约定 | 本库定义 |
|:---|:---|
| **观测** | 物理单位：RSSI 为 dBm，CSI 为复数，测距为米，角度为弧度 |
| **缺测** | NaN，从不使用 100、−110 之类的哨兵值；补值由 L2 变换显式完成 |
| **坐标** | float64，保留数据集原始坐标系，并在 `meta["crs"]` 中写明（UJI 为 EPSG:3857） |
| **标签** | 楼层、建筑为整数，单独预测、单独评测 |
| **分组** | `groups` 中的 user / device / time / trajectory 等列，用于跨设备、跨时间、留一轨迹等协议 |

规则全文见 [docs/architecture/CONTRACTS.md](docs/architecture/CONTRACTS.md)。随机按行切分会把同一位置或同一条轨迹的数据同时放进训练和测试，结果偏乐观；需要泛化结论时，请使用分组协议。

<a id="your-experiment"></a>

## 接入自己的实验

每一层都接收普通数组：

```python
import numpy as np
from indoorloc.evaluation import evaluate
from indoorloc.methods import create_model

rng = np.random.default_rng(0)
X = rng.normal(-70, 8, size=(300, 6))                     # 自己的 RSSI 矩阵（dBm）
y = rng.uniform(0, 20, size=(300, 2))                     # 自己的位置（米）
model = create_model("knn", k=3).fit(X[:250], y[:250])    # L3 直接用数组
print(evaluate(y[250:], model.predict(X[250:])))          # L4 直接评测自己的预测
```

`SampleTable` 可导出为 numpy（`to_numpy()`）、pandas（`to_dataframe()`）和 PyTorch（`to_torch()`）。命令行一步完成加载、拟合、评测和记录：

```bash
indoorloc benchmark --dataset ujiindoorloc --method knn --method wknn --method horus --out uji.json
indoorloc report uji.json --format markdown
```

<a id="architecture"></a>

## 架构与覆盖范围

<p align="center">
  <img src="assets/architecture_zh.png" width="900" alt="IndoorLoc 五层架构：数据、信号、方法、评测与应用，每层列出已交付的内容。">
</p>

完整目录（15 个数据集、17 个信号变换、21 种定位方法、11 个评测协议、跟踪 / PDR / 融合 / 导航）见 [README_zh.md](README_zh.md) 与[用户指南](docs/zh/index.md)。

## 贡献与引用

欢迎贡献数据集、方法和可复现结果；要求见[扩展指南](docs/zh/extending.md)：一个类、一条注册表项、带 References 的文档字符串，以及对已知结果的测试。

```bibtex
@software{indoorloc,
  title  = {IndoorLoc: Wireless Indoor Localization in Five Layers},
  year   = {2026},
  url    = {https://github.com/qdtiger/indoorloc},
  note   = {Version 0.2}
}
```

本例使用 Torres-Sospedra 等人发布的 [UJIIndoorLoc 数据集](https://doi.org/10.24432/C5MS59)（CC BY 4.0）与微软 [Indoor Location Competition 2.0](https://github.com/location-competition/indoor-location-competition-20) 样例数据（MIT）。使用时请同时引用原始数据集。IndoorLoc 代码遵循 [Apache License 2.0](LICENSE)。
