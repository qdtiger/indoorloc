# 扩展 IndoorLoc

新的数据集、变换、方法或协议接入对应的注册表，并遵守 [CONTRACTS.md](../architecture/CONTRACTS.md) 中的规则。
本页为每一种给出一个最小的可运行示例；完整规则以契约为准，由代码评审和测试强制执行。

[指南首页](index.md) · [English](../guide/extending.md)

## 放在哪里

| 新增内容 | 文件 | 基类 | 注册位置 | 契约 |
| --- | --- | --- | --- | --- |
| 数据集 | `indoorloc/datasets/<name>.py` | `datasets._base.Dataset` | `datasets/__init__.py` 中的 `DATASETS`（包外：`register_dataset`） | [§3](../architecture/CONTRACTS.md#3-adding-a-dataset-l1) |
| 变换 | `signals/transforms.py` 或 `signals/<modality>.py` | `signals.transforms.Transform` | 从 `signals/__init__.py` 导出 | [§4](../architecture/CONTRACTS.md#4-adding-a-transform-l2) |
| 方法 | `indoorloc/methods/<family>.py` | `methods.base.BaseLocalizer` | `methods/__init__.py` 中的 `METHODS` | [§5](../architecture/CONTRACTS.md#5-adding-a-method-l3) |
| 指标、协议或性能界 | `indoorloc/evaluation/...` | 数组函数 | 命名协议注册到 `PROTOCOLS` | [§6](../architecture/CONTRACTS.md#6-adding-evaluation-code-l4) |
| 应用 | `indoorloc/apps/<topic>.py` | `core.Estimator` | 从 `apps/__init__.py` 导出 | [§7](../architecture/CONTRACTS.md#7-applications-l5) |

注册表保存的是 `"module:Class"` 字符串，列出名称时不会导入任何东西。包外的代码无需注册：`load_dataset("mypkg.data:MyDataset")`、
`create_model("mypkg.models:MyLocalizer")`、`indoorloc benchmark --dataset mypkg.data:MyDataset --method "mypkg.models:MyLocalizer(alpha=0.1)"`
和 `--protocol mypkg.splits:MY_PROTOCOL` 都可以直接解析模块路径。想用短名称时，就注册它：`register_dataset`、`register_model`
和 `register_protocol`（前两个也可以当装饰器使用）。

## 数据集

数据集由类属性（带 sha256 的文件、下载地址、meta）和一个 `_parse` 方法描述；`_parse` 返回物理单位的 `SampleTable`，缺失读数为 NaN。
按列名解析文件，并保留数据源的坐标系和单位。

```python
import csv
import numpy as np
import indoorloc as iloc
from indoorloc.core import SampleTable
from indoorloc.datasets import Dataset, sha256sum

# a small file standing in for a downloaded dataset
with open("scans.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["AP1", "AP2", "AP3", "X", "Y", "FLOOR", "PHONE"])
    w.writerows([[-45, -70, 100, 0.0, 0.0, 0, "A"], [-60, -52, -80, 4.0, 0.0, 0, "B"],
                 [100, -48, -61, 4.0, 3.0, 1, "A"], [-72, 100, -50, 0.0, 3.0, 1, "B"]])


class TinyOffice(Dataset):
    """Tiny office: four WiFi scans on two floors (example for the guide).

    References
    ----------
    Nobody, "An example dataset", Nowhere, 2026. https://example.org
    """

    name = "tiny_office"
    urls = ()                                              # the file is placed by hand here
    files = {"all": ("scans.csv", sha256sum("scans.csv"))}  # normally a literal digest
    meta = {"modality": "wifi_rssi", "units": "dBm", "crs": "local", "pos_names": ("x", "y"),
            "pos_units": "m", "license": "CC0-1.0", "citation": "Nobody, An example dataset, 2026",
            "raw_missing_value": 100}

    def _parse(self, path, split):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        aps = [c for c in rows[0] if c.startswith("AP")]                   # columns by name
        X = np.array([[float(r[a]) for a in aps] for r in rows], dtype=np.float32)
        X[X == 100] = np.nan                                                # sentinel -> NaN
        pos = np.array([[float(r["X"]), float(r["Y"])] for r in rows])
        return SampleTable(X, pos, floor=[int(r["FLOOR"]) for r in rows],
                           groups={"device": np.array([r["PHONE"] for r in rows])},
                           ids=[f"all-{i:05d}" for i in range(len(rows))], meta={"feature_names": tuple(aps)})


table = TinyOffice(root=".").load("all")
print(table.X[0], table.meta["sha256"][:12] == sha256sum("scans.csv")[:12], sorted(table.groups))
# [-45. -70.  nan] True ['device']
```

在包外，`register_dataset` 给这个类一个名称，供 `load_dataset` 和 `dataset_info` 使用（与 `register_model` 一样，
它也可以当装饰器：`@register_dataset("tiny_office")`）。不注册时，`load_dataset` 也接受类本身或它的 `"package.module:Class"` 路径
（`load_dataset("mypkg.data:TinyOffice")`）：

```python
iloc.register_dataset("tiny_office", TinyOffice)     # 也可以用 indoorloc.datasets.register_dataset
table = iloc.load_dataset("tiny_office", root=".")    # split=None：唯一的 "all" 表
print(len(table), table.meta["name"], "tiny_office" in iloc.list_datasets(),
      len(iloc.load_dataset(TinyOffice, root=".")))
# 4 tiny_office True 4
```

像这个例子一样没有 `urls` 的数据集从不下载：文件缺失时，错误信息会说明应把文件手动放在哪里，而不会建议 `download=True`。

在包内部，这个类以字符串形式注册到 `datasets/__init__.py`（`"tiny_office": "indoorloc.datasets.tiny_office:TinyOffice"`），
之后 `load_dataset`、`indoorloc info` 和 `indoorloc benchmark` 都接受这个名称；运行 `python docs/catalog.py` 后，
[datasets.md](datasets.md) 的目录也会列出它。合并之前，请记录官方下载文件的 sha256，在 docstring 中写明许可和推荐的协议，
并在 `tests/datasets/` 中添加一个在真实文件存在时读取它的测试（`@pytest.mark.skipif(not path.is_file(), ...)`）。

## 变换

`__init__` 只保存参数，`fit` 中学到的统计量以 `_` 结尾，`_transform` 作用于单次扫描或批量的最后一维。
基类负责处理 `SampleTable`、扫描视图和 `fit_transform`。

```python
from indoorloc.signals import FillMissing
from indoorloc.signals.transforms import Transform


class ClipRSSI(Transform):
    """Clip readings to the range seen in training (NaN stays NaN)."""

    _requires_fit = True

    def __init__(self, margin_db=0.0):
        self.margin_db = margin_db

    def fit(self, X, y=None):
        super().fit(X)                      # records n_features_in_
        x = X.X if isinstance(X, SampleTable) else np.asarray(X)
        self.lo_, self.hi_ = np.nanmin(x) - self.margin_db, np.nanmax(x) + self.margin_db
        return self

    def _transform(self, x):
        return np.clip(x, self.lo_, self.hi_)


clip = ClipRSSI(margin_db=1.0).fit(table)
print(clip.lo_, clip.hi_, clip.transform(np.array([-99.0, -20.0, np.nan])))
# -81.0 -44.0 [-81. -44.  nan]
```

## 方法

方法实现 `_fit(X, pos, floor, building)` 和 `_localize(X) -> Prediction`。基类提供 `fit`、`predict`、`localize`、`score`、
`evaluate`、`save`、`clone` 和参数管理，并负责校验输入。`_allow_nan = True` 声明 `X` 可以包含缺失读数。
方法无法定位的样本得到 NaN 位置，`evaluate` 把它计入 `n_failed`。

```python
from indoorloc.core import Prediction
from indoorloc.methods import BaseLocalizer, create_model, register_model


class StrongestAPLocalizer(BaseLocalizer):
    """Place a scan at the mean training position of the scans with the same strongest AP.

    A deliberately simple proximity baseline: the position follows from which transmitter is
    heard best, not from a distance model. Readings below ``min_dbm`` count as not heard; a scan
    whose strongest transmitter was never the strongest in training is not placed (NaN).

    References
    ----------
    N. Bulusu, J. Heidemann, D. Estrin, "GPS-less low-cost outdoor localization for very small
    devices", IEEE Personal Communications 7(5), 2000. DOI 10.1109/98.878533. (Proximity
    localization from the transmitters that are heard; the strongest-transmitter rule is a
    simplification made for this example.)
    """

    _allow_nan = True

    def __init__(self, min_dbm=-100.0):
        self.min_dbm = min_dbm

    def _strongest(self, X):
        x = np.where(np.isnan(X) | (X < self.min_dbm), -np.inf, X)   # NaN or too weak: not heard
        return np.argmax(x, axis=1), np.isfinite(x).any(axis=1)     # argmax: first index on ties

    def _fit(self, X, pos, floor, building):
        top, heard = self._strongest(X)
        self.cell_pos_ = np.full((X.shape[1], pos.shape[1]), np.nan)
        for j in np.unique(top[heard]):
            self.cell_pos_[j] = pos[heard & (top == j)].mean(axis=0)

    def _localize(self, X):
        top, heard = self._strongest(X)
        return Prediction(np.where(heard[:, None], self.cell_pos_[top], np.nan))


register_model("strongest_ap", StrongestAPLocalizer)
train, test = iloc.load_dataset("synthetic_office")
print(create_model("strongest_ap").fit(train).evaluate(test))
# mean 4.3641  median 4.2789  P90 6.9086  floor n/a  building n/a  (n=200)
```

这个示例只预测位置。随 IndoorLoc 发布的方法在训练数据带有楼层和建筑标签时还要预测它们（例如由匹配到的训练扫描投票），
并且需要一个针对已知结果（闭式解、教科书例子或已发表的小例子）的测试，而不只是冒烟测试；类的 docstring 要用两三句话说明方法、
列出参数，并包含 `References` 一节（[CONTRACTS.md §8](../architecture/CONTRACTS.md#8-documentation-and-citations)）。
[methods.md](methods.md) 的目录取的就是这个 docstring 的第一行和第一篇参考文献。

## 协议

命名协议把一张表变成带标签的行下标折。注册之后，`get_protocol` 和 `indoorloc benchmark --protocol` 都可以使用它：

```python
from indoorloc.evaluation import Fold, Protocol, get_protocol, leave_one_group_out, register_protocol


def _by_room(table, seed):
    rooms = table.groups["room"]
    return [Fold(f"room={r}", tr, te) for r, (tr, te) in zip(np.unique(rooms), leave_one_group_out(rooms))]


register_protocol("leave-one-room-out", Protocol("leave-one-room-out", "each room tests once", _by_room,
                                                 ("groups['room']",)))
folds = get_protocol("leave-one-room-out").folds(train)
print(len(folds), folds[0].name, len(folds[0].test))
# 21 room=0 40
```

## 已发表结果

已发表的结果写入 `indoorloc/evaluation/literature/<dataset>.json`，包括方法名、按指标给出的数值（每个指标在文件层面定义，并注明单位）、
来源（作者、标题、期刊或会议、年份、DOI）、数字所在的表格或章节、协议，以及 `check` 状态（只有与论文全文核对后才能标为 `verified`）。
`indoorloc literature <dataset>` 和 `literature.compare` 把这些数字与复现结果并列显示，但从不混在一起。

## 提交 pull request 之前

```bash
python -m pytest -q tests                  # 每个测试文件在仿真数据上几秒内跑完
lint-imports                               # pyproject.toml 中的三条导入契约
python docs/catalog.py                     # 重新生成本指南的目录表格
python docs/catalog.py --snippets          # 运行本指南中的代码块
```
