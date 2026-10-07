# IndoorLoc 开发规划与问题工单（Development Plan）

> **本文档是本库的施工总图纸**：战略定位、架构契约（代码该怎么写）、里程碑与决策门、以及全部已知问题的分级工单。
> 执行代码任务的智能体（或人）应以本文档为准。
> **注意**：问题清单基于 2026-08-26 的代码审计（commit `9e676d8` 前后），动手修任何一条之前，先读对应文件确认现状未漂移。

---

## 1. 战略定位（一页纸）

**定位声明**：IndoorLoc 是室内无线定位领域的"可复现基准平台"——用 OpenMMLab 的方式统一 **多数据集注册表 + 算法/模型 zoo + 配置驱动评测** 三层。截至 2026-08，该组合缺口无人占据（已对抗核验），但正在关闭。

**竞争格局**（2026-08 核验）：

| 竞品 | 有什么 | 缺什么 | 与我们的关系 |
|---|---|---|---|
| COMFORD（2026-08-07 发布，Torres-Sospedra/IPIN 核心圈，pip 包） | 11 个 WiFi 指纹数据集统一加载 + ISO 18305/IPIN 标准指标 | **刻意不做算法 zoo** | 做适配器把它变成上游依赖；预印本上线当周对外接洽 |
| CSI-Bench（NeurIPS'25 D&B） | 统一训练评测 + 7 模型 zoo，含（粗粒度）定位任务 | 单一自有语料，WiFi 感知框架 | 差异化：我们是多公开数据集、坐标级定位 |
| SenseFi（~629 星） | mmlab 式 WiFi 感知基准 | 明确无定位任务 | 模板证明，不构成竞争 |
| OpenHPS | TypeScript 模块化定位工程框架 | 无数据集/zoo/基准 | 不同赛道 |
| CFM-Bench（2026-07，规范文稿） | 信道基础模型基准规范，含定位 | 尚无公开仓库 | 关注其落地进度 |

**领域证据**（论文动机数字，全部有出处）：IPIN 2022-23 论文仅 **8.3%** 开源代码；119 个公开数据集仅 **6.72%** 附标准切分（ORDIP, Internet of Things 2025）；微软竞赛同屋 22 系统误差 **0.72–10.22 m**；多数据集研究结论"无方法跨数据集稳赢"。

**铁律：日期固定，范围伸缩。** 仅有的两个不可挽回错误：
1. arXiv 时间戳拖过 ~120 天（先发权只能认领一次，COMFORD 随时可能自补算法层）；
2. 发表一个复现不了的数字（基准库的死刑）。

---

## 2. 架构契约（代码该怎么写）

### 2.0 架构选型总则（简约条款，优先级高于本节其余内容）

**不迷信 mmlab。** mmlab 的复杂度按"30+ 仓库 × 海量贡献者"的规模摊销，本项目（约 2 万行、单维护者、8 个算法）承担不起同样的维护成本。规则：**一个机制只有在当前规模下付得起租金才引入；"以后会用到"的预留一律不建**，触发点到了再建（触发点写明）。仓库里已有的空注册表与空壳包正是违反本条的产物（见 P2-3）。

已定取舍：

| 机制 | 决定 | 理由 / 触发点 |
|---|---|---|
| YAML 配置 + `_base_` 继承 | ✅ 保留 | 基准矩阵"一格 = 一 config"的根基；base 链不超过一层，不引入 Python 式 config |
| Registry | ✅ 收缩保留 | 只留实际有注册者的表；空表随 P2-3 删除，禁止新建空表预留 |
| 薄壳算法 + 共享组件 | ✅ 保留 | 复现契约的载体（见 2.1） |
| Runner / Hook 训练引擎 | ❌ 不建 | 一个朴素的 `train_model()` 函数足够；触发点 = 出现第二种真正不同的训练范式（如自监督真正落地）再抽象 |
| DataElement 式对象全家桶 | ❌ 不建 | Signal/Location 已够用；边界走裸数组（§2.4） |
| 分布式训练抽象 | ❌ 不建 | 指纹定位模型单卡分钟级可训，`device=` 参数足够 |
| 插件 / entry-point 体系 | ❌ 不建 | `register_module()` + 文档示例已覆盖第三方扩展 |

复杂度的度量标准：**新用户从 `pip install` 到用自己的数据跑通第一个模型，需要读多少行库内代码**。每个新增抽象都必须回答这个数字是变大还是变小。

### 2.1 mmlab 解剖学（写算法的姿势）

实测 mmdetection：每算法专属代码仅占全库 ~6%（薄壳编排类，平均 4.7KB），其余 ~85% 是共享组件、配置树、数据/评测/引擎基础设施。**照此办理**：

- 一个"算法" = **薄注册壳**（`@LOCALIZERS.register_module()`）+ 至多 2-3 个真正新的组件（head/loss 等）+ 复用 `_base_` 的配置文件 + **复现三件套（config + 训练日志 + checkpoint/拟合产物 + 基准表行）**。
- 训练范式级算法（自监督/元学习，未来实现时）也只是 MODELS/LOCALIZERS 注册表里的又一个包装类（参照 mmpretrain `BaseSelfSupervisor`），**不改训练引擎**。
- 缺三件套的算法不得进公开文档的"已实现"清单。

### 2.2 复现契约（机器可查的 Definition of Done）

**Verified Dataset（数据集达标定义）**：
1. 自动下载（或文档化的手动步骤）+ **固定 SHA256/MD5 校验和**；
2. **规范 train/val/test 切分以版本化产物形式提交**（切分索引文件，或确定性种子代码）；
3. 通过健康检查全部阶段（download → parse → load → fit → evaluate）；
4. 样本数/维度以断言固化；
5. 文档中的状态行由脚本生成（`scripts/dataset_status.json` → 生成表格），**禁止手编**。

**Implemented Algorithm（算法达标定义）**：
1. 配置文件 + 带种子的一条复现命令；
2. 训练日志 + checkpoint（深度）或拟合产物（浅层）随 GitHub Release 发布；
3. 基准表行（由聚合脚本生成）；
4. 通过 registry 驱动的参数化契约测试（fit/predict/save/load/evaluate）；
5. docstring 引用原论文。

**Provenance 铁则**：本仓库复现的数字与文献报告的数字**永不混列**；两者分列并标注来源；出现分歧时把分歧本身作为发现记录（DomainBed 做法）。

### 2.3 分层 CI（信任的公证人）

- **PR 级**：离线单元测试 + 小型缓存 fixture，秒级，永远绿；
- **周期 cron**：联网重跑数据集健康检查，重新生成 `dataset_status.json`，任何 verified 数据集回归即红；
- **Release 门控**：打 tag 时重跑全部 zoo 配置，复现容差内才放行。
- ruff **钉版本**；生成表格配 regenerate-and-diff 检查，防手编。

### 2.4 分层独立可用契约（à la carte，一等设计原则）

**原则**：每层对外说两种语言——层间组合用本库对象，进出边界用生态标准格式（numpy / DataFrame / torch / CSV）。三条规则：**进得来**（用户自有数据/模型可从任意层进入）、**出得去**（任意层的产物可导出为标准格式离开）、**不绑架**（只用一层不被迫安装或学习其他层）。定位语：每层独立可用，层层打通更强。

目标 API 形态（实现对应工单 P1-11~13）：

```python
# 只用数据层：拿数据走人
X, y = iloc.load_dataset("ujindoorloc", split="v1")[0].to_numpy()   # 或 to_dataframe()/to_torch()
# 只用方法层：自有数据喂本库模型（并兼容 sklearn estimator 协议）
iloc.create_model("wknn", k=5).fit(X_own, y_own).predict(X_new)
# 只用评测层：模型与数据全是用户自己的
iloc.evaluate(y_true, y_pred, dataset="ujindoorloc")   # 指标+文献对照+to_latex
iloc.make_protocol_split(meta, protocol="cross-device") # 协议切分器用于自有数据
# 只用信号层：变换管线直接吃裸数组
iloc.transforms.CSISanitize()(raw_csi)
```

配套约束：标准切分以**纯 CSV 索引**落盘（语言中立，MATLAB/R 用户可直接消费）；pip 依赖分层（基础安装不强制 torch）。**注意**：P1-1 的契约统一手术必须按"数组进出为一等公民"来定接口，一次手术同时满足本节。

### 2.5 等预算调参协议（论文方法论脊柱）

经典与深度方法使用**相同的超参搜索预算**（如每方法每数据集 20 次随机搜索）、相同切分、3 个种子；每次运行落盘 config + seed + log + metrics JSON；聚合脚本渲染基准表（mean/median/P75 误差 + 楼层命中，对齐 IPIN 惯例）。

---

## 3. 90 天里程碑（总计约 30–40 人天）

| # | 里程碑 | 人天 | 验收标准 |
|---|--------|------|----------|
| M0 | **诚实重置**（第 1 周） | 2–3 | pytest 零收集错误；空壳包删除；根目录/docs 杂物清理；README 声明与可查证据一一对应（本次已完成 README 重写与分级标注） |
| M1 | **正确性门**：修 4 个 P0 bug，各配回归测试 | 3–4 | 每个 bug 有一条修前红、修后绿的测试 |
| M2 | **数据集验证潮 1**：6 个 WiFi loader 全绿 + 校验和 + 提交切分；cron CI 上线 | 8–10 | 新 `dataset_status.json` ≥6/12 verified 且 cron 连续两周绿 |
| M3 | **契约统一**：DeepLocalizer 并入 BaseLocalizer + registry 契约测试 | 4–5 | 一个测试文件证明全注册表同一 API；**Day45 止损门**（见 §5） |
| M4 | **基准矩阵 v1**：`indoorloc-bench` 单命令=单表格单元；等预算协议；5–8 算法 × 6 数据集 × 3 种子 | 7–8 | 任一单元格第三方一条命令可复现 |
| M5 | **arXiv 预印本 + v0.2.0**：危机诊断 + 库 + 协议 + 基准表 + 发现 | 8 | Day 90–120 挂出 arXiv ID；tag 与 PyPI 同步 |
| 并行 | **COMFORD 适配器**（静默开发）；预印本上线当周对 Torres-Sospedra 发 interop 提议 | 4–5 | COMFORD 数据流入 `load_dataset()`；指标名对齐 ISO 18305/IPIN |

## 4. 4–12 个月弧线

- **M4–5 月**：COMFORD 适配器公开（数据集数 ~20）；深度行补进矩阵（config+权重+日志 mmlab 契约）。
- **M5–8 月**：评测协议做成智力贡献——规范切分（含 LongTermWiFi 跨时间漂移切分、跨设备切分）以版本化产物发布（可挂 Zenodo DOI）。
- **第 6 个月：选刊决策门**——发现强 → NeurIPS 2027 D&B（约 2027-05 截稿）；否则**默认 IEEE JISPIN**。绝不同内容两投。
- **M6–9 月**：JOSS/SoftwareX 软件论文（第二引用锚点）；月度发版节奏（对抗二次休眠）。
- **M9–12 月**：IPIN 2027 出场（教程/演示/竞赛官方 baseline）；第一个外部贡献 PR 合入。
- **仿真层（DeepMIMO v4 → Sionna RT）**：预印本之后启动，不在 90 天窗口内。

## 5. 决策门

1. **Day 45 — 深度行止损**：DeepLocalizer 契约统一 + 深度模型基准行若未跑通，放弃入表；预印本以浅层矩阵 + "切分协议敏感性"为主发现按期发出；深度 vs 经典问题留给期刊版。
2. **Day 90–120 — 时间戳硬线**：范围可缩（6 个 WiFi 数据集、浅层方法也够），日期不可移。
3. **Month 6 — 选刊门**：JISPIN 默认，D&B 条件触发，二选一。

## 6. 不做清单（各方案与评审一致同意）

- ❌ 实现自监督/元学习/PDR/滤波等新能力（论文要的是忠实复跑的成熟方法）
- ❌ 本窗口接仿真数据（DeepMIMO/Sionna 留在路线图）
- ❌ 与 COMFORD 拼数据规范化（做适配器，不做对抗）
- ❌ Web 排行榜/演示 UI（生成的 markdown 矩阵足够）
- ❌ 追平 900+ ruff 错误的完美主义（钉版 + 分文件豁免即可）
- ❌ 首投 TMC/JMLR MLOSS（慢周期/需用户社区）
- ❌ 新增第 13 个数据集 loader（新数据集走贡献者 PR 路径）
- ❌ 手工再扩充文献基准表（库没跑出来的数字不是库的证据）

---

## 7. 已知问题工单（按优先级分级）

> 审计时间 2026-08-26。修复前先读文件确认；每修一条，若属 P0/P1，必须附回归测试。

### P0 —— 会污染基准数字的正确性 bug（M1 范围，全修）

| # | 位置 | 问题 | 修复建议 |
|---|------|------|----------|
| P0-1 | `indoorloc/tools/train.py:139` | 训练期"validation"实际用 `split='test'` 构建，`results.txt`（`train.py:203`）把测试集成绩当验证集报 | 引入真验证集（从 train 划出或用数据集 val split）；命名如实 |
| P0-2 | `indoorloc/datasets/ble_rssi_uci.py:191-195` | 顺序不打乱的头尾切分（`df.iloc[:num_train]`），数据按位置有序 → train/test 覆盖不同物理位置 | 用实例级 `np.random.RandomState(self.seed)` 打乱（参照 `ibeacon_rssi.py:443-454`）；切分索引落盘 |
| P0-3 | `wlanrssi.py:203`、`csi_fingerprint.py:333`、`hwild.py:279` | `_load_data` 内调全局 `np.random.seed(...)` 污染用户 RNG 状态（wlanrssi 是字面量 42，另两处是 `seed(self.seed)`——同样是全局污染） | 改实例级 `np.random.RandomState(self.seed)`；三处同修 |
| P0-4 | `indoorloc/tools/train.py:64-68` | CLI 覆盖用 `ast.literal_eval`：`--train.fp16 false` 解析为 truthy 字符串 `"false"`，效果与意图相反；`configs/README.md:125` 教的正是小写写法 | 显式布尔解析（接受 true/false/True/False）；修正 configs/README 文档 |

### P1 —— 契约与结构缺陷（M3/M4 范围）

| # | 位置 | 问题 | 修复建议 |
|---|------|------|----------|
| P1-1 | `indoorloc/models/localizers/deep_localizer.py` vs `indoorloc/localizers/base.py` | DeepLocalizer 是 nn.Module（`deep_localizer.py:24`），**不继承 BaseLocalizer**；`predict()` 签名分裂（`:154` tensor→dict vs `base.py:134` signal→LocalizationResult，signal 版叫 `predict_single` `:644`）；`create_model` 返回注解错误（顶层 `indoorloc/__init__.py:187`，注意不是 models 子包的 `__init__`） | 统一契约（⚠️ 两套平行类体系的手术，非改继承一行；受 Day45 止损门约束）；加 registry 驱动参数化契约测试 |
| P1-2 | `deep_localizer.py:642` | `evaluate` 不传 `dataset_name` → 深度模型丢失基准对照功能 | 传入并测试 |
| P1-3 | `indoorloc/evaluation/benchmarks.py:67,72,106,162` | `mean_error=None` 时 `get_sota`/`sorted_by_error`/`beats_count`/`print_table` 崩溃；`default_metric='accuracy'` 从未被读取 | None 安全处理；支持 accuracy 型基准表 |
| P1-4 | `indoorloc/evaluation/metrics.py:246,275,249` | Floor/BuildingAccuracy 把 `None==None` 记为正确（无楼层标签的数据集报 100%）；空预测除零；与 `locations/location.py:88-114` 的匹配语义分裂成两套 | 缺标签样本剔除或计 NaN；统一到一处语义 |
| P1-5 | `indoorloc/datasets/catalog.py:82-84` | 双向子串模糊匹配：1-2 字符输入静默解析到任意首个匹配数据集 | 精确名+别名表命中；未命中报错并给建议列表 |
| P1-6 | `indoorloc/__init__.py:145-149,152-173` | `_TRADITIONAL_MODELS` 缺 `svm`/`rf` 别名（类已实现却报错）；`_is_timm_model` 子串误报（`resnet18x` 通过检查后深处崩溃） | 补别名；timm 判定改精确查询 |
| P1-7 | `indoorloc/localizers/transfer.py:21-30,91,229-232,245-249,267-282,287,329` | skada 是硬依赖但未声明于 pyproject；DANN/MDD 走 `skada.deep` 会失败；楼层/建筑模型异常被裸 `except` 吞掉；置信度硬编码 0.8 | pyproject 声明 extra；DANN/MDD 从宣传移除或真实现；异常至少 log；置信度如实 |
| P1-8 | `csi_fingerprint.py:346`、`hwild.py:294`、`haloc.py:221` vs `signals/csi.py` | CSI 幅度塞进 WiFiSignal，被 RSSI 语义（NOT_DETECTED=100、MIN_RSSI 归一化）错误处理；351 行 CSISignal 是死代码 | CSI loader 改产 CSISignal（或明确文档化降级理由） |
| P1-9 | `indoorloc/datasets/loading.py:84-86` | `split=None` 双重构建数据集：下载检查+解析跑两遍；SOD/LongTermWiFi 的派生维度在 train/test 间一致性无保障 | 单次构建后切分，或缓存共享；维度一致性断言 |
| P1-10 | `indoorloc/datasets/base.py:164-169` + `signals/wifi.py:260-267` | 归一化按**逐样本**统计（`method='standard'` 时每条信号用自身均值/方差归一化），train/test 无共享冻结统计（评测语义错配） | 数据集级统计从 train 计算、冻结后应用于 test；作为协议问题在论文中说明 |
| P1-11 | `indoorloc/datasets/base.py`（现仅有 `to_torch_tensors` :293） | 数据层缺标准格式出口：无 `to_numpy()` / `to_dataframe()`，无整库 CSV 导出；用户拿不走数据 | 补齐三个导出方法 + `iloc.export()`；切分索引以纯 CSV 落盘（§2.4） |
| P1-12 | `indoorloc/localizers/base.py:75-111` | `fit()` 只收 BaseDataset 或 Signal 列表，不收裸 ndarray；不兼容 sklearn estimator 协议，无法进 Pipeline/GridSearchCV | `fit/predict` 增加裸数组路径；通过 `sklearn.utils.estimator_checks` 的核心检查；与 P1-1 同一次手术完成 |
| P1-13 | `indoorloc/evaluation/metrics.py:438`（Evaluator 类方法） | 评测与 Dataset/Location 对象耦合，无函数式入口；自有模型+自有数据的用户用不了本库指标与文献对照 | 顶层 `iloc.evaluate(y_true, y_pred, dataset=None)` 直接吃数组；`make_protocol_split()` 独立可用（§2.4） |
| P1-14 | `pyproject.toml:52-`（optional-dependencies） | 依赖未按层拆分，只用数据/评测层的用户可能被迫装深度学习栈 | 梳理为：基础安装（numpy 系）→ `[torch]` → `[full]`；核实基础依赖不含 torch |

### P2 —— 卫生与信任（M0/M2 范围）

| # | 位置 | 问题 | 修复建议 |
|---|------|------|----------|
| P2-1 | `tests/` | **4 个**测试文件 import 已删除的类（`test_uwb_datasets`、`test_hybrid_datasets`、`test_other_signal_datasets`、`test_final_datasets`）；`test_signals.py` 测旧 API（`tests/test_signals.py:25` 用 `APInfo(rssi=-65)`，但 `signals/wifi.py:19-31` 的 APInfo 无 rssi 字段；`rssi_values` 是构造参数非属性）。审计当日全量 66/126 失败。**注意：`test_new_datasets.py` 的导入全部有效（LongTermWiFi/Tampere/WLANRSSI/TUJI1 均存在），不要删** | 删 4 个失效文件；`test_signals.py` 按现行 API 重写；目标 pytest 全绿 |
| P2-2 | `.github/workflows/ci.yml:31`、`pyproject.toml` | CI 被收窄到 2 个测试文件；ruff 未钉版（0.16.1 下 900+ 报错）；dev extras（black/flake8/isort/mypy）与 CI 实际工具不符；classifier 声称 3.12 但矩阵止于 3.11；`addopts` 强制 pytest-cov | CI 恢复全量；ruff 钉版+分文件豁免；extras 与 CI 对齐 |
| P2-3 | `indoorloc/localizers/pdr/`、`localizers/deep/`、`fusion/`、`engine/`；`registry.py:179-194` | 四个空壳包（仅 `__init__.py`）；FUSIONS/TRAINERS/VISUALIZERS 空注册表 | 删除；带着契约合格的实现再回来 |
| P2-4 | `scripts/dataset_status.json`、`scripts/dataset_health_check.py:182-185`、`scripts/health_check_output.txt` | 状态文件停在 2025-12-16（36 数据集旧注册表，fully_working=1）；健检把"缓存加载成功"计为 fully working；输出文件只有一行 shell 报错 | 对现 12 注册表重跑；区分 auto-download verified 与 cache-loaded；cron 化 |
| P2-5 | `indoorloc/utils/download.py:75-97`、`hwild.py:148-172` | 12 个数据集全部不传 MD5（check_integrity 死代码）；hwild 用 subprocess curl + ">1000 字节即成功"启发式 | 每数据集补校验和；hwild 改走统一下载器 |
| P2-6 | 仓库根/资产 | 未跟踪 TCCN 校样 PDF（勿提交，建议移出）；`Snipaste_*.jpg`(1MB)、`UI.jpg`、`UI1.jpg`(~1.4MB) 已提交且无引用；`assets/` 内 4 个 .docx；`docs/` 7 个陈旧页面迭代（`datasets_v2`–`v6`、`datasets_academic`、`datasets_neoteric`，canonical 是 `datasets.html`/`datasets_zh.html`）；`examples/` 7 个生成 HTML | 清理；大文件如需保留移 Release 附件 |
| P2-7 | `pyproject.toml:98-101`、`indoorloc/__init__.py:133,470` | project.urls 指向错误 org（github.com/indoorloc/indoorloc）与矛盾的 readthedocs；`BLERSSIU_UCI` 导出笔误 | 改指 qdtiger/indoorloc 与 GitHub Pages；改名并留兼容别名 |
| P2-8 | `indoorloc/configs/_base_/`、`tools/` | `datasets/tampere.yaml`、`sodindoorloc.yaml`、`models/cnn1d.yaml` 无任何引用；`gen_config.py` 仅支持 3 数据集、`download_dataset.py` 仅 1 个；`Config.validate()`（`utils/config.py:176-210`）从未被调用；`--opts` 与点号覆盖双语法并存 | 随 M4 的 harness 统一清理；validate 接入 train/test 入口 |
| P2-9 | `indoorloc/evaluation/benchmark_data/` | 孤儿基准表：magneticindoor/csiindoor/csi2taoa/wildv2/wificsid2d 无对应 loader；HALOC 有 loader 无表 | 孤儿表移入 quarantine 目录并标注"文献报告、未复现"；补 HALOC 表 |
| P2-10 | `signals/base.py:151-162`、`metrics.py:24,54-56`、`deep_localizer.py:510,527` | `to_dict` 丢 `sampling_rate`（序列化不可逆）；BaseMetric._results/reset 死状态；`torch.cuda.amp` 弃用 API（torch≥2.4 警告） | 顺手修；amp 迁移到 `torch.amp` |
| P2-11 | `indoorloc/signals/`（imu/uwb/magnetometer/vlc/ultrasound/hybrid，共 1977 行） | 六个信号类无任何生产者（仅测试引用） | 决策项：保留为 L2 API 面（配"未激活"文档标注）或删除；默认保留但不宣传 |

### 修复顺序建议

```
M0: P2-1(删) → P2-3 → P2-6 → P2-2(CI部分)
M1: P0-1 → P0-2 → P0-3 → P0-4
M2: P2-4 → P2-5 → (6个WiFi loader逐个过契约)
M3: P1-1(+P1-12 同一手术) → P1-2 → P1-6 → P1-11 → P1-13
M4: P1-3 → P1-4 → P1-5 → P1-9 → P1-10 → P1-14 → P2-8 → P2-9
随手: P2-7 → P2-10 → P1-7 → P1-8 → P2-11(决策后)
```

---

## 8. 调研结论存档（要点索引）

- **mmlab 解剖**：算法壳 ~6%、基础设施 ~85%（实测 mmdetection@main）；mmdet 技报（arXiv 1906.07155）自述贡献=模块化+公平对比+model zoo，非算法本身。
- **论文配方**（RecBole/LibCity/OpenOOD/DomainBed/SB3/MoleculeNet/TSlib/BasicTS+/Avalanche 九例归纳）：危机诊断 → 统一格式+官方切分 → N×M 忠实复现 → 协议即贡献 → 1-2 个爆点发现 → 版本化+社区证据。爆点候选："等预算下深度指纹是否胜过 WKNN？"（任一结果都是故事）；对冲发现："切分惯例变化导致误差差多少"。
- **选刊图**：arXiv 时间戳（Day 90-120）→ IEEE JISPIN（默认）或 NeurIPS D&B（条件）→ JOSS/SoftwareX（软件锚点）→ JMLR MLOSS（有社区后）。
- **仿真生态**（预印本后启动）：DeepMIMO v4（`pip install deepmimo`，含室内场景+位置标签+InSite/Sionna RT/AODT 三家转换器，一个 loader 打通生态）→ Sionna RT（独立 pip 包、可微分、输出逐径 ToA/AoA/CIR）→ sim2real 协议（文献：仿真 25m→实测 184m，但仿真预训练砍半实测误差；DICHASUS↔Sionna 孪生校准是标准配对）。
- **五层全栈分类树**：见 README「Taxonomy & Roadmap」章节。
