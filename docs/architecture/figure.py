"""Draw the five-layer architecture figure (assets/architecture.png and architecture_zh.png).

    python docs/architecture/figure.py

Every chip names something that ships in the package (filled), ships with a stated limitation
(outlined in colour), or is planned (grey outline). Keep the table below in sync with the code:
``tests/test_architecture_figure.py`` checks that every filled chip's module exists.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"L1": "#2563eb", "L2": "#0d9488", "L3": "#7c3aed", "L4": "#d97706", "L5": "#db2777"}

# (english, chinese, status, module that implements it)  status: "on" shipped, "part" shipped with limits, "plan" planned
LAYERS = [
    ("L5", ("Applications", "应用"), ("tracking · PDR · fusion · navigation", "跟踪 · 航位推算 · 融合 · 导航"),
     ("score your own trajectories", "评测自有轨迹"), [
        ("Kalman / RTS / EKF", "卡尔曼 / RTS / EKF", "on", "indoorloc.apps.tracking"),
        ("Particle filter + floor maps", "粒子滤波 + 楼层地图", "on", "indoorloc.apps.particle"),
        ("PDR", "行人航位推算", "on", "indoorloc.apps.pdr"),
        ("PDR + WiFi fusion", "PDR + WiFi 融合", "on", "indoorloc.apps.fusion"),
        ("Streaming inference", "流式推理", "on", "indoorloc.apps.streaming"),
        ("A* navigation", "A* 导航", "on", "indoorloc.apps.navigation"),
    ]),
    ("L4", ("Evaluation", "评测"), ("metrics · protocols · bounds · literature", "指标 · 协议 · 界 · 文献"),
     ("score your own predictions", "评测自有预测"), [
        ("Errors · CDF · bootstrap CI", "误差 · CDF · 置信区间", "on", "indoorloc.evaluation.functional"),
        ("IPIN / EvAAL scores", "IPIN / EvAAL 评分", "on", "indoorloc.evaluation.scoring"),
        ("CRLB · GDOP", "CRLB · GDOP", "on", "indoorloc.evaluation.bounds"),
        ("Cross-device / cross-time protocols", "跨设备 / 跨时间协议", "on", "indoorloc.evaluation.protocols"),
        ("Published results with provenance", "带出处的文献结果", "on", "indoorloc.evaluation.literature"),
        ("indoorloc benchmark CLI", "indoorloc benchmark 命令行", "on", "indoorloc.cli"),
    ]),
    ("L3", ("Methods", "方法"), ("fingerprinting · model-based · deep · transfer", "指纹 · 模型驱动 · 深度 · 迁移"),
     ("train on your own data", "训练自有数据"), [
        ("kNN / WKNN · Horus · GP radio map", "kNN / WKNN · Horus · 高斯过程", "on", "indoorloc.methods.probabilistic"),
        ("SVM / RF / GBDT", "SVM / RF / GBDT", "on", "indoorloc.methods.sklearn_wrap"),
        ("Ensemble · Stacking · Hierarchical", "集成 · 堆叠 · 分层", "on", "indoorloc.methods.ensemble"),
        ("Trilateration · Chan TDoA · MUSIC AoA", "三边定位 · Chan TDoA · MUSIC AoA", "on", "indoorloc.methods.geometric"),
        ("Path loss · centroid · VLC", "路径损耗 · 质心 · 可见光", "on", "indoorloc.methods.pathloss"),
        ("MLP / CNN1D / timm", "MLP / CNN1D / timm", "on", "indoorloc.methods.deep"),
        ("CORAL / TCA", "CORAL / TCA", "on", "indoorloc.methods.transfer"),
        ("Magnetic DTW", "地磁 DTW", "on", "indoorloc.methods.magnetic"),
        ("Self-supervised · channel charting", "自监督 · 信道图谱", "plan", None),
    ]),
    ("L2", ("Signals", "信号"), ("representations · calibration · CSI · IMU", "表示 · 校准 · CSI · IMU"),
     ("transform a single signal", "处理单条信号"), [
        ("RSSI representations", "RSSI 表示", "on", "indoorloc.signals.transforms"),
        ("AP selection · filtering", "AP 选择 · 过滤", "on", "indoorloc.signals.transforms"),
        ("Device calibration", "设备校准", "on", "indoorloc.signals.calibration"),
        ("Augmentation", "数据增强", "on", "indoorloc.signals.augment"),
        ("CSI phase sanitizing · Hampel", "CSI 相位净化 · Hampel", "on", "indoorloc.signals.csi"),
        ("Ranging · IMU · magnetic · VLC", "测距 · IMU · 地磁 · 可见光", "on", "indoorloc.signals.magnetic"),
    ]),
    ("L1", ("Data", "数据"), ("13 measured · 2 simulators", "13 个实测 · 2 个仿真"),
     ("export numpy / torch", "导出 numpy / torch"), [
        ("6 WiFi", "6 个 WiFi", "on", "indoorloc.datasets.ujiindoorloc"),
        ("3 BLE", "3 个 BLE", "on", "indoorloc.datasets.ble_indoor"),
        ("3 CSI", "3 个 CSI", "on", "indoorloc.datasets.haloc"),
        ("ILC 2020 IMU + WiFi + maps", "ILC 2020 IMU + WiFi + 地图", "on", "indoorloc.datasets.ilc2020"),
        ("SHA-256 verified downloads", "SHA-256 校验下载", "on", "indoorloc.datasets._base"),
        ("Synthetic office (3GPP InH, COST 231)", "合成办公室（3GPP InH、COST 231）", "on",
         "indoorloc.datasets.simulated.office"),
        ("DeepMIMO v4 adapter", "DeepMIMO v4 适配", "part", "indoorloc.datasets.simulated.deepmimo"),
        ("Sionna RT · digital twins", "Sionna RT · 数字孪生", "plan", None),
    ]),
]


def _two_lines(text: str, limit: int = 30) -> str:
    """Break a long ``a · b · c`` subtitle at the separator nearest its middle."""
    parts = text.split(" · ")
    if len(text) <= limit or len(parts) < 2:
        return text
    cut = min(range(1, len(parts)), key=lambda i: abs(len(" · ".join(parts[:i])) - len(text) / 2))
    return " · ".join(parts[:cut]) + "\n" + " · ".join(parts[cut:])


def draw(lang: str, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch

    zh = int(lang == "zh")
    # Static CJK faces (a variable font would render at its default, thinnest weight).
    cjk = []
    for font, name in (("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK JP"),
                       ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "Noto Sans CJK JP"),
                       ("C:/Windows/Fonts/msyh.ttc", "Microsoft YaHei"), ("C:/Windows/Fonts/msyhbd.ttc", "Microsoft YaHei")):
        if Path(font).is_file():
            font_manager.fontManager.addfont(font)
            cjk.append(name)
    family = [*dict.fromkeys(cjk), "DejaVu Sans"] if zh else ["DejaVu Sans"]
    plt.rcParams.update({"font.family": family, "font.size": 10})

    W, gap, left_w, body_x, body_end, chip_h, font = 10.0, 0.14, 2.05, 2.45, 7.45, 0.3, 7.4
    fig = plt.figure(figsize=(W, 8), dpi=200)
    measure = fig.canvas.get_renderer()

    def text_width(label):
        t = fig.text(0, 0, label, fontsize=font)
        width = t.get_window_extent(renderer=measure).width / fig.dpi
        t.remove()
        return width + 0.24

    # Layout pass: place the chips of every layer in wrapped lines, then size each row to fit.
    rows = []
    for lid, name, sub, exit_text, chips in LAYERS:
        lines, x = [[]], body_x + 0.15
        for chip in chips:
            width = text_width(chip[1] if zh else chip[0])
            if x + width > body_end and lines[-1]:
                lines.append([])
                x = body_x + 0.15
            lines[-1].append((chip, x, width))
            x += width + 0.09
        rows.append((lid, name, sub, exit_text, lines, max(1.02, 0.26 + len(lines) * (chip_h + 0.08) + 0.1)))
    H = sum(r[-1] + gap for r in rows) + 0.95
    fig.set_size_inches(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(W - 0.12, H - 0.16, "每一层都可单独使用" if zh else "Every layer stands alone", ha="right", va="top",
            fontsize=10.5, fontweight="bold", color="#111827")

    top = H - 0.5
    for lid, name, sub, exit_text, lines, row_h in rows:
        color = COLORS[lid]
        y0 = top - row_h
        top = y0 - gap
        ax.add_patch(FancyBboxPatch((0.25, y0), body_end - 0.25 + 0.15, row_h,
                                    boxstyle="round,pad=0,rounding_size=0.12", fc="#f9fafb", ec="#d1d5db", lw=0.8))
        ax.add_patch(FancyBboxPatch((0.25, y0), left_w, row_h, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=color, ec=color, lw=0))
        ax.text(0.4, y0 + row_h - 0.14, lid, color="white", fontsize=9.5, fontweight="bold", va="top")
        ax.text(0.4, y0 + row_h - 0.36, name[zh], color="white", fontsize=14, fontweight="bold", va="top")
        ax.text(0.4, y0 + 0.12, _two_lines(sub[zh]), color="white", fontsize=6.6, va="bottom", alpha=0.95,
                linespacing=1.3)
        y = y0 + row_h - 0.16
        for line in lines:
            for (en, cn, status, _), x, width in line:
                label = cn if zh else en
                face, edge, ink = {"on": (color, color, "white"), "part": ("#ffffff", color, color),
                                   "plan": ("#ffffff", "#d1d5db", "#6b7280")}[status]
                ax.add_patch(FancyBboxPatch((x, y - chip_h), width, chip_h, boxstyle="round,pad=0,rounding_size=0.08",
                                            fc=face, ec=edge, lw=0.8))
                ax.text(x + width / 2, y - chip_h / 2, label, ha="center", va="center", fontsize=font, color=ink)
            y -= chip_h + 0.08
        ax.annotate("", xy=(body_end + 0.5, y0 + row_h / 2), xytext=(body_end + 0.18, y0 + row_h / 2),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.2})
        ax.text(body_end + 0.56, y0 + row_h / 2, exit_text[zh], va="center", fontsize=8.6, color="#111827")

    legend = [("on", "可用" if zh else "available"), ("part", "可用，有已声明的限制" if zh else "available, stated limits"),
              ("plan", "规划中" if zh else "planned")]
    x = 0.3
    for status, text in legend:  # noqa: B007
        face, edge = {"on": (COLORS["L1"], COLORS["L1"]), "part": ("#ffffff", COLORS["L1"]),
                      "plan": ("#ffffff", "#d1d5db")}[status]
        ax.add_patch(FancyBboxPatch((x, 0.16), 0.32, 0.17, boxstyle="round,pad=0,rounding_size=0.05", fc=face, ec=edge,
                                    lw=0.8))
        label = ax.text(x + 0.42, 0.245, text, va="center", fontsize=8.2, color="#6b7280")
        x += 0.42 + label.get_window_extent(renderer=measure).width / fig.dpi + 0.4
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    for lang, name in (("en", "architecture.png"), ("zh", "architecture_zh.png")):
        print(draw(lang, args.out / name))


if __name__ == "__main__":
    main()
