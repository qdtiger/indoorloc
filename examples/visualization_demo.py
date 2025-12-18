#!/usr/bin/env python
"""
Indoor Localization Visualization Demo - Ghost Trace

This script demonstrates the "Ghost Trace" visualization concept for indoor
localization results. It shows:
- Ground Truth positions (green solid circles)
- Prediction positions (blue hollow circles)
- Error vectors (red lines connecting GT to Pred)
- CDF error distribution curve

Usage:
    python examples/visualization_demo.py

Output:
    - ghost_trace_visualization.html (interactive Plotly)
    - ghost_trace_visualization.png (static Matplotlib fallback)
"""

import numpy as np
from typing import List, Tuple, Optional

# Check available visualization backends
PLOTLY_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


def create_ghost_trace_plotly(
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    errors: np.ndarray,
    floors: np.ndarray,
    output_file: str = "ghost_trace_visualization.html"
) -> "go.Figure":
    """Create interactive Ghost Trace visualization with Plotly."""
    unique_floors = np.unique(floors)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.75, 0.25],
        subplot_titles=("Ghost Trace (GT vs Prediction)", "Error CDF"),
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    for floor_idx in unique_floors:
        mask = floors == floor_idx
        floor_gt = gt_coords[mask]
        floor_pred = pred_coords[mask]
        floor_err = errors[mask]

        # Error vectors (red lines)
        x_lines, y_lines = [], []
        for i in range(len(floor_gt)):
            x_lines.extend([floor_gt[i, 0], floor_pred[i, 0], None])
            y_lines.extend([floor_gt[i, 1], floor_pred[i, 1], None])

        fig.add_trace(
            go.Scatter(
                x=x_lines, y=y_lines,
                mode='lines',
                line=dict(color='rgba(231, 76, 60, 0.4)', width=1),
                name=f'Error (F{floor_idx})',
                legendgroup=f'Floor {floor_idx}',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Ground Truth points (green)
        fig.add_trace(
            go.Scatter(
                x=floor_gt[:, 0], y=floor_gt[:, 1],
                mode='markers',
                marker=dict(color='#2ecc71', size=6, symbol='circle'),
                name=f'GT (F{floor_idx})',
                legendgroup=f'Floor {floor_idx}',
                hovertemplate="<b>GT</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )

        # Prediction points (blue, colored by error)
        fig.add_trace(
            go.Scatter(
                x=floor_pred[:, 0], y=floor_pred[:, 1],
                mode='markers',
                marker=dict(
                    color=floor_err,
                    colorscale='Bluered',
                    cmin=0,
                    cmax=np.percentile(errors, 95),
                    size=7,
                    symbol='circle-open',
                    line=dict(width=2)
                ),
                name=f'Pred (F{floor_idx})',
                legendgroup=f'Floor {floor_idx}',
                hovertemplate=(
                    "<b>Pred</b><br>"
                    "X: %{x:.2f}<br>Y: %{y:.2f}<br>"
                    "Error: %{marker.color:.2f}m<extra></extra>"
                )
            ),
            row=1, col=1
        )

    # CDF curve
    sorted_errors = np.sort(errors)
    cdf_y = np.arange(1, len(errors) + 1) / len(errors)

    fig.add_trace(
        go.Scatter(
            x=sorted_errors, y=cdf_y,
            mode='lines',
            line=dict(color='#2c3e50', width=2),
            name='Error CDF',
            hovertemplate="Error: %{x:.2f}m<br>CDF: %{y:.2%}<extra></extra>"
        ),
        row=1, col=2
    )

    # Add percentile markers
    for pct in [50, 75, 90]:
        val = np.percentile(errors, pct)
        fig.add_trace(
            go.Scatter(
                x=[val], y=[pct / 100],
                mode='markers+text',
                marker=dict(color='#e74c3c', size=8),
                text=[f'{pct}%: {val:.1f}m'],
                textposition='top right',
                showlegend=False
            ),
            row=1, col=2
        )

    fig.update_layout(
        template="plotly_white",
        title_text=f"Indoor Localization - Ghost Trace Analysis (n={len(errors)})",
        height=700,
        hovermode="closest",
        yaxis=dict(scaleanchor="x", scaleratio=1, title="Y (m)"),
        xaxis=dict(title="X (m)"),
        xaxis2=dict(title="Error (m)", range=[0, np.percentile(errors, 98)]),
        yaxis2=dict(title="CDF", range=[0, 1.05]),
        legend=dict(groupclick="toggleitem", traceorder="grouped")
    )

    fig.write_html(output_file)
    print(f"[Plotly] Saved: {output_file}")
    return fig


def create_ghost_trace_matplotlib(
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    errors: np.ndarray,
    floors: np.ndarray,
    output_file: str = "ghost_trace_visualization.png"
) -> "plt.Figure":
    """Create static Ghost Trace visualization with Matplotlib."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_map, ax_cdf = axes

    unique_floors = np.unique(floors)
    colors_gt = '#2ecc71'
    colors_pred = '#3498db'
    colors_err = '#e74c3c'

    # Map plot
    for floor_idx in unique_floors:
        mask = floors == floor_idx
        floor_gt = gt_coords[mask]
        floor_pred = pred_coords[mask]

        # Error vectors
        for i in range(len(floor_gt)):
            ax_map.plot(
                [floor_gt[i, 0], floor_pred[i, 0]],
                [floor_gt[i, 1], floor_pred[i, 1]],
                color=colors_err, alpha=0.3, linewidth=0.8
            )

        # GT points
        ax_map.scatter(
            floor_gt[:, 0], floor_gt[:, 1],
            c=colors_gt, s=20, label=f'GT (F{floor_idx})',
            marker='o', edgecolors='none', alpha=0.8
        )

        # Pred points
        ax_map.scatter(
            floor_pred[:, 0], floor_pred[:, 1],
            c=colors_pred, s=25, label=f'Pred (F{floor_idx})',
            marker='x', alpha=0.8
        )

    ax_map.set_xlabel('X (m)')
    ax_map.set_ylabel('Y (m)')
    ax_map.set_title('Ghost Trace (GT vs Prediction)')
    ax_map.set_aspect('equal')
    ax_map.legend(loc='upper right', fontsize=8)
    ax_map.grid(True, alpha=0.3)

    # CDF plot
    sorted_errors = np.sort(errors)
    cdf_y = np.arange(1, len(errors) + 1) / len(errors)
    ax_cdf.plot(sorted_errors, cdf_y, color='#2c3e50', linewidth=2)

    # Percentile markers
    for pct, color in [(50, '#27ae60'), (75, '#f39c12'), (90, '#e74c3c')]:
        val = np.percentile(errors, pct)
        ax_cdf.axhline(y=pct / 100, color=color, linestyle='--', alpha=0.5)
        ax_cdf.axvline(x=val, color=color, linestyle='--', alpha=0.5)
        ax_cdf.scatter([val], [pct / 100], color=color, s=50, zorder=5)
        ax_cdf.annotate(
            f'{pct}%: {val:.1f}m',
            xy=(val, pct / 100),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )

    ax_cdf.set_xlabel('Error (m)')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.set_title('Error Distribution')
    ax_cdf.set_xlim(0, np.percentile(errors, 98))
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.grid(True, alpha=0.3)

    # Stats annotation
    stats_text = (
        f"Mean: {np.mean(errors):.2f}m\n"
        f"Median: {np.median(errors):.2f}m\n"
        f"Std: {np.std(errors):.2f}m"
    )
    ax_cdf.text(
        0.95, 0.05, stats_text,
        transform=ax_cdf.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[Matplotlib] Saved: {output_file}")
    return fig


def create_ghost_trace_visualization(
    gt_coords: List[Tuple[float, float]],
    pred_coords: List[Tuple[float, float]],
    errors: List[float],
    floors: List[int],
    output_prefix: str = "ghost_trace_visualization",
    backend: str = "auto"
):
    """
    Create Ghost Trace visualization for indoor localization results.

    Args:
        gt_coords: List of (x, y) ground truth coordinates
        pred_coords: List of (x, y) prediction coordinates
        errors: List of error distances (meters)
        floors: List of floor indices
        output_prefix: Output filename prefix
        backend: "plotly", "matplotlib", or "auto"

    Returns:
        Figure object (plotly or matplotlib)
    """
    gt = np.array(gt_coords)
    pred = np.array(pred_coords)
    err = np.array(errors)
    fl = np.array(floors)

    if backend == "auto":
        backend = "plotly" if PLOTLY_AVAILABLE else "matplotlib"

    if backend == "plotly":
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available. Install: pip install plotly")
        return create_ghost_trace_plotly(gt, pred, err, fl, f"{output_prefix}.html")
    else:
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib not available. Install: pip install matplotlib")
        return create_ghost_trace_matplotlib(gt, pred, err, fl, f"{output_prefix}.png")


def run_demo_with_indoorloc():
    """Run demo using actual IndoorLoc framework with synthetic data."""
    import indoorloc as iloc

    print("=" * 60)
    print("Ghost Trace Visualization Demo")
    print("=" * 60)

    np.random.seed(42)
    num_train, num_test = 500, 200
    num_aps = 520

    print("\n1. Generating synthetic indoor data...")

    def generate_signal():
        rssi = np.random.uniform(-100, -30, num_aps).astype(np.float32)
        not_detected = np.random.choice(num_aps, size=int(num_aps * 0.7), replace=False)
        rssi[not_detected] = 100
        return iloc.WiFiSignal(rssi_values=rssi)

    def generate_location(floor_id=None):
        if floor_id is None:
            floor_id = np.random.randint(0, 3)
        return iloc.Location.from_coordinates(
            x=np.random.uniform(0, 200),
            y=np.random.uniform(0, 150),
            floor=floor_id,
            building_id="0"
        )

    train_signals = [generate_signal() for _ in range(num_train)]
    train_locations = [generate_location() for _ in range(num_train)]
    test_signals = [generate_signal() for _ in range(num_test)]
    test_locations = [generate_location() for _ in range(num_test)]

    print(f"   Train: {num_train}, Test: {num_test}")

    print("\n2. Training KNN model...")
    model = iloc.create_model('KNNLocalizer', k=5, weights='distance')
    model.fit(train_signals, train_locations)
    print(f"   Model trained: {model.is_trained}")

    print("\n3. Making predictions...")
    results = model.predict_batch(test_signals)

    gt_coords = [(loc.x, loc.y) for loc in test_locations]
    pred_coords = [(r.x, r.y) for r in results]
    errors = [r.location.distance_to(gt) for r, gt in zip(results, test_locations)]
    floors = [loc.floor for loc in test_locations]

    print(f"   Mean Error: {np.mean(errors):.2f}m")
    print(f"   Median Error: {np.median(errors):.2f}m")

    print("\n4. Creating visualization...")

    backend = "plotly" if PLOTLY_AVAILABLE else "matplotlib"
    print(f"   Backend: {backend}")

    fig = create_ghost_trace_visualization(
        gt_coords=gt_coords,
        pred_coords=pred_coords,
        errors=errors,
        floors=floors,
        output_prefix="ghost_trace_visualization",
        backend=backend
    )

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)

    return fig


def run_demo_standalone():
    """Run demo with mock data (no IndoorLoc dependency)."""
    print("=" * 60)
    print("Ghost Trace Visualization Demo (Standalone)")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 200

    # Simulated walking path
    t = np.linspace(0, 4 * np.pi, n_samples)
    gt_x = 100 + 50 * np.sin(t) + np.cumsum(np.random.randn(n_samples) * 0.5)
    gt_y = 75 + 30 * np.cos(t * 0.5) + np.cumsum(np.random.randn(n_samples) * 0.3)

    # Predictions with varying noise
    noise_scale = 1.5 + np.sin(t) * 1.0
    pred_x = gt_x + np.random.randn(n_samples) * noise_scale
    pred_y = gt_y + np.random.randn(n_samples) * noise_scale

    errors = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
    floors = np.array([0] * 100 + [1] * 100)

    print(f"\nSamples: {n_samples}")
    print(f"Mean Error: {np.mean(errors):.2f}m")
    print(f"Median Error: {np.median(errors):.2f}m")

    backend = "plotly" if PLOTLY_AVAILABLE else "matplotlib"
    print(f"Backend: {backend}")

    fig = create_ghost_trace_visualization(
        gt_coords=list(zip(gt_x, gt_y)),
        pred_coords=list(zip(pred_x, pred_y)),
        errors=errors.tolist(),
        floors=floors.tolist(),
        output_prefix="ghost_trace_visualization",
        backend=backend
    )

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)

    return fig


if __name__ == "__main__":
    print(f"Plotly available: {PLOTLY_AVAILABLE}")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")

    if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
        print("\nError: No visualization backend available!")
        print("Install with: pip install plotly matplotlib")
        exit(1)

    try:
        import indoorloc
        run_demo_with_indoorloc()
    except ImportError:
        print("\nIndoorLoc not found, running standalone demo...")
        run_demo_standalone()
