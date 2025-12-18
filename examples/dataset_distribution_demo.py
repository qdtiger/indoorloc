#!/usr/bin/env python
"""
Dataset Spatial Distribution Visualization

Generates TWO separate HTML files:
- dataset_distribution_2d.html: 2D view with hover info (samples, users, etc.)
- dataset_distribution_3d.html: 3D view with all floors stacked

Usage:
    python examples/dataset_distribution_demo.py
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Academic color palette
ACADEMIC_COLORSCALE = [
    [0.0, '#f7fbff'], [0.15, '#deebf7'], [0.3, '#c6dbef'],
    [0.45, '#9ecae1'], [0.6, '#6baed6'], [0.75, '#4292c6'],
    [0.9, '#2171b5'], [1.0, '#084594']
]

FLOOR_COLORS = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']

FONT_FAMILY = "Arial, Helvetica, sans-serif"
TEXT_COLOR = "#2c3e50"
LIGHT_TEXT = "#7f8c8d"
GRID_COLOR = "#e8e8e8"
BORDER_COLOR = "#333333"


def visualize_dataset_2d(
    coords: List[Tuple[float, float]],
    floors: List[int],
    buildings: List[str],
    metadata: Optional[List[Dict]] = None,
    signals: Optional[List[np.ndarray]] = None,
    bin_size: float = 3.0,
    not_detected_value: float = 100.0,
    output_file: str = "dataset_distribution_2d.html"
) -> "go.Figure":
    """Generate pure 2D visualization with floor selection dropdown and hover info."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required: pip install plotly")

    print("Preparing 2D visualization...")

    # Process signals for AP count
    if signals is not None:
        ap_counts = [np.sum(s != not_detected_value) if isinstance(s, np.ndarray) else 0 for s in signals]
    else:
        ap_counts = [0] * len(coords)

    # Process metadata
    if metadata is not None:
        user_ids = [m.get('user_id', 0) for m in metadata]
        space_ids = [m.get('space_id', 0) for m in metadata]
    else:
        user_ids = [0] * len(coords)
        space_ids = [0] * len(coords)

    # Normalize coordinates
    raw_x = [c[0] for c in coords]
    raw_y = [c[1] for c in coords]
    x_min, y_min = min(raw_x), min(raw_y)

    df = pd.DataFrame({
        'x': [x - x_min for x in raw_x],
        'y': [y - y_min for y in raw_y],
        'floor': floors,
        'building': [str(b) for b in buildings],
        'user_id': user_ids,
        'space_id': space_ids,
        'ap_count': ap_counts
    })

    print(f"Coordinate range: X=[0, {df['x'].max():.0f}m], Y=[0, {df['y'].max():.0f}m]")

    fig = go.Figure()
    buttons = []

    groups = df.groupby(['building', 'floor'])
    combinations = sorted(groups.groups.keys(), key=lambda x: (str(x[0]), int(x[1])))

    for i, (bldg, flr) in enumerate(combinations):
        subset = groups.get_group((bldg, flr)).copy()

        # Create bins
        x_min_s = np.floor(subset['x'].min() / bin_size) * bin_size
        x_max_s = np.ceil(subset['x'].max() / bin_size) * bin_size
        y_min_s = np.floor(subset['y'].min() / bin_size) * bin_size
        y_max_s = np.ceil(subset['y'].max() / bin_size) * bin_size

        x_bins = np.arange(x_min_s, x_max_s + bin_size, bin_size)
        y_bins = np.arange(y_min_s, y_max_s + bin_size, bin_size)

        if len(x_bins) < 2 or len(y_bins) < 2:
            continue

        subset['x_bin'] = pd.cut(subset['x'], bins=x_bins, labels=x_bins[:-1], include_lowest=True)
        subset['y_bin'] = pd.cut(subset['y'], bins=y_bins, labels=y_bins[:-1], include_lowest=True)

        # Aggregate statistics per bin
        agg = subset.groupby(['x_bin', 'y_bin'], observed=True).agg({
            'user_id': 'nunique',
            'space_id': 'nunique',
            'ap_count': 'mean',
            'x': 'count'
        }).rename(columns={'x': 'count'}).reset_index()

        agg = agg[agg['count'] > 0]
        if len(agg) == 0:
            continue

        px = agg['x_bin'].astype(float) + bin_size / 2
        py = agg['y_bin'].astype(float) + bin_size / 2

        n_samples = len(subset)
        n_users = subset['user_id'].nunique()
        n_spaces = subset['space_id'].nunique()

        # Add trace
        fig.add_trace(go.Scatter(
            x=px, y=py,
            mode='markers',
            marker=dict(
                symbol='square',
                size=12,
                color=agg['count'],
                colorscale=ACADEMIC_COLORSCALE,
                showscale=(i == 0),
                line=dict(width=0.5, color='white'),
                colorbar=dict(title="Samples", thickness=12, len=0.5, x=1.02)
            ),
            customdata=np.stack((agg['count'], agg['user_id'], agg['space_id'], agg['ap_count']), axis=-1),
            hovertemplate=(
                "<b>Position</b>: (%{x:.1f}, %{y:.1f}) m<br>"
                "<b>Samples</b>: %{customdata[0]:.0f}<br>"
                "<b>Users</b>: %{customdata[1]:.0f}<br>"
                "<b>Spaces</b>: %{customdata[2]:.0f}<br>"
                "<b>Avg APs</b>: %{customdata[3]:.1f}"
                "<extra></extra>"
            ),
            visible=(i == 0),
            showlegend=False,
            name=f"B{bldg} F{flr}"
        ))

        # Dropdown button
        visible_array = [False] * len(combinations)
        visible_array[i] = True

        buttons.append(dict(
            label=f"B{bldg} Floor {flr}",
            method="update",
            args=[
                {"visible": visible_array},
                {"title": f"<b>Building {bldg}, Floor {flr}</b><br>"
                          f"<span style='font-size:12px;color:{LIGHT_TEXT}'>"
                          f"N = {n_samples:,} | Users = {n_users} | Spaces = {n_spaces}</span>"}
            ]
        ))

    # Layout
    first_combo = combinations[0] if combinations else (0, 0)
    first_subset = groups.get_group(first_combo)

    fig.update_layout(
        title=dict(
            text=f"<b>Building {first_combo[0]}, Floor {first_combo[1]}</b><br>"
                 f"<span style='font-size:12px;color:{LIGHT_TEXT}'>"
                 f"N = {len(first_subset):,} samples</span>",
            x=0.5,
            font=dict(family=FONT_FAMILY, size=16, color=TEXT_COLOR)
        ),
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
        width=900,
        height=700,
        margin=dict(t=100, l=70, r=70, b=80),
        xaxis=dict(
            title="X (m)", showgrid=True, gridcolor=GRID_COLOR,
            zeroline=False, showline=True, linecolor=BORDER_COLOR, mirror=True,
            scaleanchor="y", scaleratio=1
        ),
        yaxis=dict(
            title="Y (m)", showgrid=True, gridcolor=GRID_COLOR,
            zeroline=False, showline=True, linecolor=BORDER_COLOR, mirror=True
        ),
        hovermode="closest",
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.0, y=1.12,
            xanchor='left', yanchor='top',
            bgcolor="#ffffff",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=11, family=FONT_FAMILY, color=TEXT_COLOR),
            type="dropdown"
        )]
    )

    print(f"Saved: {output_file}")
    fig.write_html(output_file)
    return fig


def visualize_dataset_3d(
    coords: List[Tuple[float, float]],
    floors: List[int],
    buildings: List[str],
    floor_height: float = 5.0,
    output_file: str = "dataset_distribution_3d.html"
) -> "go.Figure":
    """Generate pure 3D visualization with floor selection dropdown."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required: pip install plotly")

    print("Preparing 3D visualization...")

    # Normalize coordinates
    raw_x = [c[0] for c in coords]
    raw_y = [c[1] for c in coords]
    x_min, y_min = min(raw_x), min(raw_y)

    df = pd.DataFrame({
        'x': [x - x_min for x in raw_x],
        'y': [y - y_min for y in raw_y],
        'floor': floors,
        'building': [str(b) for b in buildings]
    })

    fig = go.Figure()
    buttons = []

    unique_buildings = sorted(df['building'].unique())
    all_traces = []

    # Create traces for each building-floor
    for bldg in unique_buildings:
        bldg_data = df[df['building'] == bldg]
        for flr in sorted(bldg_data['floor'].unique()):
            floor_data = bldg_data[bldg_data['floor'] == flr]
            z_val = flr * floor_height
            color = FLOOR_COLORS[flr % len(FLOOR_COLORS)]

            fig.add_trace(go.Scatter3d(
                x=floor_data['x'],
                y=floor_data['y'],
                z=[z_val] * len(floor_data),
                mode='markers',
                marker=dict(size=2.5, color=color, opacity=0.7),
                name=f"B{bldg} F{flr}",
                visible=True,
                hovertemplate=(
                    f"<b>Building {bldg}, Floor {flr}</b><br>"
                    "X: %{x:.1f}m<br>Y: %{y:.1f}m<br>"
                    f"Samples: {len(floor_data):,}"
                    "<extra></extra>"
                )
            ))
            all_traces.append((bldg, flr, len(floor_data)))

    n_traces = len(all_traces)
    total = len(df)
    n_buildings = df['building'].nunique()
    n_floors = df['floor'].nunique()

    # "All Floors" button
    buttons.append(dict(
        label="All Floors",
        method="update",
        args=[
            {"visible": [True] * n_traces},
            {"title": f"<b>3D View — All Floors</b><br>"
                      f"<span style='font-size:12px;color:{LIGHT_TEXT}'>"
                      f"N = {total:,} | {n_buildings} buildings | {n_floors} floors</span>"}
        ]
    ))

    # Per building-floor buttons
    for i, (bldg, flr, n_samples) in enumerate(all_traces):
        visible = [False] * n_traces
        visible[i] = True
        buttons.append(dict(
            label=f"B{bldg} Floor {flr}",
            method="update",
            args=[
                {"visible": visible},
                {"title": f"<b>Building {bldg}, Floor {flr}</b><br>"
                          f"<span style='font-size:12px;color:{LIGHT_TEXT}'>"
                          f"N = {n_samples:,} samples</span>"}
            ]
        ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>3D View — All Floors</b><br>"
                 f"<span style='font-size:12px;color:{LIGHT_TEXT}'>"
                 f"N = {total:,} | {n_buildings} buildings | {n_floors} floors</span>",
            x=0.5,
            font=dict(family=FONT_FAMILY, size=16, color=TEXT_COLOR)
        ),
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
        width=900,
        height=700,
        margin=dict(t=100, l=0, r=0, b=0),
        scene=dict(
            xaxis=dict(title="X (m)", gridcolor=GRID_COLOR, backgroundcolor='white'),
            yaxis=dict(title="Y (m)", gridcolor=GRID_COLOR, backgroundcolor='white'),
            zaxis=dict(title="Floor", gridcolor=GRID_COLOR, backgroundcolor='white'),
            bgcolor='white',
            aspectmode='data'
        ),
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.0, y=1.12,
            xanchor='left', yanchor='top',
            bgcolor="#ffffff",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=11, family=FONT_FAMILY, color=TEXT_COLOR),
            type="dropdown"
        )],
        legend=dict(
            x=1.0, y=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#ccc',
            borderwidth=1
        )
    )

    print(f"Saved: {output_file}")
    fig.write_html(output_file)
    return fig


def create_combined_page(output_file: str = "dataset_distribution.html"):
    """Create a combined HTML page with 2D/3D toggle using iframes."""
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Dataset Distribution Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; }
        .header {
            background: white;
            padding: 15px 30px;
            border-bottom: 1px solid #ddd;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .header h1 {
            font-size: 18px;
            color: #2c3e50;
            font-weight: 600;
        }
        .toggle-group {
            display: flex;
            gap: 0;
            border: 1px solid #ccc;
            border-radius: 4px;
            overflow: hidden;
        }
        .toggle-btn {
            padding: 8px 20px;
            border: none;
            background: white;
            color: #666;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .toggle-btn:hover { background: #f0f0f0; }
        .toggle-btn.active {
            background: #2171b5;
            color: white;
        }
        .toggle-btn:first-child { border-right: 1px solid #ccc; }
        .frame-container {
            width: 100%;
            height: calc(100vh - 60px);
        }
        iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Dataset Distribution</h1>
        <div class="toggle-group">
            <button class="toggle-btn active" onclick="switchView('2d')">2D View</button>
            <button class="toggle-btn" onclick="switchView('3d')">3D View</button>
        </div>
    </div>
    <div class="frame-container">
        <iframe id="viz-frame" src="dataset_distribution_2d.html"></iframe>
    </div>
    <script>
        function switchView(mode) {
            const frame = document.getElementById('viz-frame');
            const btns = document.querySelectorAll('.toggle-btn');
            btns.forEach(btn => btn.classList.remove('active'));
            if (mode === '2d') {
                frame.src = 'dataset_distribution_2d.html';
                btns[0].classList.add('active');
            } else {
                frame.src = 'dataset_distribution_3d.html';
                btns[1].classList.add('active');
            }
        }
    </script>
</body>
</html>'''

    with open(output_file, 'w') as f:
        f.write(html_content)
    print(f"Saved: {output_file}")


def run_demo():
    """Run demo with UJIndoorLoc dataset."""
    import indoorloc as iloc

    print("=" * 50)
    print("Dataset Distribution Visualization")
    print("=" * 50)

    print("\nLoading UJIndoorLoc dataset...")
    try:
        dataset = iloc.UJIndoorLoc(split='train', download=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Loaded {len(dataset)} samples")

    coords, floors, buildings, metadata, signals = [], [], [], [], []

    print("Extracting data...")
    for i in range(len(dataset)):
        signal, location = dataset[i]
        meta = dataset._metadata[i] if hasattr(dataset, '_metadata') else {}

        coords.append((location.x, location.y))
        floors.append(location.floor)
        buildings.append(location.building_id)
        metadata.append(meta)
        signals.append(signal.data if hasattr(signal, 'data') else None)

    print(f"Buildings: {sorted(set(buildings))}")
    print(f"Floors: {sorted(set(floors))}")

    # Generate 2D visualization
    print("\n--- Generating 2D visualization ---")
    visualize_dataset_2d(
        coords=coords,
        floors=floors,
        buildings=buildings,
        metadata=metadata,
        signals=signals,
        bin_size=3.0,
        output_file="dataset_distribution_2d.html"
    )

    # Generate 3D visualization
    print("\n--- Generating 3D visualization ---")
    visualize_dataset_3d(
        coords=coords,
        floors=floors,
        buildings=buildings,
        floor_height=5.0,
        output_file="dataset_distribution_3d.html"
    )

    # Generate combined page
    print("\n--- Generating combined page ---")
    create_combined_page("dataset_distribution.html")

    print("\n" + "=" * 50)
    print("Done! Open: dataset_distribution.html")
    print("=" * 50)


if __name__ == "__main__":
    if not PLOTLY_AVAILABLE:
        print("Plotly required: pip install plotly")
        exit(1)
    run_demo()
