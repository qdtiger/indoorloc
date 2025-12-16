"""
Dataset Distribution Visualization

Visualize spatial distribution of indoor localization datasets.
Generates interactive HTML with 2D/3D views.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, List, Dict
import webbrowser
import tempfile

if TYPE_CHECKING:
    from ..datasets.base import BaseDataset

# Check plotly availability
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Color schemes
COLORSCALE_BLUE = [
    [0.0, '#f7fbff'], [0.15, '#deebf7'], [0.3, '#c6dbef'],
    [0.45, '#9ecae1'], [0.6, '#6baed6'], [0.75, '#4292c6'],
    [0.9, '#2171b5'], [1.0, '#084594']
]

FLOOR_COLORS = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']

# Typography
FONT_FAMILY = "Arial, Helvetica, sans-serif"
TEXT_COLOR = "#2c3e50"
LIGHT_TEXT = "#7f8c8d"
GRID_COLOR = "#e8e8e8"
BORDER_COLOR = "#333333"


def _extract_dataset_data(dataset: "BaseDataset") -> Dict:
    """Extract coordinates, floors, buildings, metadata from dataset."""
    coords, floors, buildings, metadata, signals = [], [], [], [], []

    for i in range(len(dataset)):
        signal, location = dataset[i]
        meta = dataset._metadata[i] if hasattr(dataset, '_metadata') and i < len(dataset._metadata) else {}

        coords.append((location.x, location.y))
        floors.append(location.floor)
        buildings.append(str(location.building_id))
        metadata.append(meta)

        # Get signal data for AP count
        if hasattr(signal, 'data') and signal.data is not None:
            signals.append(signal.data)
        elif hasattr(signal, 'rssi_values') and signal.rssi_values is not None:
            signals.append(signal.rssi_values)
        else:
            signals.append(None)

    return {
        'coords': coords,
        'floors': floors,
        'buildings': buildings,
        'metadata': metadata,
        'signals': signals
    }


def _generate_2d_html(
    data: Dict,
    bin_size: float = 3.0,
    not_detected_value: float = 100.0
) -> str:
    """Generate 2D visualization HTML content."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required: pip install plotly")

    # Process signals for AP count
    signals = data['signals']
    if signals and signals[0] is not None:
        ap_counts = [np.sum(s != not_detected_value) if isinstance(s, np.ndarray) else 0 for s in signals]
    else:
        ap_counts = [0] * len(data['coords'])

    # Process metadata
    metadata = data['metadata']
    user_ids = [m.get('user_id', 0) for m in metadata]
    space_ids = [m.get('space_id', 0) for m in metadata]

    # Normalize coordinates
    raw_x = [c[0] for c in data['coords']]
    raw_y = [c[1] for c in data['coords']]
    x_min, y_min = min(raw_x), min(raw_y)

    df = pd.DataFrame({
        'x': [x - x_min for x in raw_x],
        'y': [y - y_min for y in raw_y],
        'floor': data['floors'],
        'building': data['buildings'],
        'user_id': user_ids,
        'space_id': space_ids,
        'ap_count': ap_counts
    })

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

        fig.add_trace(go.Scatter(
            x=px, y=py,
            mode='markers',
            marker=dict(
                symbol='square',
                size=12,
                color=agg['count'],
                colorscale=COLORSCALE_BLUE,
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
    )

    # Generate HTML with JS to handle URL parameter
    base_html = fig.to_html(include_plotlyjs=True, full_html=True)

    # Build button config as JSON for JS
    import json
    buttons_json = json.dumps(buttons)

    # Inject JS to read URL param and update plot
    inject_js = f'''
<script>
(function() {{
    const params = new URLSearchParams(window.location.search);
    const floorIdx = parseInt(params.get('floor')) || 0;
    const buttons = {buttons_json};
    if (floorIdx >= 0 && floorIdx < buttons.length) {{
        const btn = buttons[floorIdx];
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (plotDiv && btn) {{
            Plotly.update(plotDiv, btn.args[0], btn.args[1]);
        }}
    }}
}})();
</script>
</body>'''

    return base_html.replace('</body>', inject_js)


def _generate_3d_html(
    data: Dict,
    floor_height: float = 5.0
) -> str:
    """Generate 3D visualization HTML content."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required: pip install plotly")

    # Normalize coordinates
    raw_x = [c[0] for c in data['coords']]
    raw_y = [c[1] for c in data['coords']]
    x_min, y_min = min(raw_x), min(raw_y)

    df = pd.DataFrame({
        'x': [x - x_min for x in raw_x],
        'y': [y - y_min for y in raw_y],
        'floor': data['floors'],
        'building': data['buildings']
    })

    fig = go.Figure()
    buttons = []

    unique_buildings = sorted(df['building'].unique())
    all_traces = []

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
        legend=dict(
            x=1.0, y=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#ccc',
            borderwidth=1
        )
    )

    # Generate HTML with JS to handle URL parameter
    base_html = fig.to_html(include_plotlyjs=True, full_html=True)

    # Build button config as JSON for JS
    import json
    buttons_json = json.dumps(buttons)

    # Inject JS to read URL param and update plot
    inject_js = f'''
<script>
(function() {{
    const params = new URLSearchParams(window.location.search);
    const floorIdx = parseInt(params.get('floor')) || 0;
    const buttons = {buttons_json};
    if (floorIdx >= 0 && floorIdx < buttons.length) {{
        const btn = buttons[floorIdx];
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (plotDiv && btn) {{
            Plotly.update(plotDiv, btn.args[0], btn.args[1]);
        }}
    }}
}})();
</script>
</body>'''

    return base_html.replace('</body>', inject_js)


def _generate_combined_html(html_2d_path: str, html_3d_path: str, floor_options: List[str]) -> str:
    """Generate combined HTML page with 2D/3D toggle and synchronized floor selection."""
    # Build floor options HTML
    options_html = '\n'.join([
        f'            <option value="{i}">{opt}</option>'
        for i, opt in enumerate(floor_options)
    ])

    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Dataset Distribution</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: Arial, sans-serif; background: #f5f5f5; }}
        .header {{
            background: white;
            padding: 15px 30px;
            border-bottom: 1px solid #ddd;
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .header h1 {{
            font-size: 18px;
            color: #2c3e50;
            font-weight: 600;
        }}
        .toggle-group {{
            display: flex;
            border: 1px solid #ccc;
            border-radius: 4px;
            overflow: hidden;
        }}
        .toggle-btn {{
            padding: 8px 20px;
            border: none;
            background: white;
            color: #666;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .toggle-btn:hover {{ background: #f0f0f0; }}
        .toggle-btn.active {{
            background: #2171b5;
            color: white;
        }}
        .toggle-btn:first-child {{ border-right: 1px solid #ccc; }}
        .floor-select {{
            padding: 8px 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
            color: #333;
            background: white;
            cursor: pointer;
            min-width: 150px;
        }}
        .floor-select:focus {{ outline: none; border-color: #2171b5; }}
        .frame-container {{
            width: 100%;
            height: calc(100vh - 60px);
        }}
        iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Dataset Distribution</h1>
        <div class="toggle-group">
            <button class="toggle-btn active" onclick="switchView('2d')">2D View</button>
            <button class="toggle-btn" onclick="switchView('3d')">3D View</button>
        </div>
        <select class="floor-select" id="floor-select" onchange="onFloorChange()">
{options_html}
        </select>
    </div>
    <div class="frame-container">
        <iframe id="viz-frame" src="{html_2d_path}"></iframe>
    </div>
    <script>
        let currentView = '2d';
        let currentFloor = 0;

        function switchView(mode) {{
            currentView = mode;
            const btns = document.querySelectorAll('.toggle-btn');
            btns.forEach(btn => btn.classList.remove('active'));
            if (mode === '2d') {{
                btns[0].classList.add('active');
            }} else {{
                btns[1].classList.add('active');
            }}
            updateFrame();
        }}

        function onFloorChange() {{
            currentFloor = document.getElementById('floor-select').value;
            updateFrame();
        }}

        function updateFrame() {{
            const frame = document.getElementById('viz-frame');
            const baseSrc = currentView === '2d' ? '{html_2d_path}' : '{html_3d_path}';
            frame.src = baseSrc + '?floor=' + currentFloor;
        }}
    </script>
</body>
</html>'''


def plot_dataset(
    dataset: "BaseDataset",
    mode: str = "combined",
    output_dir: Optional[Union[str, Path]] = None,
    bin_size: float = 3.0,
    floor_height: float = 5.0,
    open_browser: bool = True,
    inline: bool = False
) -> str:
    """
    Visualize spatial distribution of a dataset.

    Args:
        dataset: IndoorLoc dataset instance.
        mode: Visualization mode - '2d', '3d', or 'combined' (default).
        output_dir: Output directory. If None, uses temp directory.
        bin_size: Grid bin size in meters for 2D view (default 3.0).
        floor_height: Visual height between floors in 3D view (default 5.0).
        open_browser: Whether to open result in browser (default True).
        inline: If True and in Jupyter, display inline (default False).

    Returns:
        Path to the generated HTML file.

    Example:
        >>> import indoorloc as iloc
        >>> train = iloc.UJIndoorLoc(split='train')
        >>> train.plot()  # Quick visualization
        >>>
        >>> # Or with options
        >>> from indoorloc.visualization import plot_dataset
        >>> plot_dataset(train, mode='3d', bin_size=5.0)
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for visualization. "
            "Install it with: pip install plotly"
        )

    # Setup output directory
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "indoorloc_viz"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data from dataset
    print(f"Extracting data from {dataset.dataset_name}...")
    data = _extract_dataset_data(dataset)
    print(f"  {len(data['coords'])} samples, "
          f"{len(set(data['buildings']))} buildings, "
          f"{len(set(data['floors']))} floors")

    # Generate visualizations
    output_path = None

    if mode in ('2d', 'combined'):
        print("Generating 2D visualization...")
        html_2d = _generate_2d_html(data, bin_size=bin_size)
        path_2d = output_dir / "distribution_2d.html"
        with open(path_2d, 'w') as f:
            f.write(html_2d)
        if mode == '2d':
            output_path = path_2d

    if mode in ('3d', 'combined'):
        print("Generating 3D visualization...")
        html_3d = _generate_3d_html(data, floor_height=floor_height)
        path_3d = output_dir / "distribution_3d.html"
        with open(path_3d, 'w') as f:
            f.write(html_3d)
        if mode == '3d':
            output_path = path_3d

    if mode == 'combined':
        print("Generating combined view...")
        # Build floor options for the dropdown
        df = pd.DataFrame({
            'floor': data['floors'],
            'building': data['buildings']
        })
        groups = df.groupby(['building', 'floor'])
        combinations = sorted(groups.groups.keys(), key=lambda x: (str(x[0]), int(x[1])))
        floor_options = [f"B{bldg} Floor {flr}" for bldg, flr in combinations]

        html_combined = _generate_combined_html("distribution_2d.html", "distribution_3d.html", floor_options)
        output_path = output_dir / "distribution.html"
        with open(output_path, 'w') as f:
            f.write(html_combined)

    print(f"Saved: {output_path}")

    # Display
    if inline:
        try:
            from IPython.display import IFrame, display
            display(IFrame(str(output_path), width=920, height=720))
        except ImportError:
            pass
    elif open_browser:
        webbrowser.open(f"file://{output_path}")

    return str(output_path)
