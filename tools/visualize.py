#!/usr/bin/env python
"""
Visualize indoor localization datasets.

Usage:
    python tools/visualize.py ujindoorloc
    python tools/visualize.py ujindoorloc --mode 3d
    python tools/visualize.py ujindoorloc --output viz_output/
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize dataset distribution')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., ujindoorloc)')
    parser.add_argument('--split', type=str, default='train', help='Dataset split')
    parser.add_argument('--mode', type=str, default='combined',
                        choices=['2d', '3d', 'combined'],
                        help='Visualization mode')
    parser.add_argument('--output', '-o', type=str, help='Output directory')
    parser.add_argument('--bin-size', type=float, default=3.0,
                        help='Grid bin size for 2D view (meters)')
    parser.add_argument('--floor-height', type=float, default=5.0,
                        help='Floor height for 3D view (meters)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open browser automatically')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("IndoorLoc Dataset Visualization")
    print("=" * 60)

    try:
        import indoorloc as iloc
        from indoorloc.visualization import plot_dataset

        print(f"\nLoading {args.dataset} ({args.split})...")
        dataset = iloc.load_dataset(args.dataset, split=args.split)
        print(f"  Samples: {len(dataset)}")

        output_dir = args.output or f"viz_{args.dataset}"

        path = plot_dataset(
            dataset,
            mode=args.mode,
            output_dir=output_dir,
            bin_size=args.bin_size,
            floor_height=args.floor_height,
            open_browser=not args.no_browser
        )

        print(f"\nVisualization saved to: {path}")
        return 0

    except ImportError as e:
        print(f"Error: {e}")
        print("Install plotly: pip install plotly")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
