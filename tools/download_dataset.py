#!/usr/bin/env python
"""
Download datasets for IndoorLoc

Usage:
    python tools/download_dataset.py ujindoorloc

    # Specify output directory
    python tools/download_dataset.py ujindoorloc --output data/ujindoorloc

    # Force re-download
    python tools/download_dataset.py ujindoorloc --force
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


DATASETS = {
    'ujindoorloc': {
        'name': 'UJIndoorLoc',
        'url': 'https://archive.ics.uci.edu/dataset/310/ujiindoorloc',
        'default_path': 'data/ujindoorloc',
        'description': 'WiFi fingerprinting dataset from Universitat Jaume I, Spain',
    },
    # Add more datasets here as they are implemented
}


def parse_args():
    parser = argparse.ArgumentParser(description='Download indoor localization datasets')

    parser.add_argument('dataset', choices=list(DATASETS.keys()) + ['list'],
                        help='Dataset name to download, or "list" to show available datasets')
    parser.add_argument('--output', '-o', type=str, help='Output directory')
    parser.add_argument('--force', '-f', action='store_true', help='Force re-download')

    return parser.parse_args()


def list_datasets():
    print("\nAvailable Datasets:")
    print("=" * 60)

    for key, info in DATASETS.items():
        print(f"\n{info['name']} ({key})")
        print(f"  Description: {info['description']}")
        print(f"  Default path: {info['default_path']}")
        print(f"  URL: {info['url']}")

    print("\n" + "=" * 60)


def download_ujindoorloc(output_dir: str, force: bool = False):
    """Download UJIndoorLoc dataset."""
    from indoorloc.datasets import UJIndoorLocDataset

    print(f"\nDownloading UJIndoorLoc dataset to: {output_dir}")
    UJIndoorLocDataset.download(output_dir, force=force)

    # Verify download
    print("\nVerifying dataset...")
    try:
        dataset = UJIndoorLocDataset(output_dir, split='train')
        print(f"  Training samples: {len(dataset)}")

        dataset = UJIndoorLocDataset(output_dir, split='test')
        print(f"  Test samples: {len(dataset)}")

        stats = dataset.get_statistics()
        print(f"  Buildings: {stats['num_buildings']}")
        print(f"  Floors: {stats['num_floors']}")
        print(f"  Access Points: {dataset.num_aps}")

        print("\nDataset downloaded and verified successfully!")
    except Exception as e:
        print(f"Warning: Verification failed: {e}")


def main():
    args = parse_args()

    if args.dataset == 'list':
        list_datasets()
        return 0

    dataset_info = DATASETS[args.dataset]
    output_dir = args.output or dataset_info['default_path']

    print("=" * 60)
    print(f"IndoorLoc Dataset Downloader")
    print("=" * 60)
    print(f"\nDataset: {dataset_info['name']}")
    print(f"Output: {output_dir}")

    if args.dataset == 'ujindoorloc':
        download_ujindoorloc(output_dir, args.force)
    else:
        print(f"Download not implemented for: {args.dataset}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
