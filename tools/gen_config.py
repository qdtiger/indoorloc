#!/usr/bin/env python3
"""
Generate configuration files for IndoorLoc experiments.

Usage:
    python tools/gen_config.py resnet18 ujindoorloc -o my_config.yaml
    python tools/gen_config.py knn tampere -o knn_tampere.yaml
    python tools/gen_config.py --list-models
    python tools/gen_config.py --list-datasets
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


# Model templates
MODEL_CONFIGS = {
    'knn': {
        'type': 'traditional',
        'model': {
            'type': 'KNNLocalizer',
            'k': 5,
            'weights': 'distance',
            'metric': 'euclidean',
            'predict_floor': True,
            'predict_building': True,
        }
    },
    'wknn': {
        'type': 'traditional',
        'model': {
            'type': 'WKNNLocalizer',
            'k': 5,
            'weights': 'distance',
            'predict_floor': True,
            'predict_building': True,
        }
    },
    'mlp': {
        'type': 'deep',
        'model': {
            'type': 'DeepLocalizer',
            'backbone': {
                'type': 'MLPBackbone',
                'hidden_dims': [512, 256, 128],
                'dropout': 0.3,
                'batch_norm': True,
            },
            'head': {
                'type': 'RegressionHead',
                'num_coords': 2,
                'hidden_dims': [64],
                'dropout': 0.2,
            }
        }
    },
    'resnet18': {
        'type': 'deep',
        'model': {
            'type': 'DeepLocalizer',
            'backbone': {
                'type': 'TimmBackbone',
                'model_name': 'resnet18',
                'pretrained': True,
                'input_type': '1d',
            },
            'head': {
                'type': 'RegressionHead',
                'num_coords': 2,
                'hidden_dims': [256, 128],
                'dropout': 0.5,
            }
        }
    },
    'resnet34': {
        'type': 'deep',
        'model': {
            'type': 'DeepLocalizer',
            'backbone': {
                'type': 'TimmBackbone',
                'model_name': 'resnet34',
                'pretrained': True,
                'input_type': '1d',
            },
            'head': {
                'type': 'RegressionHead',
                'num_coords': 2,
                'hidden_dims': [256, 128],
                'dropout': 0.5,
            }
        }
    },
    'resnet50': {
        'type': 'deep',
        'model': {
            'type': 'DeepLocalizer',
            'backbone': {
                'type': 'TimmBackbone',
                'model_name': 'resnet50',
                'pretrained': True,
                'input_type': '1d',
            },
            'head': {
                'type': 'RegressionHead',
                'num_coords': 2,
                'hidden_dims': [512, 256],
                'dropout': 0.5,
            }
        }
    },
    'efficientnet_b0': {
        'type': 'deep',
        'model': {
            'type': 'DeepLocalizer',
            'backbone': {
                'type': 'TimmBackbone',
                'model_name': 'efficientnet_b0',
                'pretrained': True,
                'input_type': '1d',
            },
            'head': {
                'type': 'RegressionHead',
                'num_coords': 2,
                'hidden_dims': [256, 128],
                'dropout': 0.5,
            }
        }
    },
    'cnn1d': {
        'type': 'deep',
        'model': {
            'type': 'DeepLocalizer',
            'backbone': {
                'type': 'CNN1DBackbone',
                'channels': [64, 128, 256],
                'kernel_sizes': [7, 5, 3],
                'dropout': 0.3,
                'batch_norm': True,
            },
            'head': {
                'type': 'RegressionHead',
                'num_coords': 2,
                'hidden_dims': [128, 64],
                'dropout': 0.3,
            }
        }
    },
}

# Dataset templates
DATASET_CONFIGS = {
    'ujindoorloc': {
        'type': 'UJIndoorLocDataset',
        'num_waps': 520,
        'num_floors': 5,
        'num_buildings': 3,
        'download': True,
    },
    'tampere': {
        'type': 'TampereDataset',
        'num_waps': 489,
        'num_floors': 5,
        'num_buildings': 1,
        'download': True,
    },
    'sodindoorloc': {
        'type': 'SODIndoorLocDataset',
        'num_waps': 168,
        'num_floors': 1,
        'num_buildings': 1,
        'download': True,
    },
}

# Training schedule templates
TRAIN_CONFIGS = {
    'traditional': {
        # Traditional models don't need extensive training config
    },
    'deep': {
        'epochs': 100,
        'batch_size': 64,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'early_stopping': 10,
        'device': None,  # Auto-detect
        'fp16': False,
        'num_workers': 0,
        'pin_memory': True,
        'logging_steps': 10,
    }
}


def generate_config(model_name: str, dataset_name: str) -> dict:
    """Generate a complete configuration dictionary."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    model_template = MODEL_CONFIGS[model_name]
    dataset_template = DATASET_CONFIGS[dataset_name]
    model_type = model_template['type']

    config = {
        '# Generated by gen_config.py': None,
        '# Model': model_name,
        '# Dataset': dataset_name,
    }

    # Dataset section
    config['dataset'] = dataset_template.copy()

    # Model section
    config['model'] = model_template['model'].copy()

    # Add dataset-specific head config for deep models
    if model_type == 'deep' and 'head' in config['model']:
        head = config['model']['head']
        if dataset_template['num_floors'] > 1:
            head['type'] = 'HybridHead'
            head['num_floors'] = dataset_template['num_floors']
        if dataset_template['num_buildings'] > 1:
            head['type'] = 'HybridHead'
            head['num_buildings'] = dataset_template['num_buildings']

    # Training section (for deep learning models)
    if model_type == 'deep':
        config['train'] = TRAIN_CONFIGS['deep'].copy()

    # Evaluation section
    config['evaluation'] = {
        'metrics': [
            {'type': 'MeanPositionError'},
            {'type': 'MedianPositionError'},
            {'type': 'FloorAccuracy'},
            {'type': 'BuildingAccuracy'},
        ]
    }

    # Work directory
    config['work_dir'] = f"work_dirs/{model_name}_{dataset_name}"

    return config


def save_config(config: dict, filepath: str):
    """Save configuration to YAML file."""
    # Remove comment markers (they were just for display)
    clean_config = {k: v for k, v in config.items() if not k.startswith('#')}

    with open(filepath, 'w') as f:
        # Add header comment
        f.write(f"# IndoorLoc Configuration\n")
        f.write(f"# Generated by: python tools/gen_config.py\n")
        f.write(f"#\n")
        f.write(f"# Usage:\n")
        f.write(f"#   python tools/train.py {filepath}\n")
        f.write(f"#   python tools/train.py {filepath} --train.epochs 200\n")
        f.write(f"\n")
        yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def list_models():
    """Print available models."""
    print("\nAvailable Models:")
    print("-" * 50)
    print(f"{'Model':<20} {'Type':<15} {'Description'}")
    print("-" * 50)
    for name, cfg in MODEL_CONFIGS.items():
        model_type = cfg['type']
        desc = cfg['model'].get('backbone', {}).get('type', cfg['model']['type'])
        print(f"{name:<20} {model_type:<15} {desc}")
    print()


def list_datasets():
    """Print available datasets."""
    print("\nAvailable Datasets:")
    print("-" * 60)
    print(f"{'Dataset':<20} {'APs':<8} {'Floors':<8} {'Buildings'}")
    print("-" * 60)
    for name, cfg in DATASET_CONFIGS.items():
        print(f"{name:<20} {cfg['num_waps']:<8} {cfg['num_floors']:<8} {cfg['num_buildings']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate IndoorLoc configuration files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s resnet18 ujindoorloc -o config.yaml
  %(prog)s knn tampere -o knn_tampere.yaml
  %(prog)s --list-models
  %(prog)s --list-datasets
        """
    )

    parser.add_argument('model', nargs='?', help='Model name (e.g., resnet18, knn, mlp)')
    parser.add_argument('dataset', nargs='?', help='Dataset name (e.g., ujindoorloc, tampere)')
    parser.add_argument('-o', '--output', default='config.yaml', help='Output file path')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--list-datasets', action='store_true', help='List available datasets')

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if args.list_datasets:
        list_datasets()
        return

    if not args.model or not args.dataset:
        parser.print_help()
        print("\nError: Both model and dataset are required.")
        print("Use --list-models and --list-datasets to see available options.")
        sys.exit(1)

    try:
        config = generate_config(args.model, args.dataset)
        save_config(config, args.output)
        print(f"Configuration saved to: {args.output}")
        print(f"\nTo train, run:")
        print(f"  python tools/train.py {args.output}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
