# IndoorLoc Configuration System

This directory contains YAML configuration files for training and evaluating indoor localization models.

## Quick Start

### Beginners: Use Python Code (Recommended)

```python
import indoorloc as iloc

# 3 lines to train and evaluate
train = iloc.load_dataset('ujindoorloc', split='train')
test = iloc.load_dataset('ujindoorloc', split='test')
model = iloc.create_model('resnet18', dataset=train)
results = model.fit(train, epochs=50).evaluate(test)
print(results.summary())
```

### Advanced: Use Config Files

```bash
# Use pre-defined config
python tools/train.py indoorloc/configs/wifi/resnet18_ujindoorloc.yaml

# Override parameters via command line
python tools/train.py indoorloc/configs/wifi/resnet18_ujindoorloc.yaml \
    --model.backbone.model_name efficientnet_b0 \
    --train.lr 5e-4 \
    --train.epochs 200
```

## Directory Structure

```
configs/
├── _base_/                    # Base configurations (inherited by others)
│   ├── default.yaml           # Default runtime settings
│   ├── datasets/              # Dataset configurations
│   │   ├── ujindoorloc.yaml
│   │   ├── tampere.yaml
│   │   └── sodindoorloc.yaml
│   ├── models/                # Model architectures
│   │   ├── knn.yaml           # k-Nearest Neighbors
│   │   ├── wknn.yaml          # Weighted k-NN
│   │   ├── mlp.yaml           # Multi-Layer Perceptron
│   │   ├── resnet.yaml        # ResNet (via timm)
│   │   └── cnn1d.yaml         # 1D Convolutional Network
│   └── schedules/             # Training schedules
│       ├── schedule_1x.yaml   # Standard (100 epochs)
│       └── schedule_2x.yaml   # Extended (200 epochs)
└── wifi/                      # Complete task configs
    ├── knn_ujindoorloc.yaml
    ├── wknn_ujindoorloc.yaml
    ├── resnet18_ujindoorloc.yaml
    ├── mlp_ujindoorloc.yaml
    └── efficientnet_ujindoorloc.yaml
```

## Config Inheritance

Use `_base_` to inherit from base configurations:

```yaml
_base_:
  - ../_base_/default.yaml
  - ../_base_/datasets/ujindoorloc.yaml
  - ../_base_/models/resnet.yaml
  - ../_base_/schedules/schedule_1x.yaml

# Override specific parameters
model:
  backbone:
    model_name: efficientnet_b0  # Change only the model name
  head:
    dropout: 0.3                 # Adjust dropout

train:
  epochs: 150                    # More training
  lr: 5e-4                       # Lower learning rate
```

## Available Models

| Type | Config | Description |
|------|--------|-------------|
| KNN | `_base_/models/knn.yaml` | k-Nearest Neighbors (traditional) |
| WKNN | `_base_/models/wknn.yaml` | Weighted k-NN |
| MLP | `_base_/models/mlp.yaml` | Multi-Layer Perceptron |
| ResNet | `_base_/models/resnet.yaml` | ResNet via timm (resnet18/34/50) |
| CNN1D | `_base_/models/cnn1d.yaml` | 1D Convolutional Network |

### Model Parameters

**KNN/WKNN:**
- `k`: Number of neighbors (default: 5)
- `weights`: 'uniform' or 'distance'
- `metric`: 'euclidean', 'manhattan', etc.

**Deep Learning (MLP, ResNet, CNN1D):**
- `backbone.hidden_dims`: Layer dimensions
- `backbone.dropout`: Dropout rate
- `head.num_coords`: Output coordinates (2 for x,y)
- `head.num_floors`: Number of floors (for classification)

## Training Schedules

| Schedule | Epochs | Batch Size | Description |
|----------|--------|------------|-------------|
| `schedule_1x` | 100 | 64 | Standard training |
| `schedule_2x` | 200 | 128 | Extended with fp16 |

## Command Line Overrides

Any config value can be overridden via command line:

```bash
# Change model
python tools/train.py config.yaml --model.backbone.model_name resnet50

# Change training
python tools/train.py config.yaml --train.epochs 200 --train.lr 1e-4

# Enable GPU + mixed precision
python tools/train.py config.yaml --train.device cuda --train.fp16 true

# Change output directory
python tools/train.py config.yaml --work_dir my_experiment
```

## Generate Custom Config

Use the config generator to create a starting point:

```bash
python tools/gen_config.py resnet18 ujindoorloc -o my_config.yaml
```

Then edit `my_config.yaml` to customize.

## View Available Parameters

```python
import indoorloc as iloc

# Show all parameters for a model type
iloc.print_config_help('TimmBackbone')
iloc.print_config_help('KNNLocalizer')
iloc.print_config_help('RegressionHead')
```

## Examples

### Train ResNet18 on UJIndoorLoc

```bash
python tools/train.py indoorloc/configs/wifi/resnet18_ujindoorloc.yaml
```

### Train with Custom Settings

```yaml
# my_experiment.yaml
_base_:
  - indoorloc/configs/wifi/resnet18_ujindoorloc.yaml

model:
  backbone:
    model_name: efficientnet_b0
    pretrained: true

train:
  epochs: 150
  batch_size: 32
  lr: 3e-4
  fp16: true

work_dir: work_dirs/my_experiment
```

```bash
python tools/train.py my_experiment.yaml
```

### Compare Multiple Models

```bash
# Run different models
python tools/train.py indoorloc/configs/wifi/knn_ujindoorloc.yaml
python tools/train.py indoorloc/configs/wifi/wknn_ujindoorloc.yaml
python tools/train.py indoorloc/configs/wifi/resnet18_ujindoorloc.yaml

# Results saved to respective work_dirs
```
