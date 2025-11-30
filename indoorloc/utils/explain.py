"""
Configuration Explanation Module

Provides educational guidance for transitioning from beginner to expert usage.
Helps users understand model parameters, dataset characteristics, and config files.
"""
from typing import Optional, Dict, Any
from pathlib import Path


# Model documentation data
MODEL_DOCS = {
    'resnet18': {
        'name': 'ResNet18',
        'category': 'CNN (Convolutional Neural Network)',
        'description': '18层残差网络，ImageNet预训练，适合RSSI指纹定位',
        'use_cases': ['WiFi/BLE RSSI 指纹定位', '中等规模数据集 (1k-50k样本)'],
        'recommended_datasets': ['UJIndoorLoc', 'Tampere', 'SODIndoorLoc'],
        'expected_accuracy': '5-10m (取决于数据集和训练设置)',
        'params': {
            'model_name': {
                'default': 'resnet18',
                'description': '骨干网络名称',
                'options': ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'mobilenetv3_small'],
                'tips': '更大的模型精度更高但更慢'
            },
            'pretrained': {
                'default': True,
                'description': '使用预训练权重',
                'tips': '强烈推荐开启，可加速收敛并提高精度'
            },
            'dropout': {
                'default': 0.5,
                'description': '防止过拟合',
                'tips': '数据量少时增大 (0.5-0.7)，数据量大时减小 (0.2-0.3)'
            },
            'lr': {
                'default': 0.001,
                'description': '学习率',
                'tips': '最重要的超参数！精调时减小 (5e-4)'
            },
            'epochs': {
                'default': 100,
                'description': '训练轮数',
                'tips': '配合 early_stopping 使用'
            },
            'batch_size': {
                'default': 64,
                'description': '批次大小',
                'tips': '显存不足时减小'
            }
        },
        'quick_start': "model = iloc.create_model('resnet18', dataset=train)\nmodel.fit(train, epochs=50)",
        'config_path': 'configs/wifi/resnet18_ujindoorloc.yaml'
    },
    'efficientnet_b0': {
        'name': 'EfficientNet-B0',
        'category': 'CNN (Efficient Architecture)',
        'description': '高效卷积网络，参数更少，精度更高',
        'use_cases': ['WiFi/BLE RSSI 指纹定位', '大规模数据集'],
        'recommended_datasets': ['UJIndoorLoc', 'CSI数据集'],
        'expected_accuracy': '4-8m',
        'params': {
            'model_name': {
                'default': 'efficientnet_b0',
                'description': '骨干网络名称',
                'options': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2'],
                'tips': 'b0最快，b2最准'
            },
            'pretrained': {
                'default': True,
                'description': '使用预训练权重',
                'tips': '强烈推荐开启'
            }
        },
        'quick_start': "model = iloc.create_model('efficientnet_b0', dataset=train)\nmodel.fit(train, epochs=50)",
        'config_path': 'configs/wifi/efficientnet_ujindoorloc.yaml'
    },
    'wknn': {
        'name': 'Weighted k-NN',
        'category': 'Traditional ML (Instance-based)',
        'description': '加权K近邻，距离越近权重越大',
        'use_cases': ['小规模数据集 (<5000样本)', '快速原型验证', '无需GPU'],
        'recommended_datasets': ['所有RSSI数据集'],
        'expected_accuracy': '7-15m',
        'params': {
            'k': {
                'default': 5,
                'description': '近邻数量',
                'tips': '通常 3-10，交叉验证选择最优'
            },
            'weights': {
                'default': 'distance',
                'description': '权重方式',
                'options': ['uniform', 'distance'],
                'tips': 'distance 通常更好'
            },
            'metric': {
                'default': 'euclidean',
                'description': '距离度量',
                'options': ['euclidean', 'manhattan', 'cosine'],
                'tips': 'euclidean 适合大多数场景'
            }
        },
        'quick_start': "model = iloc.create_model('wknn', k=5)\nmodel.fit(train)",
        'config_path': 'configs/wifi/wknn_ujindoorloc.yaml'
    },
    'knn': {
        'name': 'k-Nearest Neighbors',
        'category': 'Traditional ML (Instance-based)',
        'description': '经典K近邻算法，简单有效',
        'use_cases': ['小规模数据集', '基准对比', '教学演示'],
        'recommended_datasets': ['所有RSSI数据集'],
        'expected_accuracy': '8-18m',
        'params': {
            'k': {
                'default': 5,
                'description': '近邻数量',
                'tips': '通常 1-10'
            }
        },
        'quick_start': "model = iloc.create_model('knn', k=5)\nmodel.fit(train)",
        'config_path': 'configs/wifi/knn_ujindoorloc.yaml'
    }
}

# Dataset documentation data
DATASET_DOCS = {
    'ujindoorloc': {
        'name': 'UJIndoorLoc',
        'full_name': 'Universitat Jaume I Indoor Localization',
        'description': '西班牙 Jaume I 大学收集的大规模 WiFi 指纹数据集',
        'signal_type': 'WiFi RSSI',
        'stats': {
            'train_samples': 19937,
            'test_samples': 1111,
            'num_waps': 520,
            'num_floors': 5,
            'num_buildings': 3,
            'area': '~110,000 m²'
        },
        'characteristics': [
            '最常用的室内定位基准数据集',
            '包含多建筑、多楼层场景',
            '训练集和测试集在不同时间采集',
            '存在信号漂移和设备差异'
        ],
        'benchmark': {
            'sota': '2.45m (GBDT + Sample Diff, 2021)',
            'knn_baseline': '7.9m',
            'wknn_baseline': '7.3m'
        },
        'quick_start': "train, test = iloc.load_dataset('ujindoorloc')",
        'source': 'UCI Machine Learning Repository'
    },
    'tampere': {
        'name': 'Tampere',
        'full_name': 'Tampere University Indoor Dataset',
        'description': '芬兰坦佩雷大学收集的 WiFi 指纹数据集',
        'signal_type': 'WiFi RSSI',
        'stats': {
            'samples': '~1000',
            'num_waps': 489,
            'num_floors': 4,
            'num_buildings': 1
        },
        'characteristics': [
            '单建筑多楼层场景',
            '数据量较小，适合快速实验',
            '信号质量较好'
        ],
        'quick_start': "train, test = iloc.load_dataset('tampere')",
        'source': 'Tampere University'
    },
    'sodindoorloc': {
        'name': 'SODIndoorLoc',
        'full_name': 'SOD Indoor Localization Dataset',
        'description': '中国收集的大规模室内定位数据集',
        'signal_type': 'WiFi RSSI',
        'stats': {
            'buildings': ['CETC331', 'HCXY'],
            'floors': 'varies by building'
        },
        'characteristics': [
            '包含多个中国建筑场景',
            '数据量较大',
            '适合跨建筑泛化研究'
        ],
        'quick_start': "train = iloc.SODIndoorLoc(building='CETC331', download=True)",
        'source': 'GitHub'
    }
}


def explain_model(model_name: str, show_all: bool = False):
    """
    Print educational explanation of a model's parameters and usage.

    Args:
        model_name: Model name (e.g., 'resnet18', 'wknn', 'knn')
        show_all: Whether to show all available models

    Example:
        >>> import indoorloc as iloc
        >>> iloc.explain_model('resnet18')
    """
    if show_all or model_name.lower() == 'all':
        print("\n" + "=" * 60)
        print("可用模型列表")
        print("=" * 60)
        for name, doc in MODEL_DOCS.items():
            print(f"\n  {name:<20} - {doc['category']}")
            print(f"  {' ' * 20}   {doc['description'][:40]}...")
        print("\n" + "=" * 60)
        print("使用 iloc.explain_model('model_name') 查看详细参数")
        print("=" * 60 + "\n")
        return

    model_key = model_name.lower()
    if model_key not in MODEL_DOCS:
        print(f"未知模型: {model_name}")
        print(f"可用模型: {', '.join(MODEL_DOCS.keys())}")
        print("使用 iloc.explain_model('all') 查看所有模型")
        return

    doc = MODEL_DOCS[model_key]

    # Print header
    print("\n" + "╔" + "═" * 60 + "╗")
    print(f"║{doc['name']:^60}║")
    print("╠" + "═" * 60 + "╣")
    print(f"║ 类型: {doc['category']:<52}║")
    print(f"║ {doc['description']:<59}║")
    print("╠" + "═" * 60 + "╣")

    # Use cases
    print("║ 适用场景:                                                  ║")
    for use_case in doc['use_cases']:
        print(f"║   • {use_case:<55}║")

    # Recommended datasets
    datasets = ', '.join(doc['recommended_datasets'])
    print(f"║ 推荐数据集: {datasets:<47}║")

    # Expected accuracy
    print(f"║ 预计精度: {doc['expected_accuracy']:<49}║")

    # Parameters
    print("╠" + "═" * 60 + "╣")
    print("║                      可调参数                              ║")
    print("╠" + "─" * 60 + "╣")
    print("║ 参数            │ 默认值      │ 说明                       ║")
    print("╠" + "─" * 60 + "╣")

    for param_name, param_info in doc['params'].items():
        default = str(param_info['default'])
        desc = param_info['description']
        print(f"║ {param_name:<15} │ {default:<11} │ {desc:<27}║")

        if 'tips' in param_info:
            tips = param_info['tips']
            # Split long tips
            if len(tips) > 45:
                print(f"║ {'':<15} │ {'':<11} │   {tips[:43]}║")
                print(f"║ {'':<15} │ {'':<11} │   {tips[43:]:<43}║")
            else:
                print(f"║ {'':<15} │ {'':<11} │   {tips:<43}║")

    print("╚" + "═" * 60 + "╝")

    # Quick start
    print("\n💡 快速开始:")
    for line in doc['quick_start'].split('\n'):
        print(f"   {line}")

    # Config path
    print(f"\n📖 进阶配置:")
    print(f"   python tools/train.py {doc['config_path']}")
    print()


def explain_dataset(dataset_name: str, show_all: bool = False):
    """
    Print educational explanation of a dataset's characteristics.

    Args:
        dataset_name: Dataset name (e.g., 'ujindoorloc', 'tampere')
        show_all: Whether to show all available datasets

    Example:
        >>> import indoorloc as iloc
        >>> iloc.explain_dataset('ujindoorloc')
    """
    if show_all or dataset_name.lower() == 'all':
        print("\n" + "=" * 60)
        print("可用数据集列表")
        print("=" * 60)
        for name, doc in DATASET_DOCS.items():
            print(f"\n  {name:<20} - {doc['signal_type']}")
            print(f"  {' ' * 20}   {doc['description'][:40]}...")
        print("\n" + "=" * 60)
        print("使用 iloc.explain_dataset('dataset_name') 查看详细信息")
        print("=" * 60 + "\n")
        return

    dataset_key = dataset_name.lower()
    if dataset_key not in DATASET_DOCS:
        print(f"未知数据集: {dataset_name}")
        print(f"可用数据集: {', '.join(DATASET_DOCS.keys())}")
        print("使用 iloc.explain_dataset('all') 查看所有数据集")
        return

    doc = DATASET_DOCS[dataset_key]

    # Print header
    print("\n" + "╔" + "═" * 60 + "╗")
    print(f"║{doc['name']:^60}║")
    print("╠" + "═" * 60 + "╣")
    print(f"║ {doc['full_name']:<59}║")
    print(f"║ 信号类型: {doc['signal_type']:<49}║")
    print("╠" + "═" * 60 + "╣")

    # Description
    desc = doc['description']
    print(f"║ {desc:<59}║")

    # Stats
    print("╠" + "═" * 60 + "╣")
    print("║ 数据集统计:                                                ║")
    for stat_name, stat_value in doc['stats'].items():
        print(f"║   {stat_name}: {str(stat_value):<50}║")

    # Characteristics
    print("╠" + "═" * 60 + "╣")
    print("║ 特点:                                                      ║")
    for char in doc['characteristics']:
        if len(char) > 55:
            print(f"║   • {char[:55]}║")
            print(f"║     {char[55:]:<55}║")
        else:
            print(f"║   • {char:<55}║")

    # Benchmark (if available)
    if 'benchmark' in doc:
        print("╠" + "═" * 60 + "╣")
        print("║ 基准结果:                                                  ║")
        for bench_name, bench_value in doc['benchmark'].items():
            print(f"║   {bench_name}: {bench_value:<50}║")

    print("╚" + "═" * 60 + "╝")

    # Quick start
    print("\n💡 快速开始:")
    print(f"   {doc['quick_start']}")

    # Source
    print(f"\n📚 数据来源: {doc['source']}")
    print()


def explain_config(config_path: str):
    """
    Print educational explanation of a configuration file.

    Args:
        config_path: Path to YAML config file

    Example:
        >>> import indoorloc as iloc
        >>> iloc.explain_config('configs/wifi/resnet18_ujindoorloc.yaml')
    """
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        return

    print("\n" + "=" * 60)
    print(f"配置文件: {config_path.name}")
    print("=" * 60)

    # Read and print the file content with line numbers
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(content)

    # Print usage tips
    print("=" * 60)
    print("使用方法:")
    print(f"  python tools/train.py {config_path}")
    print()
    print("命令行覆盖参数:")
    print(f"  python tools/train.py {config_path} --train.lr 5e-4")
    print(f"  python tools/train.py {config_path} --model.backbone.model_name efficientnet_b0")
    print("=" * 60 + "\n")


__all__ = ['explain_model', 'explain_dataset', 'explain_config']
