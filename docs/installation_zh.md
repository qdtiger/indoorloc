# 安装（进阶）

本页面包含：可选 extras、legacy 环境、以及常见排错。
推荐安装方式请看 `README_zh.md`。

## 可选 extras

```bash
pip install "indoorloc[vision]"     # torchvision + opencv-python
pip install "indoorloc[datasets]"   # scipy + h5py + requests
pip install "indoorloc[deep]"       # timm backbones
pip install "indoorloc[deepmimo]"   # DeepMIMO（需要 Python >= 3.10）
```

## Legacy（Python 3.8 + PyTorch 1.10.1）

### GPU (CUDA 11.3)

```bash
conda create -n indoorloc-py38 python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate indoorloc-py38
pip install "indoorloc[full]"
```

### 仅 CPU

```bash
conda create -n indoorloc-py38 python=3.8 pytorch==1.10.1 torchvision==0.11.2 cpuonly -c pytorch -y
conda activate indoorloc-py38
pip install "indoorloc[full]"
```

## 常见问题

- 如果环境名已存在：`conda env remove -n indoorloc`
- 如果在 GPU 机器上 `torch.cuda.is_available()` 为 `False`：先确认 NVIDIA 驱动正常安装，并且与所选 CUDA runtime 匹配。

