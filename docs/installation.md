# Installation (Advanced)

This page contains optional extras, legacy environments, and troubleshooting tips.
For the recommended install, see `README.md`.

## Optional extras

```bash
pip install "indoorloc[vision]"     # torchvision + opencv-python
pip install "indoorloc[datasets]"   # scipy + h5py + requests
pip install "indoorloc[deep]"       # timm backbones
pip install "indoorloc[deepmimo]"   # DeepMIMO (requires Python >= 3.10)
```

## Legacy (Python 3.8 + PyTorch 1.10.1)

### GPU (CUDA 11.3)

```bash
conda create -n indoorloc-py38 python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate indoorloc-py38
pip install "indoorloc[full]"
```

### CPU-only

```bash
conda create -n indoorloc-py38 python=3.8 pytorch==1.10.1 torchvision==0.11.2 cpuonly -c pytorch -y
conda activate indoorloc-py38
pip install "indoorloc[full]"
```

## Troubleshooting

- If you already created the env, remove it first: `conda env remove -n indoorloc`.
- If `torch.cuda.is_available()` is `False` on a GPU machine, verify your NVIDIA driver is installed and matches the CUDA runtime you selected.

