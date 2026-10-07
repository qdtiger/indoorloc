from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("INDOORLOC_DATA") or Path.home() / ".cache" / "indoorloc" / "datasets")
UJI_ROOT = Path(os.environ.get("INDOORLOC_UJI_ROOT") or os.environ.get("UJI_ROOT") or DATA_ROOT / "ujiindoorloc")

# Every third-party package the 0.1 code imports, besides numpy: none may load when a layer is imported.
HEAVY = ("torch", "sklearn", "scipy", "pandas", "timm", "skada", "matplotlib", "plotly", "h5py",
         "requests", "urllib3", "yaml", "joblib", "tqdm", "IPython", "torchvision", "cv2", "seaborn", "deepmimo")
# The declared modules allowed to import them at top level (the package methods.deep itself is torch-free).
BRIDGES = ("indoorloc.datasets.torch_adapter", "indoorloc.methods.deep.backbones", "indoorloc.methods.deep.heads",
           "indoorloc.methods.deep.training")
