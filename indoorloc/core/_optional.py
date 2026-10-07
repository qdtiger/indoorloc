"""The one way to use an optional dependency: import it inside the function that needs it."""
from __future__ import annotations

import importlib


def requires(module: str, extra: str):
    """Import ``module`` or raise an ImportError that names the pip extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(f"this feature needs {module}: pip install 'indoorloc[{extra}]'") from err
