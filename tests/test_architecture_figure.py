"""The architecture figure may only mark as available what the package ships."""
from __future__ import annotations

import importlib.util
import sys

from conftest import PROJECT


def test_every_available_chip_names_a_shipped_module():
    sys.path.insert(0, str(PROJECT / "docs" / "architecture"))
    try:
        import figure
    finally:
        sys.path.pop(0)
    for lid, _, _, _, chips in figure.LAYERS:
        for english, chinese, status, module in chips:
            assert status in ("on", "part", "plan"), (lid, english)
            assert english and chinese
            if status == "plan":
                assert module is None, f"{english}: a planned chip names no module"
            else:
                assert module and importlib.util.find_spec(module) is not None, f"{lid} {english}: {module} missing"
