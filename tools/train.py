#!/usr/bin/env python
"""
Train Script for IndoorLoc (Development Wrapper)

This is a thin wrapper for development use. The actual implementation
is in indoorloc.tools.train.

For installed package, use: indoorloc-train

Usage:
    python tools/train.py configs/wifi/knn_ujindoorloc.yaml
"""
import sys
from pathlib import Path

# Add project root to path for development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indoorloc.tools.train import main

if __name__ == '__main__':
    sys.exit(main())
