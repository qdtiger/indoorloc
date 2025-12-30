#!/usr/bin/env python
"""
Test Script for IndoorLoc (Development Wrapper)

This is a thin wrapper for development use. The actual implementation
is in indoorloc.tools.test.

For installed package, use: indoorloc-test

Usage:
    python tools/test.py indoorloc/configs/wifi/knn_ujindoorloc.yaml checkpoint.pkl
"""
import sys
from pathlib import Path

# Add project root to path for development
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indoorloc.tools.test import main

if __name__ == '__main__':
    sys.exit(main())
