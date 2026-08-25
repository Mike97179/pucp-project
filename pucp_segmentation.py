#!/usr/bin/env python3
"""
Entry point of the structural element segmentation software.

    python pucp_segmentation.py --help
    python pucp_segmentation.py modelos
    python pucp_segmentation.py entrenar --imgsz 800 --oversample 3
"""

import sys

from structures.cli import main

if __name__ == '__main__':
    sys.exit(main())
