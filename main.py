"""Openhouse AI Assistant — entry point."""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from app import run

if __name__ == "__main__":
    run()
