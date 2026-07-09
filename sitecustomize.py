# Auto-adds common/ to Python's import path so scripts in step folders can
# import the shared modules (reasoning_tools, puzzles, etc.) with no code changes.
# Python imports this file automatically at startup when run from this directory.
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "common"))
