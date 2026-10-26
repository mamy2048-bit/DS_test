"""Repo-root pytest config: make `dsc_control` (ROS package source) and `analysis` importable without colcon."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "dsc_control"))
sys.path.insert(0, ROOT)
