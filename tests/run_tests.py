"""Pytest test suite entrypoint (discover + run tests in tests/)."""
import os
import subprocess
import sys

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "-m", "pytest", os.path.join(root, "tests"), "-v"])