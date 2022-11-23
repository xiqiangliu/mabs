import os
import sys
from pathlib import Path

try:
    import mabs
except ImportError:
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", str(Path(__file__).parent)]
    )
    import mabs


def main(args):
    if "test" in args:
        print("This is a test.")


if __name__ == "__main__":
    main(sys.argv[1:])
