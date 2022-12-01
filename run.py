import logging
import subprocess
import sys
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
RESULT_ROOT = Path(__file__).resolve().parent / "results"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("180A-Submission")


def main(args):
    if "all" in args:
        logger.info("Running all experiments")
        configs = CONFIG_ROOT.rglob("*.py")
    elif "etc" in args:
        logger.info("Running ETC policy experiments.")
        configs = (CONFIG_ROOT / "etc").rglob("*.py")
    elif "lints" in args:
        logger.info("Running Linear TS experiments.")
        configs = (CONFIG_ROOT / "lints").rglob("*.py")
    elif "linucb" in args:
        logger.info("Running LinearUCB experiments.")
        configs = (CONFIG_ROOT / "linucb").rglob("*.py")
    elif "optimal" in args:
        logger.info("Running Bayesian Optimal Policy experiments.")
        configs = (CONFIG_ROOT / "optimal").rglob("*.py")
    elif "ts" in args:
        logger.info("Running TS experiments.")
        configs = (CONFIG_ROOT / "ts").rglob("*.py")
    elif "ucb" in args:
        logger.info("Running UCB experiments.")
        configs = (CONFIG_ROOT / "ucb").rglob("*.py")

    for f in configs:
        logger.info(f"Running {f}")
        subprocess.call(
            args=[
                "python",
                str(f),
                str(RESULT_ROOT / f.parent.name / f"{f.name[:-2]}pkl"),
            ]
        )


if __name__ == "__main__":
    main(sys.argv[1:])
