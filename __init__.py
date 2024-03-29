__version__ = "1.0.1"
__author__ = "Zhaoyang Li"
__email__ = "zhaoyangli@stanford.edu"

import os
import logging
import multiprocessing as mp

# Set the number of threads for NumExpr
os.environ["NUMEXPR_MAX_THREADS"] = str(mp.cpu_count() // 2)

# Set the logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
