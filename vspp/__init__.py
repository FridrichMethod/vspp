import logging
import multiprocessing as mp
import os

from pandarallel import pandarallel

# Set the number of threads for NumExpr
os.environ["NUMEXPR_MAX_THREADS"] = str(mp.cpu_count() // 2)

# Set the logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Initialize pandarallel for parallel pandas processing
pandarallel.initialize(progress_bar=True)
