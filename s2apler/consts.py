import numpy as np
from pathlib import Path
import s2apler
import os
import json
import logging

logger = logging.getLogger("s2apler")

package_dir = os.path.abspath(s2apler.__path__[0])
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(package_dir, os.pardir))

# load up the path_configs and check if they are set
CONFIG_LOCATION = os.path.join(PROJECT_ROOT_PATH, "data", "path_config.json")
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

if CONFIG["main_data_dir"] == "absolute path of wherever you downloaded the data to":
    logger.warning("You haven't set `main_data_dir` in data/path_config.json! Using data/ as default data directory.")
    CONFIG["main_data_dir"] = os.path.join(PROJECT_ROOT_PATH, "data")

assert os.path.exists(CONFIG["main_data_dir"]), "The `main_data_dir` specified in data/path_config.json doesn't exist."

# feature caching related consts
CACHE_ROOT = Path(os.getenv("S2APLER_CACHE", str(Path.home() / ".s2apler")))
FEATURIZER_VERSION = 1

# important constant values
NUMPY_NAN = np.nan
DEFAULT_CHUNK_SIZE = 100
LARGE_DISTANCE = 1e4
LARGE_INTEGER = 10 * LARGE_DISTANCE
CLUSTER_SEEDS_LOOKUP = {"require": 0, "disallow": LARGE_DISTANCE}
# this is the key for the orphan cluster, which are papers that do not
# belong to any known cluster but may or may not cluster with each other
ORPHAN_CLUSTER_KEY = "orphans"
