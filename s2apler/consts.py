import numpy as np
import s2apler
import os
import json
import logging

logger = logging.getLogger("s2apler")

package_dir = os.path.abspath(s2apler.__path__[0])
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(package_dir, os.pardir))

# load up the path_configs and check if they are set
CONFIG_LOCATION = os.path.join(PROJECT_ROOT_PATH, "data", "path_config.json")

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT_PATH, "data")
UNSET_DATA_DIR = "absolute path of wherever you downloaded the data to"

# An installed wheel has no data/path_config.json: the data directory sits next to the
# package only in a checkout. Import must not create or require it, because installs can be
# read-only and inference (TIMO) passes its own artifacts directory.
CONFIG: dict = {}
if os.path.exists(CONFIG_LOCATION):
    with open(CONFIG_LOCATION) as _json_file:
        CONFIG = json.load(_json_file)

if CONFIG.get("main_data_dir", UNSET_DATA_DIR) == UNSET_DATA_DIR:
    if os.path.exists(CONFIG_LOCATION):
        logger.warning(
            "You haven't set `main_data_dir` in data/path_config.json! Using data/ as default data directory."
        )
    CONFIG["main_data_dir"] = DEFAULT_DATA_DIR
elif not os.path.exists(CONFIG["main_data_dir"]):
    # Only an explicitly configured directory must exist; catch typos at import.
    raise FileNotFoundError(f"The `main_data_dir` in data/path_config.json doesn't exist: {CONFIG['main_data_dir']}")

# important constant values
NUMPY_NAN = np.nan
DEFAULT_CHUNK_SIZE = 100
LARGE_DISTANCE = 1e4
LARGE_INTEGER = 10 * LARGE_DISTANCE
CLUSTER_SEEDS_LOOKUP = {"require": 0, "disallow": LARGE_DISTANCE}
# this is the key for the orphan cluster, which are papers that do not
# belong to any known cluster but may or may not cluster with each other
ORPHAN_CLUSTER_KEY = "orphans"
