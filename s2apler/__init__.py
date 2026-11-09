import logging
from importlib.metadata import version as package_version

__version__ = package_version("s2apler")

logger = logging.getLogger("s2apler")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)
