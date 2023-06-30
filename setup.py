import re
import setuptools
from os import path

requirements_file = path.join(path.dirname(__file__), "requirements.in")
requirements = [r for r in open(requirements_file).read().split("\n") if not re.match(r"^\-", r)]

setuptools.setup(
    name="s2apler",
    version="0.1.3",
    description="S2APLER: Semantic Scholar (S2) Agglomeration of Papers with Low Error Rate",
    url="https://github.com/allenai/S2APLER",
    packages=setuptools.find_packages(),
    install_requires=requirements,  # dependencies specified in requirements.in
)
