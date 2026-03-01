import os
from pathlib import Path

from setuptools import find_packages
from conda_setup import setup

if __name__ == "__main__":
    requirements = Path(__file__).parent / "requirements.yml"
    os.system(f"conda env update --file {requirements}")
    setup(
        name="binpy",
        packages=find_packages(),
        scripts=[],
        entry_points={},
        requirements_yml='requirements.yml',
    )
