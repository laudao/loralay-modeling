#!/usr/bin/env python3
from setuptools import find_packages, setup
setup(
    name="lla_summ",
    version="0.1.1",
    description="",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.6",
    extras_require={"dev": ["flake8", "isort", "black"]},
)
