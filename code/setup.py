"""
setup.py
"""

from setuptools import setup, find_packages
from typing import Dict
import os


NAME = "pyspecks"
AUTHOR = "Alexander Lin"
EMAIL = "alin5250@gmail.com"
DESCRIPTION = "Python package for spectral k-segmentation."


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def required():
    with open('requirements.txt') as f:
        return f.read().splitlines()


VERSION: Dict[str, str] = {}
with open("pyspecks/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)


setup(

    name=NAME,
    version=os.environ.get("TAG_VERSION", VERSION['VERSION']),

    description=DESCRIPTION,

    # Author information
    author=AUTHOR,
    author_email=EMAIL,

    # What is packaged here.
    packages=find_packages(),

    install_requires=required(),
    include_package_data=True,

    python_requires='>=3.6.1',
    zip_safe=True

)
