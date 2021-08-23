#!/usr/bin/env python

from codecs import open
import os
import setuptools
from setuptools.command.test import test as testing_cmd
import sys

here = os.path.abspath(os.path.dirname(__file__))
metadata = {}

with open(os.path.join(here, '__version__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), metadata)

with open('README.md', 'r', 'utf-8') as f:
    readme = f.read()

setuptools.setup(
    name=metadata['__title__'],
    version=metadata['__version__'],
    author=metadata['__author__'],
    author_email=metadata['__author_email__'],
    description=metadata['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    url=metadata['__url__'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    package_data={
        'ml_helpers': ['*.sh'],
        '': ['LICENSE']
    },
    install_requires=[
        'numpy>=1.11',
        'pandas>=1.0.0,!=1.0.4',
        'matplotlib',
    ],
)