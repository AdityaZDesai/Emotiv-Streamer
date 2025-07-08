#!/usr/bin/env python3
# encoding: utf-8

from setuptools import setup, find_packages

# This setup.py is kept for backward compatibility and data file handling
# Modern installations should use pyproject.toml
if __name__ == "__main__":
    setup(
        packages=find_packages(),
        data_files=[('/etc/udev/rules.d', ['udev/99-emotiv-epoc.rules'])],
    )
