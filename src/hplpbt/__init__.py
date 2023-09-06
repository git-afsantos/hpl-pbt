# SPDX-License-Identifier: MIT
# Copyright © 2021 André Santos

###############################################################################
# Imports
###############################################################################

from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

from hplpbt.gen import generate_tests

###############################################################################
# Constants
###############################################################################

try:
    __version__ = version('bakeapy')
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = 'unknown'
