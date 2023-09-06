# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

import hplpbt

###############################################################################
# Tests
###############################################################################


def test_import_was_ok():
    assert True


def test_pkg_has_version():
    assert hasattr(hplpbt, '__version__')
    assert isinstance(hplpbt.__version__, str)
    assert hplpbt.__version__ != ''
