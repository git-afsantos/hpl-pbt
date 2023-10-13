# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Exceptions
###############################################################################


class CyclicDependencyError(Exception):
    pass


class ContradictionError(Exception):
    pass
