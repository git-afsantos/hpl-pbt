# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Iterable

from attrs import field, frozen

################################################################################
# Internal Structures: Traces and Segments
################################################################################


@frozen
class TraceSegment:
    delay: float
    timeout: float
    mandatory: str
    spam: Iterable[str] = field(factory=tuple, converter=tuple)


@frozen
class Trace:
    segments: Iterable[TraceSegment] = field(factory=tuple, converter=tuple)
