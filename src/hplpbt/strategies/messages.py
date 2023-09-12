# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Iterable, List, Union

from hpl.ast import HplProperty, HplSpecification

###############################################################################
# Interface
###############################################################################


class MessageStrategy:
    pass


def message_strategies_for_spec(
    spec: Union[HplSpecification, Iterable[HplProperty]]
) -> List[MessageStrategy]:
    if isinstance(spec, HplSpecification):
        spec = spec.properties
    return [strat for property in spec for strat in message_strategies_for_property(property)]


def message_strategies_for_property() -> List[MessageStrategy]:
    return []
