# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, List, Optional

from enum import Enum, auto

from attrs import define, field
from hpl.ast import HplExpression
from hpl.types import TypeToken

################################################################################
# Internal Structures: Basic (Data) Field Generator
################################################################################


class FieldGeneratorState(Enum):
    PENDING = auto()  # waiting for the parent to be initialized
    READY = auto()  # the parent is available, this can be initialized
    INITIALIZED = auto()  # the field has a value, but there is more to do
    FINALIZED = auto()  # the field and all subfields are fully processed


@define
class FieldGenerator:
    """
    A FieldGenerator is composed of multiple statements (initialization,
    assumptions, ...) and each statement has its own dependencies (references
    to other local/external fields) so that they can be sorted individually.
    This maximizes flexibility, and results in code that is closer to what a
    human would write, knowing in advance what each statement needs.
    Internally, the FieldGenerator can be seen as a sort of state machine.
    When generating code, it changes its state as the requirements of each
    statement are satisfied and the statements are processed.
    """

    expression: HplExpression
    type_token: TypeToken
    parent: Optional['FieldGenerator']
    strategy: Any
    assumptions: List[Any] = field(factory=list)
    is_ranged: bool = False
    reference_count: int = field(default=0, init=False, eq=False)
    _ready_ref: Any = None
    _init_ref: Any = None
    _loop_context: Optional[Any] = None
