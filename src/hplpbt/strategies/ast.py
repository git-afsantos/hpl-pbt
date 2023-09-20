# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Iterable, Set

from enum import auto, Enum

from attrs import field, frozen
from attrs.validators import deep_iterable, instance_of
from hpl.ast import HplExpression

################################################################################
# Strategy AST - Value Expressions
################################################################################


class ExpressionType(Enum):
    LITERAL = auto()
    REFERENCE = auto()


@frozen
class Expression:
    """
    Base class that represents concrete values in the AST, for example, values
    given as function arguments and values used within other expressions.
    """

    @property
    def type(self) -> ExpressionType:
        raise NotImplementedError()

    @property
    def is_literal(self) -> bool:
        return self.type == ExpressionType.LITERAL

    @property
    def is_reference(self) -> bool:
        return self.type == ExpressionType.REFERENCE

    def references(self) -> Set[str]:
        raise NotImplementedError()


@frozen
class Literal(Expression):
    value: Any

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.LITERAL

    @property
    def is_none(self) -> bool:
        return self.value is None

    @property
    def is_bool(self) -> bool:
        return isinstance(self.value, bool)

    @property
    def is_int(self) -> bool:
        return isinstance(self.value, int)

    @property
    def is_float(self) -> bool:
        return isinstance(self.value, float)

    @property
    def is_string(self) -> bool:
        return isinstance(self.value, str)

    @property
    def is_array(self) -> bool:
        return isinstance(self.value, (list, tuple))

    @classmethod
    def none(cls) -> 'Literal':
        return cls(None)

    @classmethod
    def true(cls) -> 'Literal':
        return cls(True)

    @classmethod
    def false(cls) -> 'Literal':
        return cls(False)

    @classmethod
    def zero(cls) -> 'Literal':
        return cls(0)

    @classmethod
    def empty_string(cls) -> 'Literal':
        return cls('')

    @classmethod
    def empty_list(cls) -> 'Literal':
        return cls([])

    def references(self) -> Set[str]:
        return set()

    def __str__(self) -> str:
        return repr(self.value)


@frozen
class Reference(Expression):
    variable: str

    @property
    def type(self) -> ExpressionType:
        return ExpressionType.REFERENCE

    def references(self) -> Set[str]:
        return {self.variable}

    def __str__(self) -> str:
        return self.variable


################################################################################
# Strategy AST - Value Generators
################################################################################


class GeneratorType(Enum):
    CONSTANT = auto()
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    ARRAY = auto()
    SAMPLE = auto()


@frozen
class ValueGenerator:
    """
    Base class that represents calls to strategies within a function's body,
    for example, to generate arguments and other variables.
    """

    @property
    def type(self) -> GeneratorType:
        raise NotImplementedError()

    @property
    def is_constant(self) -> bool:
        return self.type == GeneratorType.CONSTANT

    @property
    def is_bool(self) -> bool:
        return self.type == GeneratorType.BOOL

    @property
    def is_int(self) -> bool:
        return self.type == GeneratorType.INT

    @property
    def is_float(self) -> bool:
        return self.type == GeneratorType.FLOAT

    @property
    def is_string(self) -> bool:
        return self.type == GeneratorType.STRING

    @property
    def is_array(self) -> bool:
        return self.type == GeneratorType.ARRAY

    @property
    def is_sample(self) -> bool:
        return self.type == GeneratorType.SAMPLE

    def dependencies(self) -> Set[str]:
        return set()


@frozen
class ConstantValue(ValueGenerator):
    expression: Expression = field(validator=instance_of(Expression))

    @property
    def type(self) -> GeneratorType:
        return GeneratorType.CONSTANT

    @property
    def value(self) -> Expression:
        return self.expression  # alias

    def dependencies(self) -> Set[str]:
        return self.expression.references()

    def __str__(self) -> str:
        return f'just({self.expression})'


@frozen
class RandomBool(ValueGenerator):
    @property
    def type(self) -> GeneratorType:
        return GeneratorType.BOOL

    def dependencies(self) -> Set[str]:
        return set()

    def __str__(self) -> str:
        return 'booleans()'


@frozen
class RandomInt(ValueGenerator):
    min_value: Expression = field(factory=Literal.none, validator=instance_of(Expression))
    max_value: Expression = field(factory=Literal.none, validator=instance_of(Expression))

    @property
    def type(self) -> GeneratorType:
        return GeneratorType.INT

    def dependencies(self) -> Set[str]:
        return self.min_value.references() | self.max_value.references()

    def __str__(self) -> str:
        return f'integers(min_value={self.min_value}, max_value={self.max_value})'


@frozen
class RandomFloat(ValueGenerator):
    min_value: Expression = field(factory=Literal.none, validator=instance_of(Expression))
    max_value: Expression = field(factory=Literal.none, validator=instance_of(Expression))
    # allow_nan: bool = True
    # allow_infinity: bool = True
    # allow_subnormal: bool = True
    # width: int = 64
    # exclude_min: bool = False
    # exclude_max: bool = False

    @property
    def type(self) -> GeneratorType:
        return GeneratorType.FLOAT

    def dependencies(self) -> Set[str]:
        return self.min_value.references() | self.max_value.references()

    def __str__(self) -> str:
        # min_value=None
        # max_value=None
        # allow_nan=None
        # allow_infinity=None
        # allow_subnormal=None
        # width=64
        # exclude_min=False
        # exclude_max=False
        return f'floats(min_value={self.min_value}, max_value={self.max_value})'


@frozen
class RandomString(ValueGenerator):
    min_size: Expression = field(factory=Literal.zero, validator=instance_of(Expression))
    max_size: Expression = field(factory=Literal.none, validator=instance_of(Expression))
    # alphabet: Optional[Expression] = None

    @property
    def type(self) -> GeneratorType:
        return GeneratorType.STRING

    def dependencies(self) -> Set[str]:
        return self.min_size.references() | self.max_size.references()

    def __str__(self) -> str:
        # alphabet=characters(codec='utf-8')
        # min_size=0
        # max_size=None
        return f'text(min_size={self.min_size}, max_size={self.max_size})'


@frozen
class RandomArray(ValueGenerator):
    elements: ValueGenerator = field(validator=instance_of(ValueGenerator))
    min_size: Expression = field(factory=Literal.zero, validator=instance_of(Expression))
    max_size: Expression = field(factory=Literal.none, validator=instance_of(Expression))
    unique: bool = False

    @property
    def type(self) -> GeneratorType:
        return GeneratorType.ARRAY

    def dependencies(self) -> Set[str]:
        return self.min_size.references() | self.max_size.references()

    def __str__(self) -> str:
        # min_size=0
        # max_size=None
        # unique_by=None
        # unique=False
        return (
            f'lists({self.elements}'
            f', min_size={self.min_size}'
            f', max_size={self.max_size}'
            f', unique={self.unique})'
        )


@frozen
class RandomSample(ValueGenerator):
    elements: Iterable[Expression] = field(
        factory=tuple,
        converter=tuple,
        validator=deep_iterable(instance_of(Expression)),
    )

    @property
    def type(self) -> GeneratorType:
        return GeneratorType.SAMPLE

    def dependencies(self) -> Set[str]:
        names = set()
        for expresion in self.elements:
            names.update(expresion.references())
        return names

    def __str__(self) -> str:
        return f'sampled_from({self.elements})'


################################################################################
# Strategy AST - Statements
################################################################################


class StatementType(Enum):
    ASSIGN = auto()
    ASSUME = auto()
    LOOP = auto()
    BLOCK = auto()


@frozen
class Statement:
    @property
    def type(self) -> StatementType:
        raise NotImplementedError()

    @property
    def is_assignment(self) -> bool:
        return self.type == StatementType.ASSIGN

    @property
    def is_assumption(self) -> bool:
        return self.type == StatementType.ASSUME

    @property
    def is_loop(self) -> bool:
        return self.type == StatementType.LOOP

    @property
    def is_block(self) -> bool:
        return self.type == StatementType.BLOCK

    def merge(self, other: 'Statement') -> 'Statement':
        c1 = type(self).__name__
        c2 = type(other).__name__
        raise TypeError(f'unable to merge {c1} with {c2}')


################################################################################
# Strategy AST - Assignment
################################################################################


@frozen
class Assignment(Statement):
    variable: str
    expression: HplExpression

    @property
    def type(self) -> StatementType:
        return StatementType.ASSIGN

    def __str__(self) -> str:
        return f'{self.variable} = {self.expression}'


################################################################################
# Strategy AST - Assumption
################################################################################


@frozen
class Assumption(Statement):
    expression: HplExpression

    @property
    def type(self) -> StatementType:
        return StatementType.ASSUME

    def __str__(self) -> str:
        return f'assume({self.expression})'
