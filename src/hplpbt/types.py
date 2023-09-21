# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Dict, Iterable, Mapping, Set

from enum import Enum

from attrs import field, frozen
from attrs.validators import deep_iterable, deep_mapping, instance_of, matches_re
from hpl.ast import HplPredicate, HplVacuousTruth
from hpl.parser import condition_parser
# from hpl.types import TypeToken
from typeguard import check_type, typechecked

################################################################################
# Type Definition
################################################################################


class BuiltinParameterType(Enum):
    BOOL = 'bool'
    UINT = 'uint'
    UINT8 = 'uint8'
    UINT16 = 'uint16'
    UINT32 = 'uint32'
    UINT64 = 'uint64'
    INT = 'int'
    INT8 = 'int8'
    INT16 = 'int16'
    INT32 = 'int32'
    INT64 = 'int64'
    FLOAT = 'float'
    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    STRING = 'string'


@frozen
class ParameterDefinition:
    base_type: str
    is_array: bool = False

    @property
    def is_builtin(self) -> bool:
        try:
            BuiltinParameterType(self.base_type)
            return True
        except ValueError:
            return False

    @classmethod
    def from_type_string(cls, type_string: str) -> 'ParameterDefinition':
        parts = type_string.split('[', maxsplit=1)
        base_type: str = parts[0]
        is_array: bool = len(parts) > 1
        return cls(base_type, is_array=is_array)


@frozen
class MessageType:
    name: str = field(validator=[instance_of(str), matches_re(r'\w+')])
    package: str = field(default='', validator=[instance_of(str), matches_re(r'\w*')])
    class_name: str = field(default='', validator=[instance_of(str), matches_re(r'\w*')])
    positional_parameters: Iterable[ParameterDefinition] = field(
        factory=list,
        validator=deep_iterable(
            instance_of(ParameterDefinition),
            iterable_validator=instance_of(Iterable),
        ),
    )
    keyword_parameters: Mapping[str, ParameterDefinition] = field(
        factory=dict,
        validator=deep_mapping(
            instance_of(str),
            instance_of(ParameterDefinition),
            mapping_validator=instance_of(Mapping),
        ),
    )
    precondition: HplPredicate = field(
        factory=HplVacuousTruth,
        validator=instance_of(HplPredicate),
    )

    def __attrs_post_init__(self):
        if not self.class_name:
            object.__setattr__(self, 'class_name', self.name)

    @property
    def qualified_name(self) -> str:
        return self.class_name if not self.package else f'{self.package}.{self.class_name}'

    @classmethod
    @typechecked
    def from_data(cls, name: str, data: Mapping[str, Any]) -> 'MessageType':
        dependency = check_type(data.get('import', ''), str).split('.', maxsplit=1)
        package = dependency[0]
        class_name = dependency[1] if len(dependency) > 1 else ''
        arg_data = check_type(data.get('args', ()), Iterable[str])
        params = list(map(ParameterDefinition.from_type_string, arg_data))
        kwarg_data = check_type(data.get('kwargs', {}), Mapping[str, str])
        kwparams = {
            key: ParameterDefinition.from_type_string(value)
            for key, value in kwarg_data.items()
        }
        precondition_data = check_type(data.get('assume', ()), Iterable[str])
        parser = condition_parser()
        predicate = HplVacuousTruth()
        for expr in precondition_data:
            predicate = predicate.join(parser.parse(expr))
        return cls(
            name,
            package=package,
            class_name=class_name,
            positional_parameters=params,
            keyword_parameters=kwparams,
            precondition=predicate,
        )

    def dependencies(self) -> Set[str]:
        deps = {p.base_type for p in self.positional_parameters if not p.is_builtin}
        deps.update(p.base_type for p in self.keyword_parameters.values() if not p.is_builtin)
        return deps

    def __str__(self) -> str:
        return f'{self.name} ({self.qualified_name})'


################################################################################
# Interface
################################################################################


def message_from_data(name: str, data: Mapping[str, Any]) -> MessageType:
    return MessageType.from_data(name, data)


@typechecked
def type_map_from_data(data: Mapping[str, Mapping[str, Any]]) -> Dict[str, MessageType]:
    return {name: message_from_data(name, type_def) for name, type_def in data.items()}
