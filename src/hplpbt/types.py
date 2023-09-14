# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from attrs import field, frozen
# from hpl.types import TypeToken
from typeguard import check_type, typechecked

################################################################################
# Type Definition
################################################################################


@frozen
class ParameterDefinition:
    type: str

    @property
    def is_array(self) -> bool:
        return '[' in self.type


@frozen
class MessageType:
    name: str
    package: str = ''
    positional_parameters: Iterable[ParameterDefinition] = field(factory=list)
    keyword_parameters: Mapping[str, ParameterDefinition] = field(factory=dict)

    def __str__(self) -> str:
        return self.name if self.package is None else f'{self.package}.{self.name}'


################################################################################
# Interface
################################################################################


@typechecked
def message_from_data(name: str, data: Mapping[str, Any]) -> MessageType:
    package = check_type(data.get('import', ''), str)
    arg_data = check_type(data.get('args', ()), Iterable[str])
    params = list(map(ParameterDefinition, arg_data))
    kwarg_data = check_type(data.get('kwargs', {}), Mapping[str, str])
    kwparams = {key: ParameterDefinition(value) for key, value in kwarg_data.items()}
    return MessageType(
        name,
        package=package,
        positional_parameters=params,
        keyword_parameters=kwparams,
    )


@typechecked
def type_map_from_data(data: Mapping[str, Mapping[str, Any]]) -> Dict[str, MessageType]:
    return {name: message_from_data(name, type_def) for name, type_def in data.items()}


def default_message_types() -> Dict[str, MessageType]:
    return {
        'bool': MessageType('bool'),
        'int': MessageType('int'),
        'float': MessageType('float'),
        'string': MessageType('string'),
    }
