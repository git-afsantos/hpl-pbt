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
    name: str = ''

    @property
    def is_array(self) -> bool:
        return self.type.endswith(']')


@frozen
class MessageType:
    name: str
    package: Optional[str] = None
    parameters: List[ParameterDefinition] = field(factory=list)

    def __str__(self) -> str:
        return self.name if self.package is None else f'{self.package}.{self.name}'


################################################################################
# Interface
################################################################################


@typechecked
def param_from_data(data: Union[str, Mapping[str, Any]]) -> ParameterDefinition:
    if isinstance(data, str):
        data = {'type': data}
    name: str = data.get('name', '')
    type_string: str = check_type(data['type'], str)
    return ParameterDefinition(type_string, name=name)


@typechecked
def message_from_data(name: str, data: Mapping[str, Any]) -> MessageType:
    package = check_type(data.get('import'), Optional[str])
    param_data = check_type(data['params'], Iterable[Union[str, Mapping[str, Any]]])
    params: List[ParameterDefinition] = list(map(param_from_data, param_data))
    return MessageType(name, package=package, parameters=params)


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