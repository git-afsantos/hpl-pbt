# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Dict, Iterable, Mapping

from attrs import field, frozen
from attrs.validators import deep_iterable, deep_mapping, instance_of, matches_re
from hpl.ast import HplPredicate, HplVacuousTruth
from hpl.parser import condition_parser
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
    name: str = field(validator=[instance_of(str), matches_re(r'\w+')])
    package: str = field(default='', validator=[instance_of(str), matches_re(r'\w*')])
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
    precondition_data = check_type(data.get('assume', ()), Iterable[str])
    parser = condition_parser()
    predicate = HplVacuousTruth()
    for expr in precondition_data:
        predicate = predicate.join(parser.parse(expr))
    return MessageType(
        name,
        package=package,
        positional_parameters=params,
        keyword_parameters=kwparams,
        precondition=predicate,
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
