# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Final, Iterable, List, Mapping, Optional, Tuple, Union

from pathlib import Path

from hpl.ast import HplProperty, HplSpecification
from hpl.parser import property_parser, specification_parser
from hpl.rewrite import canonical_form

from hplpbt.logic import split_assumptions
from hplpbt.strategies.messages import strategies_from_spec
from hplpbt.types import type_map_from_data

###############################################################################
# Constants
###############################################################################

MSG_TYPES_KEY_CHANNELS: Final[str] = 'messages'
MSG_TYPES_KEY_TYPEDEFS: Final[str] = 'data'

###############################################################################
# Interface
###############################################################################


def generate_tests_from_files(
    paths: Iterable[Union[Path, str]],
    msg_types: Mapping[str, Mapping[str, Any]],
) -> str:
    """
    Produces test code snippets,
    given a list of paths to HPL files with specifications.
    """
    parser = specification_parser()
    properties: List[HplProperty] = []
    for input_path in paths:
        path: Path = Path(input_path).resolve(strict=True)
        text: str = path.read_text(encoding='utf-8').strip()
        spec: HplSpecification = parser.parse(text)
        properties.extend(spec.properties)
    return generate_tests(properties, msg_types)


def generate_tests_from_spec(
    spec: Union[HplSpecification, str],
    msg_types: Mapping[str, Mapping[str, Any]],
) -> str:
    """
    Produces test code snippets, given an HPL specification.
    """
    if not isinstance(spec, HplSpecification):
        parser = specification_parser()
        spec = parser.parse(spec)
    return generate_tests(spec.properties, msg_types)


def generate_tests(
    input_properties: Iterable[Union[str, HplProperty]],
    msg_types: Mapping[str, Mapping[str, Any]],
) -> str:
    _validate_msg_types(msg_types)
    input_properties = _parsed_properties(input_properties)
    input_channels = msg_types[MSG_TYPES_KEY_CHANNELS]
    canonical_properties = [p for ps in map(canonical_form, input_properties) for p in ps]
    assumptions, behaviour = split_assumptions(canonical_properties, input_channels)
    type_defs = type_map_from_data(msg_types[MSG_TYPES_KEY_TYPEDEFS])
    msg_strategies = strategies_from_spec(behaviour, input_channels, type_defs)
    parts = ['# assumptions']
    parts.extend(map(repr, map(str, assumptions)))
    parts.append('# behaviour')
    parts.extend(map(repr, map(str, behaviour)))
    parts.append('# strategies')
    parts.extend(map(repr, map(str, msg_strategies)))
    return '\n'.join(parts)


###############################################################################
# Helper Functions
###############################################################################


def _parsed_properties(properties: Iterable[Union[str, HplProperty]]) -> List[HplProperty]:
    parser = property_parser()
    return [p if isinstance(p, HplProperty) else parser.parse(p) for p in properties]


###############################################################################
# Input Validation
###############################################################################


def _validate_msg_types(msg_types: Mapping[str, Mapping[str, Any]]):
    if not isinstance(msg_types, Mapping):
        raise TypeError(f'expected Mapping[str, Mapping[str, Any]], got {msg_types!r}')
    typedefs = set()
    data: Mapping[str, Mapping[str, Any]] = msg_types[MSG_TYPES_KEY_TYPEDEFS]
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError(f"expected str keys in '{MSG_TYPES_KEY_TYPEDEFS}', found {key!r}")
        if not isinstance(value, Mapping):
            raise TypeError(
                f"expected Mapping[str, Any] values for each '{MSG_TYPES_KEY_TYPEDEFS}' key,"
                f' found {value!r}'
            )
        _validate_type_def(key, value)
        typedefs.add(key)
    messages: Mapping[str, str] = msg_types[MSG_TYPES_KEY_CHANNELS]
    for key, value in messages.items():
        if not isinstance(key, str):
            raise TypeError(f"expected str keys in '{MSG_TYPES_KEY_CHANNELS}', found {key!r}")
        if not isinstance(value, str):
            raise TypeError(
                f"expected str values for each '{MSG_TYPES_KEY_CHANNELS}' key,"
                f" found {value!r}"
            )
        if value not in typedefs:
            raise ValueError(f"unknown message type {value!r}")


def _validate_type_def(name: str, type_def: Mapping[str, Any]):
    module: Optional[str] = type_def.get('import')
    if module is not None and not isinstance(module, str):
        raise TypeError(f"expected str 'import' value for {name}, found {module!r}")
    _validate_type_pos_args(name, type_def.get('args', ()))
    _validate_type_kwargs(name, type_def.get('kwargs', {}))


def _validate_type_pos_args(name: str, args: Iterable[str]):
    if not isinstance(args, Iterable):
        raise TypeError(f"expected Iterable 'args' value for {name}, found {args!r}")
    for item in args:
        if not isinstance(item, str):
            raise TypeError(f"expected str 'args' entry for {name}, found {item!r}")


def _validate_type_kwargs(name: str, kwargs: Mapping[str, str]):
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"expected Mapping 'kwargs' value for {name}, found {kwargs!r}")
    for key, value in kwargs.items():
        if not isinstance(key, str):
            raise TypeError(f"expected str 'kwargs' key for {name}, found {key!r}")
        if not isinstance(value, str):
            raise TypeError(f"expected str 'kwargs' value for {name}, found {value!r}")
