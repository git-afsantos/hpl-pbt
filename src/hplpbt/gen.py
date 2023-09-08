# SPDX-License-Identifier: MIT
# Copyright © 2023 André Santos

###############################################################################
# Imports
###############################################################################

from typing import Any, Final, Iterable, List, Mapping, Optional, Tuple, Union

from pathlib import Path

from hpl.ast import HplProperty, HplSpecification
from hpl.parser import property_parser, specification_parser

from hplpbt.logic import split_assumptions

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
    properties: Iterable[Union[str, HplProperty]],
    msg_types: Mapping[str, Mapping[str, Any]],
) -> str:
    _validate_msg_types(msg_types)
    parser = property_parser()
    properties = [p if isinstance(p, HplProperty) else parser.parse(p) for p in properties]
    input_channels = msg_types[MSG_TYPES_KEY_CHANNELS]
    assumptions, behaviour = split_assumptions(properties, input_channels)
    return f'assumptions:\n{assumptions}\n\nbehaviour:\n{behaviour}'


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
    params: Iterable[Any] = type_def['params']
    if not isinstance(params, Iterable):
        raise TypeError(f"expected Iterable 'params' value for {name}, found {params!r}")
    for item in params:
        if isinstance(item, str):
            continue
        if isinstance(item, Mapping):
            _validate_param_constraints(name, item)
            continue
        raise TypeError(f"expected str or Mapping 'params' entry for {name}, found {item!r}")


def _validate_param_constraints(name: str, entry: Mapping[str, Any]):
    value = entry.get('type')
    if value is None:
        raise ValueError(f"parameter without a 'type' in {name}")
    if not isinstance(value, str):
        raise TypeError(f"expected str 'type' value for parameter of {name}, found {value!r}")
