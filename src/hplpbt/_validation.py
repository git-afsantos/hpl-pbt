
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
    try:
        _validate_type_pos_args(type_def.get('args', ()))
        _validate_type_kwargs(type_def.get('kwargs', {}))
    except TypeError as err:
        raise TypeError(f"in type definition '{name}': {err}") from err
    except ValueError as err:
        raise ValueError(f"in type definition '{name}': {err}") from err


def _validate_type_pos_args(args: Iterable[Union[str, Mapping[str, Any]]]):
    if not isinstance(args, Iterable):
        raise TypeError(f"expected Iterable 'args' value, found {args!r}")
    for param_type in args:
        try:
            _validate_param_type(param_type)
        except TypeError as err:
            raise TypeError(f"in 'args' entry: {err}") from err
        except ValueError as err:
            raise ValueError(f"in 'args' entry: {err}") from err


def _validate_type_kwargs(kwargs: Mapping[str, Union[str, Mapping[str, Any]]]):
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"expected Mapping 'kwargs' value, found {kwargs!r}")
    for name, param_type in kwargs.items():
        if not isinstance(name, str):
            raise TypeError(f"expected str 'kwargs' key, found {name!r}")
        try:
            _validate_param_type(param_type)
        except TypeError as err:
            raise TypeError(f"in 'kwargs' value for '{name}': {err}") from err
        except ValueError as err:
            raise ValueError(f"in 'kwargs' value for '{name}': {err}") from err


def _validate_param_type(param_type: Union[str, Mapping[str, Any]]):
    if isinstance(param_type, str):
        return
    if isinstance(param_type, Mapping):
        type_name = param_type.get('type')
        if type_name is None:
            raise ValueError(f"missing key 'type'")
        if not isinstance(type_name, str):
            raise TypeError(f"expected str for value of parameter 'type', found {type_name!r}")
        return
    raise TypeError(f"expected str|Mapping parameter type, found {param_type!r}")


###############################################################################
###############################################################################


@typechecked
def param_from_data(data: Union[str, Mapping[str, Any]]) -> ParameterDefinition:
    if isinstance(data, str):
        data = {'type': data}
    # name: str = data.get('name', '')
    type_string: str = check_type(data['type'], str)
    constraints = {key: value for key, value in data.items() if key != 'type'}
    return ParameterDefinition(type_string, constraints=constraints)
