"""
Module for adding arguments and their default values to a config.
"""
from typing import Dict, Union, Any, get_args, Iterable

from copy import deepcopy
import inspect
import logging

from allennlp.common import Registrable, Params, Lazy, FromParams

logger = logging.getLogger(__name__)


def fill_config_with_default_values(
    base_class: Union[FromParams, Registrable], config: Union[Dict, Params]
) -> Dict:
    """
    Fill a `config` with the arguments and their default values from a
    `base_class`. When it encounters nested objects (i.e. non-builtin classes,
    for now limited to AllenNLP `FromParams` and `Registrable`) it will recurse
    and create the dictionary with defaults.

    This does NOT change the mutable config nor does it override keys if they
    exist in the config.

    **NOTE:** When AllenNLP expands upon their
    [`to_params`](https://docs.allennlp.org/main/api/common/from_params/#to_params)
    api, this will rely on that, but for the time being it relies on the
    [`inspect`](https://docs.python.org/3/library/inspect.html) module.

    # Parameters

    base_class: `Union[FromParams, Registrable]`
        The base class that you want to use to fill the `config` with the
        arguments and their default values. These arguments are for the
        `base_class.__init__` function.

    config: `Union[Dict, Params]`
        The configuration dict to fill with the defaults. It does **NOT** make
        any changes to the mutable data but instead uses `deepcopy` to copy it
        to a new object.

        Existing keys will not be overwritten.

    # Returns

    cfg_with_defaults: `Dict` The filled config with defaults.
    """
    # Copy to avoid mutable changes
    if isinstance(config, Params):
        config_dict = config.params
    else:
        config_dict = config

    # Check if we need to use the default or not.
    if issubclass(base_class, Registrable):
        class_to_use = base_class.by_name(
            config_dict.get("type", base_class.default_implementation)
        )
    else:
        class_to_use = base_class

    # Get the init parameters from the underlying class of `initialized_class`
    init_params = inspect.signature(class_to_use)

    output_config = deepcopy(config_dict)
    for parameter in init_params.parameters.values():
        # Positional parameters do not have default values, so we skip them.
        if parameter.default == inspect.Parameter.empty:
            continue

        # The parameter is a keyword and has a default value.
        if parameter.name not in config_dict:
            output_config[parameter.name] = get_default_value_for_parameter(parameter)
            continue

        if parameter.annotation is None:
            raise ValueError(
                f"Argument {parameter.name} in class"
                f" {base_class.__name__} does not have a type"
                f" annotation"
            )

        # If it is not a dict, then it is not a class constructor argument and
        # thus no need to recurse.
        if not isinstance(config_dict[parameter.name], dict):
            continue

        output_config[parameter.name] = dict(
            fill_config_with_default_values(
                get_annotation_class(parameter), Params(config_dict[parameter.name])
            )
        )

    return output_config


def get_default_value_for_parameter(parameter: inspect.Parameter) -> Any:
    """
    Get the default value for a parameter.

    If there is a type annotation, check if it is a subclass of Registrable. If
    that is the case, check if the default value is an instance of that class.
    If it is, then recurse. Otherwise continue. Handles the case where the
    default value is an initialized registrable. We also find what name the
    class was registered as to add that as the `type` value.

    # Parameters

        parameter: `inspect.Parameter`
            The parameter to get the default value of.

    # Returns

    `Any` The default value of the parameter.
    """
    if parameter.annotation is not None:
        # Annotation found, check if it is a Registrable.

        annotation_type = get_annotation_class(parameter)
        if annotation_type is None:
            return parameter.default

        try:
            is_subclass_of_registrable = issubclass(annotation_type, Registrable)
        except TypeError:
            # If there is some typing error, return the default value and call
            # it a day.
            return parameter.default

        cfg_dict = None
        if is_subclass_of_registrable and isinstance(
            parameter.default, annotation_type
        ):
            # Reverse lookup what the class was registered as. We do need to
            # access the protected `_registry` attribute of registrable, but
            # there is currently no other way to do this.
            parameter_default_class = parameter.default.__class__
            registered_name = None
            if annotation_type not in Registrable._registry:
                raise KeyError(
                    f"'{parameter.annotation.__name__}' has no" f" registered classes"
                )

            for name, registered_class in Registrable._registry[
                parameter.annotation
            ].items():
                if registered_class[0] == parameter_default_class:
                    registered_name = name
                    break

            if registered_name is None:
                raise KeyError(
                    f"'{parameter_default_class.__name__}' was never" f" registered."
                )

            cfg_dict = {"type": registered_name}
        elif is_subclass_of_registrable and isinstance(parameter.default, Lazy):
            logger.warning(
                f"{parameter.name} has a Lazy object for its "
                f"default. That is not currently supported and will"
                f" be handled as getting the arguments for the"
                f" default annotation."
            )

            cfg_dict = {"type": annotation_type.default_implementation}
        elif not is_subclass_of_registrable and issubclass(annotation_type, FromParams):
            cfg_dict = {}

        if cfg_dict is not None:
            return fill_config_with_default_values(annotation_type, cfg_dict)

    # No annotation found or it is not a registrable, so return the default
    # value.
    return parameter.default


def get_annotation_class(parameter: inspect.Parameter) -> Union[type, None]:
    """
    Get the class of the annotation.

    # Parameters

        parameter: `inspect.Parameter`
            The parameter to get the class of.

    # Returns

    `Union[type, None]` The type or None.

    """
    # If the type is a `_GenericAlias`, we care what the arguments are inside.
    # Thus, we get the annotation args. If it is empty, it is not a
    # `_GenericAlias` and we can return the value.
    annotation_args = get_args(parameter.annotation)
    if not annotation_args:
        return parameter.annotation

    # If there are more than one argument, we return None because it is easier
    # to handle.
    if len(annotation_args) > 1:
        return None

    # Otherwise return the found type.
    return annotation_args[0]


def get_positional_arguments(cls_type: FromParams) -> Iterable[str]:
    # Get the init parameters from the underlying class of `initialized_class`
    init_params = inspect.signature(cls_type)

    for parameter in init_params.parameters.values():
        if (
            parameter.default == inspect.Parameter.empty
            and parameter.kind != inspect.Parameter.VAR_KEYWORD
        ):
            yield parameter.name
