from copy import deepcopy
import inspect
import logging

from allennlp.common import Registrable, Params

logger = logging.getLogger(__name__)


def fill_config_with_default_values(base_class: Registrable, config: Params) -> Params:
    # Copy to avoid mutable changes
    initialized_class = base_class.from_params(Params(deepcopy(config.params)))

    # Get the init parameters from the underlying class of `initialized_class`
    init_params = inspect.signature(initialized_class.__class__)

    output_config = dict(deepcopy(config.params))

    for parameter in init_params.parameters.values():
        # Positional parameters do not have default values, so we skip them.
        if parameter.default == inspect.Parameter.empty:
            continue

        # The parameter is a keyword and has a default value.
        if parameter.name not in config:
            output_config[parameter.name] = parameter.default
            continue

        if parameter.annotation is None:
            raise ValueError(f"Argument {parameter.name} in class"
                             f" {base_class.__name__} does not have a type"
                             f" annotation")

        # If it is not a dict, then it is not a class constructor argument and
        # thus no need to recurse.
        if not isinstance(config.params[parameter.name], dict):
            continue

        output_config[parameter.name] = dict(fill_config_with_default_values(
            parameter.annotation,
            Params(config.params[parameter.name])
        ).params)

    return Params(output_config)
