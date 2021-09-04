"""
Command to convert a class to a yaml. Registered as `cls2yml`
"""
import sys
from typing import Dict

import argparse
import logging
from pathlib import Path

from allennlp.common import Registrable
from allennlp.commands.subcommand import Subcommand
from overrides import overrides
import re
import yaml

from allennlp_hydra.config.fill_defaults import (
    fill_config_with_default_values,
    get_positional_arguments,
)

logger = logging.getLogger(__name__)


@Subcommand.register("class2yaml")
class ClassToYaml(Subcommand):
    @overrides
    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """Convert a registered class to its YAML representation"""
        subparser = parser.add_parser(
            self.name, description=description, help=description
        )

        subparser.add_argument(
            "cls_name",
            type=str,
            help="Name the class was registered with. This"
            " is will be used as the name of the "
            "created `.yaml` file.",
        )

        subparser.add_argument(
            "base_cls_name", type=str, help="Name of the Base Class."
        )

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="Directory to save the file to. The name of the file will "
            "be `{cls_name}.json`. All non alphanumeric and `_` characters "
            "will be replaced with `_`.",
        )

        subparser.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Force overwriting of existing `.yaml` file.",
        )

        subparser.set_defaults(func=class_to_yaml_from_args)

        return subparser


def class_to_yaml_from_args(args: argparse.Namespace) -> Dict:
    return class_to_yaml(
        cls_name=args.cls_name,
        base_cls_name=args.base_cls_name,
        serialization_dir=args.serialization_dir,
        force=args.force,
    )


def class_to_yaml(
    cls_name: str, base_cls_name: str, serialization_dir: str, force: bool = False
) -> Dict:
    print(f"Looking for the '{base_cls_name}' registered as {cls_name}")
    base_class = None

    # Go through the registered classes and find the `base_cls_name`
    for base_class_object in Registrable.__subclasses__():
        if base_class_object.__name__ == base_cls_name:
            base_class = base_class_object
            break

    if base_class is None:
        raise ValueError(
            f"The class '{base_cls_name}' could not be found as a"
            f" subclass of Registrable."
        )

    try:
        cls_to_get = base_class.by_name(cls_name)
    except KeyError:
        raise KeyError(f"'{cls_name}' is not registered as a {base_cls_name}")

    config_for_class = fill_config_with_default_values(base_class, {"type": cls_name})

    # Add the positional args to the config.
    for postional_arg in get_positional_arguments(cls_to_get):
        # '???' is the OmegaConf variable for required. Will throw an error if
        # not filled.
        config_for_class[postional_arg] = "???"

    serialization_dir = Path(serialization_dir)
    if not serialization_dir.exists():
        raise ValueError(f"{serialization_dir} does not exist")

    file_name = re.sub(r"[^\w\._]+", "_", cls_name)
    out_path = serialization_dir.joinpath(f"{file_name}.yaml")
    if out_path.exists() and not force:
        raise ValueError(f"{out_path} already exists. use --force to override.")

    yaml.dump(config_for_class, out_path.open("w", encoding="utf-8"), yaml.Dumper)

    print(f"Config for {base_cls_name}.{cls_name}:")
    print("")
    yaml.dump(config_for_class, sys.stdout, yaml.Dumper)

    return config_for_class
