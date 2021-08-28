from typing import Dict, Union

import argparse
import json
from os import PathLike
import logging
from pathlib import Path

from allennlp.commands.subcommand import Subcommand
import hydra
from omegaconf import OmegaConf
from overrides import overrides

logger = logging.getLogger(__name__)


@Subcommand.register("compose")
class ComposeConfig(Subcommand):
    @overrides
    def add_subparser(
            self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """Compose a config with Hydra"""
        subparser = parser.add_parser(
            self.name, description=description, help=description
        )

        subparser.add_argument(
            "config_path", type=str, help="Path to the config directory."
        )

        subparser.add_argument(
            "config_name", type=str, help="Name of the config file to use."
        )
        subparser.add_argument("job_name", type=str, help="Name of the job.")

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="Directory to save the config to. The name of the config will "
                 "be `{config_name}.json`",
        )
        subparser.set_defaults(func=compose_config_from_args)

        return subparser


def compose_config_from_args(args: argparse.Namespace) -> Dict:
    """
    Wrapper for compose so that it can be called with `argparse` arguments from
    the CLI.
    # Parameters
    args: `argparse.Namespace`
        The parsed args from `argparse`.

    # Returns
        The composed config.

    """
    return compose_config(
        config_path=args.config_path,
        config_name=args.config_name,
        job_name=args.job_name,
        serialization_dir=args.serialization_dir,
    )


def compose_config(
        config_path: Union[str, PathLike], config_name: str, job_name: str,
        serialization_dir: Union[str, PathLike]
) -> Dict:
    # Make the config path relative to the location of THIS file. I.E. make it
    # relative to the `allennlp_hydra/commands` subdirectory.
    config_path = Path(config_path).absolute().resolve()
    if not config_path.exists():
        raise ValueError(f"Config path '{config_path}' does not exist")
    if not config_path.is_dir():
        raise ValueError(f"Config path '{config_path}' is not a directory")

    # Compose the config with hydra.
    with hydra.initialize_config_dir(config_dir=str(config_path), job_name=job_name):
        cfg = hydra.compose(config_name=config_name)

    # cfg is a `DictConfig` object, so we need to convert it to a normal dict
    # using OmegaConf in order to save it.
    cfg = OmegaConf.to_object(cfg)
    serialization_dir = Path(serialization_dir)
    with serialization_dir.joinpath(f"{config_name}.json").open(
            "w", encoding="utf-8"
    ) as cfg_file:
        # Add the extra options for readability.
        json.dump(cfg, cfg_file, indent=True, sort_keys=True)

    return cfg
