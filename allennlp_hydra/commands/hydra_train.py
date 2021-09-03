"""
The `hydra-train` creates an AllenNLP config by composing a set of `yaml` files
with Hydra's [`Compose API`](https://hydra.cc/docs/advanced/compose_api) then
passing it to AllenNLP's `train` method.

Overriding using Hydra's [`Override Grammar`](https://hydra.cc/docs/advanced/compose_api)
is also supported.

The `hydra-train` command uses the underlying functionality of the
[`compose`](/allennlp-hydra/site/hydra/commands/compose_config) command. So the
use that documentation for understanding the composition of `.yaml` files.

# Parameters

config_path: `Union[str, PathLike]`
    Path to the root config directory.

config_name: `str`
    The name of the root config file. Do NOT include the `.yaml`.

job_name: `str`
    The job name. This is passed to Hydra and is not used here.

-s/--serialization-dir: `Union[str, PathLike]`
    The directory where everything is saved.

-r/--recover: `bool`, optional (default=`False`)
    Flag. Recover training from the state in `serialization_dir`

-f/--force: `bool`, optional (default=`False`)
    Flag. Overwrite the output directory if it exists.

--node-rank: `int`, optional (default=`0`)
    The rank of this node in the distributed setup

--dry-run: `bool`, optional (default=`False`)
    Flag. Do not train the model, but create a vocabulary, show dataset statistics
    and other training information

--file-friendly-logging: `bool`, optional (default=`False`)
    Flag. Outputs tqdm status on separate lines and slows tqdm refresh rate

-o/--overrides: `List[str]`, optional (default=`[]`)
    Keyword arguments passed will be used as a list of overrides using Hydra's
    override grammar for the config.

    Example usage:

    ```zsh
    --overrides A=B C="D"
    ```
    Will be interpreted as overrides `['A=B', 'C="D"']`
"""

import argparse
import logging

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.commands.train import train_model
from allennlp.common import Params

logger = logging.getLogger(__name__)


@Subcommand.register("hydra-train")
class HydraTrain(Subcommand):
    @overrides
    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """
        Train the specified model on the specified dataset with a hydra config.
        """
        subparser = parser.add_parser(
            self.name, description=description, help="Train a model."
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
            help="directory in which to save the model and its logs",
        )

        subparser.add_argument(
            "-r",
            "--recover",
            action="store_true",
            default=False,
            help="recover training from the state in serialization_dir",
        )

        subparser.add_argument(
            "-f",
            "--force",
            action="store_true",
            required=False,
            help="overwrite the output directory if it exists",
        )

        subparser.add_argument(
            "--node-rank",
            type=int,
            default=0,
            help="rank of this node in the distributed setup",
        )

        subparser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help=(
                "do not train a model, but create a vocabulary, show dataset statistics and "
                "other training information"
            ),
        )
        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )
        subparser.add_argument(
            "-o",
            "--overrides",
            nargs="*",
            help="Any key=value arguments to override config values "
            "(use dots for.nested=overrides)",
        )

        subparser.add_argument(
            "--fill-defaults",
            action="store_true",
            default=False,
            help="Add default arguments from each loaded class to the config.",
        )

        subparser.set_defaults(func=hydra_train_model_from_args)

        return subparser


def hydra_train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an `argparse.Namespace` object to string paths.
    """

    # Load the hydra config, overrides will be used here.
    from allennlp_hydra.commands import compose_config

    # We do NOT pass a serialization dir to the compose because we do not want
    # to save the config here. `train_model` handles that for us.
    config = compose_config.compose_config(
        config_path=args.config_path,
        config_name=args.config_name,
        job_name=args.job_name,
        serialization_dir=None,
        config_overrides=args.overrides,
        fill_defaults=args.fill_defaults,
    )

    return train_model(
        params=Params(config),
        serialization_dir=args.serialization_dir,
        recover=args.recover,
        force=args.force,
        node_rank=args.node_rank,
        include_package=args.include_package,
        dry_run=args.dry_run,
        file_friendly_logging=args.file_friendly_logging,
    )
