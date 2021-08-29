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
            "overrides",
            nargs="*",
            help="Any key=value arguments to override config values "
            "(use dots for.nested=overrides)",
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
