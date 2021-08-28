import argparse
import logging

from overrides import overrides

from allennlp.commands.subcommand import Subcommand

logger = logging.getLogger(__name__)


@Subcommand.register("compose")
class Compose(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Compose a config with Hydra"""
        subparser = parser.add_parser(self.name, description=description, help=description)

        subparser.add_argument("config_path",
                               type=str,
                               help="Path to the config directory.")
        subparser.add_argument("job_name",
                               type=str,
                               help="Name of the job.")

        subparser.set_defaults(func=compose_from_args)

        return subparser


def compose_from_args(args: argparse.Namespace):
    return compose(
        config_path=args.config_path,
        job_name=args.job_name
    )


def compose(
        config_path: str,
        job_name: str
):
    raise NotImplementedError()
