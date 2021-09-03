import argparse
from copy import deepcopy
import json
import pytest

from allennlp_hydra.utils.testing import BaseTestCase
from allennlp_hydra.commands import compose_config


class TestComposeCommand(BaseTestCase):
    """
    Tests for the compose command.
    """

    def test_simple_config(self, simple_config):
        """
        Test creating a simple config with compose.
        """
        result = compose_config.compose_config(
            config_path=str(self.FIXTURES_ROOT.joinpath("conf")),
            config_name="simple_config",
            serialization_dir=str(self.TEST_DIR.absolute().resolve()),
            job_name="test_simple_config",
        )

        # Test that it saved the config to the output directory
        output_config = self.TEST_DIR.joinpath("simple_config.json")
        assert output_config.exists()

        # Read the contents of that file and check it is equal to what was
        # returned.
        saved_config = json.loads(output_config.read_text("utf-8"))
        assert saved_config == result

        assert result == simple_config

    def test_simple_config_overrides(self, simple_config):
        # Setup the overrides, copy the original config to make sure it is not
        # changed.
        expected = deepcopy(simple_config)
        expected["dataset_reader"]["word_tag_delimiter"] = "__"
        expected["trainer"]["learning_rate_scheduler"] = {
            "type"        : "polynomial_decay",
            "power"       : 2,
            "warmup_steps": 250,
        }
        expected["trainer"]["grad_clipping"] = 1.0
        expected["trainer"].pop("grad_norm")

        result = compose_config.compose_config(
            config_path=str(self.FIXTURES_ROOT.joinpath("conf")),
            config_name="simple_config",
            serialization_dir=str(self.TEST_DIR.absolute().resolve()),
            job_name="test_simple_config",
            config_overrides=[
                'dataset_reader.word_tag_delimiter="__"',
                "trainer/learning_rate_scheduler=polynomial_decay",
                "++trainer.learning_rate_scheduler.warmup_steps=250",
                "+trainer.grad_clipping=1.0",
                "~trainer.grad_norm",
            ],
        )

        output_config = self.TEST_DIR.joinpath("simple_config.json")
        assert output_config.exists()

        # Read the contents of that file and check it is equal to what was
        # returned.
        saved_config = json.loads(output_config.read_text("utf-8"))
        assert saved_config == result

        assert result == expected

    @pytest.mark.parametrize('serialization_arg', ['-s', '--serialization-dir'])
    @pytest.mark.parametrize('override_arg', ['-o', '--overrides', None])
    def test_cli_args(self, serialization_arg, override_arg):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title="Commands", metavar="")
        compose_config.ComposeConfig().add_subparser(subparsers)

        raw_args = [
            "compose",
            "path/to/config",
            "config_name",
            "job_name",
            serialization_arg,
            "serialization_dir"
        ]

        expected_overrides = None
        if override_arg is not None:
            expected_overrides = [
                "dataset_reader.word_tag_delimiter='__'",
                "++trainer.learning_rate_scheduler.warmup_steps=250"
            ]
            raw_args.extend([
                override_arg,
                *expected_overrides
            ])

        args = parser.parse_args(raw_args)

        assert args.func == compose_config.compose_config_from_args
        assert args.config_path == "path/to/config"
        assert args.config_name == "config_name"
        assert args.job_name == "job_name"
        assert args.serialization_dir == "serialization_dir"
        assert args.overrides == expected_overrides
