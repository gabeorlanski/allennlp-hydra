import argparse
import os
import json
from copy import deepcopy
from unittest.mock import patch

import pytest
from allennlp.common import Params
from allennlp.commands.train import train_model

from allennlp_hydra.utils.testing import BaseTestCase, assert_models_weights_equal
from allennlp_hydra.commands import hydra_train


class TestHydraTrainCommand(BaseTestCase):
    @pytest.fixture()
    def train_args(self):
        # Create the args needed to test the function.
        args = argparse.Namespace()
        args.config_path = self.FIXTURES_ROOT.joinpath("conf")
        args.config_name = "simple_tagger"
        args.job_name = "testing"
        args.overrides = []
        args.serialization_dir = self.TEST_DIR.joinpath("test_hydra_train")
        args.recover = False
        args.force = False
        args.node_rank = 0
        args.include_package = []
        args.dry_run = False
        args.file_friendly_logging = True
        yield args

    def test_mock_call_with_args(self, simple_config, train_args):
        train_args.config_name = "simple_config"
        train_args.overrides = ["+trainer.grad_clipping=1.0"]

        # Setup the expected config dict
        expected = deepcopy(simple_config)
        expected["trainer"]["grad_clipping"] = 1.0

        # Patch the train function because it is straight from AllenNLP and
        # we do not need to test that.
        with patch("allennlp_hydra.commands.hydra_train.train_model") as mock_train:
            hydra_train.hydra_train_model_from_args(train_args)
            assert mock_train.call_count == 1
            for k, v in mock_train.call_args.kwargs.items():
                if k == "params":
                    continue
                assert v == getattr(train_args, k, f"MISSING ATTRIBUTE {k}"), k

            params = mock_train.call_args.kwargs["params"]
            assert isinstance(params, Params)
            assert params.params == expected

    def test_call_with_args(self, simple_tagger_config, train_args):
        if os.getcwd() != str(self.PROJECT_ROOT.absolute()):
            os.chdir(self.PROJECT_ROOT)

        train_args.overrides = [
            "trainer/learning_rate_scheduler=polynomial_decay",
            "trainer.learning_rate_scheduler.warmup_steps=0",
        ]
        hydra_train.hydra_train_model_from_args(train_args)

        serial_dir = self.TEST_DIR.joinpath("test_hydra_train")
        assert serial_dir.exists()
        assert serial_dir.is_dir()

        assert serial_dir.joinpath("config.json").exists()

        saved_config = json.loads(serial_dir.joinpath("config.json").read_text("utf-8"))
        assert saved_config == simple_tagger_config

    def test_hydra_train_same_allennlp_train(self, simple_tagger_config, train_args):
        if os.getcwd() != str(self.PROJECT_ROOT.absolute()):
            os.chdir(self.PROJECT_ROOT)

        train_args.overrides = [
            "trainer/learning_rate_scheduler=polynomial_decay",
            "trainer.learning_rate_scheduler.warmup_steps=0",
        ]
        hydra_result = hydra_train.hydra_train_model_from_args(train_args)

        allennlp_result = train_model(
            params=Params(simple_tagger_config),
            serialization_dir=self.TEST_DIR.joinpath("allennlp_train"),
            recover=train_args.recover,
            force=train_args.force,
            node_rank=train_args.node_rank,
            include_package=train_args.include_package,
            dry_run=train_args.dry_run,
            file_friendly_logging=train_args.file_friendly_logging,
        )

        # Check the representations
        assert str(allennlp_result) == str(hydra_result)

        # Check the parameters
        assert_models_weights_equal(allennlp_result, hydra_result)

        # Check the results
        hydra_metrics = json.loads(
            train_args.serialization_dir.joinpath("metrics.json").read_text("utf-8")
        )
        allennlp_metrics = json.loads(
            self.TEST_DIR.joinpath("allennlp_train/metrics.json").read_text("utf-8")
        )
        for k, v in hydra_metrics.items():
            if "best" not in k:
                continue

            assert allennlp_metrics[k] == v, k

    @pytest.mark.parametrize('serialization_arg', ['-s', '--serialization-dir'])
    @pytest.mark.parametrize('override_arg', ['-o', '--overrides', None])
    def test_cli_args(self, serialization_arg, override_arg):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title="Commands", metavar="")
        hydra_train.HydraTrain().add_subparser(subparsers)

        raw_args = [
            "hydra-train",
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

        assert args.func == hydra_train.hydra_train_model_from_args
        assert args.config_path == "path/to/config"
        assert args.config_name == "config_name"
        assert args.job_name == "job_name"
        assert args.serialization_dir == "serialization_dir"
        assert args.overrides == expected_overrides
