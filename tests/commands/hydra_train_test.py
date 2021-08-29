import argparse
from copy import deepcopy
from unittest.mock import patch

from allennlp.common import Params

from allennlp_hydra.utils.testing import BaseTestCase
from allennlp_hydra.commands import hydra_train


class TestHydraTrainCommand(BaseTestCase):
    def test_call_with_args(self, simple_config):
        # Create the args needed to test the function.
        args = argparse.Namespace()
        args.config_path = self.FIXTURES_ROOT.joinpath("conf")
        args.config_name = "simple_config"
        args.job_name = "testing"
        args.overrides = ["+trainer.grad_clipping=1.0"]
        args.serialization_dir = self.TEST_DIR
        args.recover = False
        args.force = False
        args.node_rank = 0
        args.include_package = ["src"]
        args.dry_run = True
        args.file_friendly_logging = True

        # Setup the expected config dict
        expected = deepcopy(simple_config)
        expected["trainer"]["grad_clipping"] = 1.0

        # Patch the train function because it is straight from AllenNLP and
        # we do not need to test that.
        with patch("allennlp_hydra.commands.hydra_train.train_model") as mock_train:
            hydra_train.hydra_train_model_from_args(args)
            assert mock_train.call_count == 1
            for k, v in mock_train.call_args.kwargs.items():
                if k == "params":
                    continue
                assert v == getattr(args, k, f"MISSING ATTRIBUTE {k}"), k

            params = mock_train.call_args.kwargs["params"]
            assert isinstance(params, Params)
            assert params.params == expected
