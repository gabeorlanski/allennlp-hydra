from copy import deepcopy
import json

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
            "type": "polynomial_decay",
            "power": 2,
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
