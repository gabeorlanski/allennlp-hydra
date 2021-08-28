import json

from allennlp.commands import Subcommand

from allennlp_hydra.utils.testing import BaseTestCase
from allennlp_hydra.commands import compose_config


class TestComposeCommand(BaseTestCase):
    """
    Tests for the compose command.
    """

    def test_is_registered(self):
        """
        Test if compose is registered
        """
        assert "compose" in Subcommand.list_available()
        assert Subcommand.by_name("compose") == compose_config.ComposeConfig

    def test_simple_config(self):
        """
        Test creating a simple config with compose.
        """
        result = compose_config.compose_config(
            config_path=str(
                self.FIXTURES_ROOT.joinpath("conf")
            ),
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

        # Load the expected config.
        expected = json.loads(
            self.FIXTURES_ROOT.joinpath(
                "expected_configs/simple_config.json"
            ).read_text("utf-8")
        )
        assert result == expected
