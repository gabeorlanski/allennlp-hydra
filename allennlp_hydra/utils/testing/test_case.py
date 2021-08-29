import pathlib
import pytest
import logging

__all__ = ["PROJECT_ROOT", "TEST_ROOT", "FIXTURES_ROOT", "BaseTestCase"]

PROJECT_ROOT = pathlib.Path(__file__).parents[3].resolve()
TEST_ROOT = PROJECT_ROOT.joinpath("tests")
FIXTURES_ROOT = PROJECT_ROOT.joinpath("test_fixtures")


class BaseTestCase:
    """
    A custom testing class that disables some of the more verbose AllenNLP
    logging and that creates and destroys a temp directory as a test fixture.
    """

    PROJECT_ROOT = PROJECT_ROOT
    MODULE_ROOT = PROJECT_ROOT / "src"
    TESTS_ROOT = TEST_ROOT
    FIXTURES_ROOT = FIXTURES_ROOT
    FIXTURES_DATA_PATH = FIXTURES_ROOT / "data"

    # This is crucial for making the test cases not have permission denied
    # issues later down the line.
    @pytest.fixture(autouse=True)
    def test_dir(self, tmpdir):
        self.TEST_DIR = pathlib.Path(tmpdir.mkdir("allennlp_hydra"))

    def setup_method(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.DEBUG,
        )
        # Disabling some of the more verbose logging statements that typically aren't very helpful
        # in tests.
        logging.getLogger("allennlp.common.params").disabled = True
        logging.getLogger("allennlp.nn.initializers").disabled = True
        logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(
            logging.INFO
        )
        logging.getLogger("urllib3.connectionpool").disabled = True
