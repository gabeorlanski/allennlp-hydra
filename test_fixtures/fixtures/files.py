import json
import pytest

from allennlp_hydra.utils.testing import FIXTURES_ROOT


@pytest.fixture()
def simple_config():
    yield json.loads(
        FIXTURES_ROOT.joinpath("expected_configs/simple_config.json").read_text("utf-8")
    )


@pytest.fixture()
def simple_tagger_config():
    yield json.loads(
        FIXTURES_ROOT.joinpath("expected_configs/simple_tagger.jsonnet").read_text(
            "utf-8"
        )
    )
