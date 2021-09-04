import argparse
import pytest
import yaml

from allennlp_hydra.utils.testing import BaseTestCase
from allennlp_hydra.commands import class_to_yaml


class TestClassToYaml(BaseTestCase):
    """
    Tests for the class to yaml command.
    """

    @pytest.mark.parametrize("serialization_arg", ["-s", "--serialization-dir"])
    def test_cli_args(self, serialization_arg):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title="Commands", metavar="")
        class_to_yaml.ClassToYaml().add_subparser(subparsers)

        raw_args = [
            "class2yaml",
            "sequence_tagging",
            "DatasetReader",
            serialization_arg,
            "serialization_dir",
            "--force",
        ]

        args = parser.parse_args(raw_args)

        assert args.func == class_to_yaml.class_to_yaml_from_args
        assert args.cls_name == "sequence_tagging"
        assert args.base_cls_name == "DatasetReader"
        assert args.serialization_dir == "serialization_dir"
        assert args.force

    @pytest.mark.parametrize(
        "base_cls,expected",
        [
            [
                "DatasetReader",
                {
                    "type": "sequence_tagging",
                    "word_tag_delimiter": "###",
                    "token_delimiter": None,
                    "token_indexers": None,
                },
            ],
            ["DatasetReader", {"type": "multitask", "readers": "???"}],
        ],
        ids=["simple", "positional_args"],
    )
    def test_class_to_yaml(self, base_cls, expected):
        result = class_to_yaml.class_to_yaml(
            cls_name=expected["type"],
            base_cls_name=base_cls,
            serialization_dir=str(self.TEST_DIR),
            force=False,
        )

        assert self.TEST_DIR.joinpath(f"{expected['type']}.yaml").exists()

        saved_file = yaml.load(
            self.TEST_DIR.joinpath(f"{expected['type']}.yaml").open(
                "r", encoding="utf-8"
            ),
            yaml.Loader,
        )

        assert result == saved_file
        assert result == expected
