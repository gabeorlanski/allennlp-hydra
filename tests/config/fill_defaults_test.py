import pytest

from allennlp.common import Registrable, Params
from allennlp.data import DataLoader
from allennlp.training import Trainer
from allennlp.models import Model

from allennlp_hydra.utils.testing import BaseTestCase
from allennlp_hydra.config import fill_defaults


class A(Registrable):
    def __init__(self, arg_a: str, arg_b: str, kwarg_a: int = 5):
        self.arg_a = arg_a
        self.arg_b = arg_b
        self.kwarg_a = kwarg_a

    def __eq__(self, other):
        if other.arg_a != self.arg_a:
            return False
        if other.arg_b != self.arg_b:
            return False
        return self.kwarg_a == other.kwarg_a


@A.register("B")
class B(A):
    def __init__(
        self,
        arg_a: str,
        arg_c: bool,
        kwarg_a: int = 3,
        kwarg_b: bool = False,
        kwarg_c: int = None,
    ):
        super(B, self).__init__(arg_a, "N/A", kwarg_a)
        self.arg_c = arg_c
        self.kwarg_b = kwarg_b
        self.kwarg_c = kwarg_c

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if self.arg_c != other.arg_c:
            return False
        if self.kwarg_b != other.kwarg_b:
            return False
        return self.kwarg_c == other.kwarg_c


@A.register("C")
class C(A):
    def __init__(self, arg_a: str, arg_b: str, kwarg_a: int = 1):
        super(C, self).__init__(arg_a, arg_b, kwarg_a)


@A.register("D")
class D(A):
    def __init__(self, arg_a: str, arg_b: str, kwarg_a: int = 5, kwarg_b: A = None):
        super(D, self).__init__(arg_a, arg_b, kwarg_a)
        self.kwarg_b = kwarg_b or A(arg_a, arg_b, kwarg_a)

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        return self.kwarg_b == other.kwarg_b


@A.register("E")
class E(A):
    def __init__(self):
        super(E, self).__init__("This", "Empty", 3)


class TestFillDefaults(BaseTestCase):
    """
    Tests for the functions in `allennlp_hydra.config.fill_defaults`.
    """

    @pytest.mark.parametrize(
        "cfg, expected",
        [
            [
                {"type": "B", "arg_a": "test", "arg_c": False},
                {
                    "type": "B",
                    "arg_a": "test",
                    "arg_c": False,
                    "kwarg_a": 3,
                    "kwarg_b": False,
                    "kwarg_c": None,
                },
            ],
            [
                {
                    "type": "B",
                    "arg_a": "test",
                    "arg_c": False,
                    "kwarg_a": 2,
                    "kwarg_b": True,
                    "kwarg_c": 1,
                },
                {
                    "type": "B",
                    "arg_a": "test",
                    "arg_c": False,
                    "kwarg_a": 2,
                    "kwarg_b": True,
                    "kwarg_c": 1,
                },
            ],
            [
                {
                    "type": "D",
                    "arg_a": "test",
                    "arg_b": "nest",
                    "kwarg_a": 2,
                    "kwarg_b": {"type": "C", "arg_a": "nested", "arg_b": "class"},
                },
                {
                    "type": "D",
                    "arg_a": "test",
                    "arg_b": "nest",
                    "kwarg_a": 2,
                    "kwarg_b": {
                        "type": "C",
                        "arg_a": "nested",
                        "arg_b": "class",
                        "kwarg_a": 1,
                    },
                },
            ],
            [
                {
                    "type": "D",
                    "arg_a": "test",
                    "arg_b": "nest",
                    "kwarg_a": 2,
                    "kwarg_b": {"type": "E"},
                },
                {
                    "type": "D",
                    "arg_a": "test",
                    "arg_b": "nest",
                    "kwarg_a": 2,
                    "kwarg_b": {"type": "E"},
                },
            ],
        ],
        ids=["empty", "full", "nested", "nested_empty"],
    )
    def test_fill_config(self, cfg, expected):
        result = fill_defaults.fill_config_with_default_values(A, Params(cfg))

        assert result == expected

        actual_class = A.from_params(Params(result))
        expected_class = A.from_params(Params(cfg))
        assert actual_class == expected_class

    def test_fill_default_init_class(self):
        result = fill_defaults.fill_config_with_default_values(
            DataLoader,
            {
                "batch_sampler": {
                    "batch_size": 80,
                    "padding_noise": 0.0,
                    "sorting_keys": ["tokens"],
                    "type": "bucket",
                }
            },
        )
        assert result == {
            "batch_size": None,
            "drop_last": False,
            "shuffle": False,
            "batch_sampler": {
                "batch_size": 80,
                "padding_noise": 0.0,
                "sorting_keys": ["tokens"],
                "type": "bucket",
                "drop_last": False,
                "shuffle": True,
            },
            "batches_per_epoch": None,
            "num_workers": 0,
            "max_instances_in_memory": None,
            "start_method": "fork",
            "cuda_device": None,
            "quiet": False,
            "collate_fn": {"type": "allennlp"},
        }

    def test_default_lazy_object(self):
        result = fill_defaults.fill_config_with_default_values(
            Trainer,
            {
                "cuda_device": -1,
                "grad_norm": 1.0,
                "learning_rate_scheduler": {
                    "model_size": 1024,
                    "type": "noam",
                    "warmup_steps": 5,
                },
                "num_epochs": 1,
                "patience": 500,
            },
        )

        assert result == {
            "cuda_device": -1,
            "grad_norm": 1.0,
            "learning_rate_scheduler": {
                "model_size": 1024,
                "type": "noam",
                "warmup_steps": 5,
                "factor": 1.0,
                "last_epoch": -1,
            },
            "num_epochs": 1,
            "patience": 500,
            "validation_data_loader": None,
            "local_rank": 0,
            "validation_metric": "-loss",
            "grad_clipping": None,
            "distributed": False,
            "world_size": 1,
            "num_gradient_accumulation_steps": 1,
            "use_amp": False,
            "no_grad": None,
            "optimizer": {
                "type": "adam",
                "parameter_groups": None,
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0.0,
                "amsgrad": False,
            },
            "momentum_scheduler": None,
            "moving_average": None,
            "checkpointer": {
                "type": "default",
                "save_completed_epochs": True,
                "save_every_num_seconds": None,
                "save_every_num_batches": None,
                "keep_most_recent_by_count": 2,
                "keep_most_recent_by_age": None,
            },
            "callbacks": None,
            "enable_default_callbacks": True,
            "run_confidence_checks": True,
        }

    def test_from_params_default(self):
        result = fill_defaults.fill_config_with_default_values(
            Model,
            {
                "text_field_embedder": [
                    {
                        "token_embedders": [
                            {
                                "tokens": [
                                    {"type": "embedding"},
                                    {"vocab_namespace": "source_tokens"},
                                    {"embedding_dim": 512},
                                    {"trainable": True},
                                ]
                            }
                        ]
                    }
                ],
                "type": "basic_classifier",
            },
        )

        assert result == {
            "text_field_embedder": [
                {
                    "token_embedders": [
                        {
                            "tokens": [
                                {"type": "embedding"},
                                {"vocab_namespace": "source_tokens"},
                                {"embedding_dim": 512},
                                {"trainable": True},
                            ]
                        }
                    ]
                }
            ],
            "type": "basic_classifier",
            "seq2seq_encoder": None,
            "feedforward": None,
            "dropout": None,
            "num_labels": None,
            "label_namespace": "labels",
            "namespace": "tokens",
            "initializer": {"regexes": None, "prevent_regexes": None},
        }
