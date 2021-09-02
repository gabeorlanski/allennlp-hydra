from copy import deepcopy
import json
import pytest

from allennlp.common import Registrable, Params

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


@A.register('B')
class B(A):
    def __init__(
            self,
            arg_a: str,
            arg_c: bool,
            kwarg_a: int = 3,
            kwarg_b: bool = False,
            kwarg_c: int = None
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


@A.register('D')
class D(A):
    def __init__(
            self,
            arg_a: str,
            arg_b: str,
            kwarg_a: int = 5,
            kwarg_b: A = None
    ):
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

    @pytest.mark.parametrize("cfg, expected", [
        [
            {"type": "B", "arg_a": "test", "arg_c": False},
            {
                "type"   : "B", "arg_a": "test", "arg_c": False, "kwarg_a": 3, "kwarg_b": False,
                "kwarg_c": None
            }
        ],
        [
            {
                "type"   : "B", "arg_a": "test", "arg_c": False, "kwarg_a": 2, "kwarg_b": True,
                "kwarg_c": 1
            },
            {
                "type"   : "B", "arg_a": "test", "arg_c": False, "kwarg_a": 2, "kwarg_b": True,
                "kwarg_c": 1
            }
        ],
        [

            {
                "type"   : "D", "arg_a": "test", "arg_b": "nest", "kwarg_a": 2,
                "kwarg_b": {
                    "type": "C", "arg_a": "nested", "arg_b": "class"
                }
            },
            {
                "type"   : "D", "arg_a": "test", "arg_b": "nest", "kwarg_a": 2,
                "kwarg_b": {
                    "type": "C", "arg_a": "nested", "arg_b": "class", "kwarg_a": 1
                }
            },
        ],
        [

            {
                "type"   : "D", "arg_a": "test", "arg_b": "nest", "kwarg_a": 2,
                "kwarg_b": {
                    "type": "E"
                }
            },
            {
                "type"   : "D", "arg_a": "test", "arg_b": "nest", "kwarg_a": 2,
                "kwarg_b": {
                    "type": "E"
                }
            },
        ]
    ], ids=[
        "empty",
        "full",
        "nested",
        "nested_empty"
    ])
    def test_fill_config(self, cfg, expected):
        result = fill_defaults.fill_config_with_default_values(A, Params(cfg))

        assert result.params == expected

        actual_class = A.from_params(result)
        expected_class = A.from_params(Params(cfg))
        assert actual_class == expected_class
