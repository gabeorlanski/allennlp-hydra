import pytest
import pkgutil
from pathlib import Path
from allennlp_hydra.utils.testing import PROJECT_ROOT


def test_imports():
    """
    Test import every file to make sure there are no invalid imports
    """
    cwd = PROJECT_ROOT
    src_code_path = cwd.joinpath('allennlp_hydra')
    assert src_code_path.exists()
    paths = [str(src_code_path.absolute().resolve())]
    for loader, module_name, is_pkg in pkgutil.walk_packages(paths):
        _module = loader.find_module(module_name).load_module(module_name)
