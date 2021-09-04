"""
The `compose` command creates an AllenNLP config by composing a set of `yaml` files with Hydra's
[`Compose API`](https://hydra.cc/docs/advanced/compose_api). Overriding
using Hydra's [`Override Grammar`](https://hydra.cc/docs/advanced/compose_api)
is also supported.

# Parameters

config_path: `Union[str, PathLike]`
    Path to the root config directory.

config_name: `str`
    The name of the root config file. Do NOT include the `.yaml`.

job_name: `str`
    The job name. This is passed to Hydra and is not used here.

-s/--serialization-dir: `Union[str, PathLike]`
    The directory where the new AllenNLP config will be saved to. The name used
     for saving will be the `config_name` and it will be a `.json` file.

-o/--overrides: `List[str]`, optional (default=`[]`)
    Keyword arguments passed will be used as a list of overrides using Hydra's
    override grammar for the config.

    Example usage:

    ```zsh
    allennlp compose conf config example -s example --overrides A=B C="D"
    ```
    Will be interpreted as overrides `['A=B', 'C="D"']`



# Example

Say you have the following directory structure:

```
project
+-- conf
|   +-- dataset_readers
|   |   +-- A.yaml
|   |   +-- B.yaml
|   +-- models
|   |   +-- C.yaml
|   |   +-- D.yaml
|   +-- config.yaml
+-- experiments
```

`conf/dataset_readers/A.yaml`:

```yml
type: A
start_token: <s>
end_token: </s>
```

`conf/dataset_readers/B.yaml`:

```yml
type: B
start_token: [CLS]
end_token: [SEP]
```


`conf/models/C.yaml`:

```
type: C
layers: 5
```

`conf/models/D.yaml`:

```YAML
type: D
input_dim: 10
```


`config.yaml`
```yml
defaults:
    - dataset_reader: A
    - model: C

debug: false
```

Then running the command
```zsh
allennlp compose conf config example -s experiments
```
Produces the file `project/experiments/config.json`
```json
{
    "dataset_reader":{
        "type": "A",
        "start_token": "<s>",
        "end_token": "</s>"
    },
    "model": {
        "type": "C",
        "layers": 5
    },
    "debug": false
}
```

If you want to override the config and use the `B` dataset reader with the `D`
model, you would modify the previous command:
```zsh
allennlp compose conf config example -s experiments -o model=D dataset_reader=B
```
Produces the file `project/experiments/config.json`
```json
{
    "dataset_reader":{
        "type": "B",
        "start_token": "[CLS]",
        "end_token": "[SEP]"
    },
    "model": {
        "type": "D",
        "input_dim": 10
    },
    "debug": false
}
```

And if you wanted to change `input_dim` of model `D` to 25:
```zsh
allennlp compose conf config example -s experiments -o model=D dataset_reader=B model.input_dim=25
```

Produces the file `project/experiments/config.json`
```json
{
    "dataset_reader":{
        "type": "B",
        "start_token": "[CLS]",
        "end_token": "[SEP]"
    },
    "model": {
        "type": "D",
        "input_dim": 25
    },
    "debug": false
}
```
"""

from typing import Dict, Union, List, Optional

import argparse
import json
from os import PathLike
import logging
from pathlib import Path

from allennlp.data import DataLoader, DatasetReader
from allennlp.training import Trainer
from allennlp.models import Model
from allennlp.commands.subcommand import Subcommand
import hydra
from omegaconf import OmegaConf
from overrides import overrides

from allennlp_hydra.config.fill_defaults import fill_config_with_default_values

logger = logging.getLogger(__name__)


@Subcommand.register("compose")
class ComposeConfig(Subcommand):
    @overrides
    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        description = """Compose a config with Hydra"""
        subparser = parser.add_parser(
            self.name, description=description, help=description
        )

        subparser.add_argument(
            "config_path", type=str, help="Path to the config directory."
        )

        subparser.add_argument(
            "config_name", type=str, help="Name of the config file to use."
        )
        subparser.add_argument("job_name", type=str, help="Name of the job.")

        subparser.add_argument(
            "-s",
            "--serialization-dir",
            required=True,
            type=str,
            help="Directory to save the config to. The name of the config will "
            "be `{config_name}.json`",
        )
        subparser.add_argument(
            "-o",
            "--overrides",
            nargs="*",
            help="Any key=value arguments to override config values "
            "(use dots for.nested=overrides)",
        )

        subparser.add_argument(
            "--fill-defaults",
            action="store_true",
            default=False,
            help="Add default arguments from each loaded class to the config.",
        )

        subparser.set_defaults(func=compose_config_from_args)

        return subparser


def compose_config_from_args(args: argparse.Namespace) -> Dict:
    """
    Wrapper for compose so that it can be called with `argparse` arguments from
    the CLI.
    # Parameters
    args: `argparse.Namespace`
        The parsed args from `argparse`.

    # Returns
        The composed config.

    """
    return compose_config(
        config_path=args.config_path,
        config_name=args.config_name,
        job_name=args.job_name,
        serialization_dir=args.serialization_dir,
        config_overrides=args.overrides,
    )


def compose_config(
    config_path: Union[str, PathLike],
    config_name: str,
    job_name: str,
    serialization_dir: Optional[Union[str, PathLike]] = None,
    config_overrides: List[str] = None,
    fill_defaults: bool = False,
) -> Dict:
    """
    Create an AllenNLP config by composing a set of `yaml` files with Hydra's
    [`Compose API`](https://hydra.cc/docs/advanced/compose_api). Overriding
    using Hydra's [`Override Grammar`](https://hydra.cc/docs/advanced/compose_api)
    is also supported.

    # Parameters

    config_path: `Union[str, PathLike]`
        Path to the root config directory.

    config_name: `str`
        The name of the root config file.

    job_name: `str`
        The job name. This is passed to Hydra and is not used here.

    serialization_dir: `Optional[Union[str, PathLike]]`, optional (default=`None`)
        If this is passed, it is the directory where the new AllenNLP config
        will be saved to. The name used for saving will be the `config_name` and
         it will be a `.json` file.

    config_overrides: `List[str]`, optional (default=`[]`)
        List of overrides using Hydra's override grammar for the config.

    fill_defaults: `bool`, optional (default=`False`)
        Add arguments and their default values to the config if they are not
        specified.

    # Returns

    `Dict`
        The dictionary config generated by Hydra.

    """
    if config_overrides is None:
        config_overrides = []

    # Make the config path relative to the location of THIS file. I.E. make it
    # relative to the `allennlp_hydra/commands` subdirectory.
    config_path = Path(config_path).absolute().resolve()
    if not config_path.exists():
        raise ValueError(f"Config path '{config_path}' does not exist")
    if not config_path.is_dir():
        raise ValueError(f"Config path '{config_path}' is not a directory")

    # Compose the config with hydra.
    with hydra.initialize_config_dir(config_dir=str(config_path), job_name=job_name):
        cfg = hydra.compose(config_name=config_name, overrides=config_overrides)

    # cfg is a `DictConfig` object, so we need to convert it to a normal dict
    # using OmegaConf in order to save it.
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # If filling the defaults, fill them here.
    if fill_defaults:
        cfg["data_loader"] = fill_config_with_default_values(
            DataLoader, cfg["data_loader"]
        )
        cfg["dataset_reader"] = fill_config_with_default_values(
            DatasetReader, cfg["dataset_reader"]
        )
        cfg["model"] = fill_config_with_default_values(Model, cfg["model"])
        cfg["trainer"] = fill_config_with_default_values(Trainer, cfg["trainer"])

    # We only save if a serialization dir was passed.
    if serialization_dir is not None:
        cfg_save_path = Path(serialization_dir).joinpath(f"{config_name}.json")
        with cfg_save_path.open("w", encoding="utf-8") as cfg_file:
            # Add the extra options for readability.
            json.dump(cfg, cfg_file, indent=True, sort_keys=True)

    return cfg
