# AllenNLP-Hydra

Plugin For [AllenNLP](https://github.com/allenai/allennlp) that enables 
composing configs through the use of the 
[Hydra Library from Facebook Research](https://github.com/facebookresearch/hydra).

**NOTE** there is no affiliation between this project and AllenNLP or the Allen 
Institute for AI.

We use the same 
[contributions guideline](https://github.com/gabeorlanski/allennlp-hydra/blob/master/CONTRIBUTING.md) 
as AllenNLP in order to maintain similar code styles. For this reason [our style 
guide](https://github.com/gabeorlanski/allennlp-hydra/blob/master/STYLE.md) is 
the same as [that found in their repository](https://github.com/allenai/allennlp/blob/main/STYLE.md).


# Install Instructions

Clone the repo

```shell
pip install .
echo allennlp_hydra >> ~.allennlp_plugins
```

The second line adds `allennlp-hydra` to the allennlp plugins file so that it 
can globally be recognized.

# Basic Guide

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