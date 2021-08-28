# AllenNLP-Hydra

Plugin For [AllenNLP](https://github.com/allenai/allennlp) that enables 
composing configs through the use of the 
[Hydra Library from Facebook Research](https://github.com/facebookresearch/hydra).

We use the same 
[contributions guideline](https://github.com/gabeorlanski/allennlp-hydra/blob/master/CONTRIBUTING.md) 
as AllenNLP in order to maintain similar code styles. For this reason [our style 
guide](https://github.com/gabeorlanski/allennlp-hydra/blob/master/STYLE.md) is 
the same as [that found in their repository](https://github.com/allenai/allennlp/blob/main/STYLE.md).


# Main Components:

1. `allennlp compose`: Command that uses Hydra to compose a config from multiple 
YAML files. 

2. `allennlp fill-config`: Command to fill a config with the default values it 
does not specify.