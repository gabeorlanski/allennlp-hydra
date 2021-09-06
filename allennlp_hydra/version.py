"""
Taken from
https://github.com/allenai/allennlp-models/blob/main/allennlp_models/version.py
"""
_MAJOR = "0"
_MINOR = "0"
_PATCH = "3"

VERSION_SHORT = "{0}.{1}".format(_MAJOR, _MINOR)
VERSION = "{0}.{1}.{2}".format(_MAJOR, _MINOR, _PATCH)
__version__ = VERSION