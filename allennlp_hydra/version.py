"""
Taken from https://github.com/allenai/allennlp-models/blob/b17d114ed49a711e66fd1bbe422b964f86296a6d/allennlp_models/version.py
"""
import os

_MAJOR = "0"
_MINOR = "0"
_PATCH = "1"

VERSION_SHORT = "{0}.{1}".format(_MAJOR, _MINOR)
VERSION = "{0}.{1}.{2}{3}".format(_MAJOR, _MINOR, _PATCH)
