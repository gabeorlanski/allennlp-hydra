"""
Taken from https://github.com/allenai/allennlp-models/blob/b17d114ed49a711e66fd1bbe422b964f86296a6d/allennlp_models/version.py
"""
import os

_MAJOR = "0"
_MINOR = "1"
_PATCH = "1"
# This is mainly for nightly builds which have the suffix ".dev$DATE". See
# https://semver.org/#is-v123-a-semantic-version for the semantics.
_SUFFIX = os.environ.get("ALLENNLP_MODELS_VERSION_SUFFIX", "")

VERSION_SHORT = "{0}.{1}".format(_MAJOR, _MINOR)
VERSION = "{0}.{1}.{2}{3}".format(_MAJOR, _MINOR, _PATCH, _SUFFIX)