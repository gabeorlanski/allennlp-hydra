import setuptools
from allennlp_hydra.version import VERSION

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="allennlp-hydra",
    version=VERSION,
    author="Gabriel Orlanski",
    author_email="gabeorlanski@gmail.com",
    description="Hydra plugin for allennlp",
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "test_*"],
    ),
    project_urls={
        "Documentation": "https://github.com/gabeorlanski/allennlp-hydra",
        "Source Code": "https://github.com/gabeorlanski/allennlp-hydra",
    },
    install_requires=[
        "allennlp>=2.6.0",
        "hydra-core>=1.1.1",
        "overrides>=3.1.0",
        "omegaconf>=2.1",
    ],
    requires_python=">=3.7",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
