import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="allennlp-hydra",
    version="1.0.0",
    author="Gabriel Orlanski",
    author_email="gabeorlanski@gmail.com",
    description="Hydra plugin for allennlp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "test_*"],
    ),
    project_urls={
        "Documentation": "https://ssljax.readthedocs.io/en/stable",
        "Source Code": "https://github.com/gabeorlanski/pathml",
    },
    install_requires=["allennlp", "hydra", "pytest"],
    requires_python=">=3.7",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
