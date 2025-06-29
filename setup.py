import setuptools
from setuptools import setup

setup(
    name="treetoolml",
    version="1.0.1",
    description="Python package for tree detection, segmentation and extraction of DBH",
    url="https://github.com/porteratzo/TreeToolML",
    author="Omar Montoya",
    author_email="omar.alfonso.montoya@hotmail.com",
    license="MIT License",
    packages=setuptools.find_packages(),
    install_requires=[
        "treetool",
        "open3d",
        "pandas",
        "matplotlib",
        "tqdm",
        "numpy",
        "yacs",
        "plyfile",
        "fvcore",
        "tensorboard",
        "torchsummary",
        "torch",
        "torchvision",
        "jupyter",
        "ipykernel",
        "scipy",
        "python-pdal",
        "lsq-ellipse",
    ],
    dependency_links=[
        "git+https://github.com/porteratzo/TreeTool.git#egg=treetool",
        "git+https://github.com/porteratzo/porteratzo3D.git#egg=porteratzo3D",
    ],
    classifiers=[],
)
