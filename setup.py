import setuptools
from setuptools import setup

setup(
    name='treetoolml',
    version='1.0.1',    
    description='Python package for tree detection, segmentation and extraction of DBH',
    url='https://github.com/porteratzo/TreeToolML',
    author='Omar Montoya',
    author_email='omar.alfonso.montoya@hotmail.com',
    license='MIT License',
    packages=setuptools.find_packages(),
    install_requires=['treetool','open3d', 'numpy', 'pandas',
    'matplotlib', 'lsq-ellipse', 'tqdm', 'numpy', 'yacs', 'plyfile', 'fvcore', 'tensorboard', 'torchsummary'],
    classifiers=[
    ],
)