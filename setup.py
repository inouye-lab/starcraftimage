import setuptools
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, 'sc2image'))
from version import __version__

print(f'Version {__version__}')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sc2image",
    version=__version__,
    author="StarCraftImage team",
    author_email="dinouye@purdue.edu",
    url="https://starcraftdata.davidinouye.com/",
    description="The StarCraft Image dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = [
        'torch>=1.7.0',
        'numpy>=1.19.1',
        'tqdm>=4.53.0',
        'torchvision>=0.8.2',
        'outdated>=0.2.0',
        'pandas>=1.1.0',
        'pillow>=7.2.0',
        'tqdm>=4.6.0',
    ],
    license='MIT',
    packages=setuptools.find_packages(),
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)