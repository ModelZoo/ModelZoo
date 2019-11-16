from os.path import abspath, dirname
from tensorflow_core.python.keras import datasets
from . import flags
from .framework import Framework
from . import preprocess

Model = Framework
version_file = dirname(abspath(__file__)) + '/VERSION'


def version():
    return open(version_file).read().strip()


__all__ = ['flags', 'version', 'datasets']
