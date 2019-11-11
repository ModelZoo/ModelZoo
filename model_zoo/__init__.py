from os.path import abspath, dirname
from absl import flags

version_file = dirname(abspath(__file__)) + '/VERSION'


def version():
    return open(version_file).read().strip()


__all__ = ['flags', 'version']
