from setuptools import setup, find_packages
from os.path import join, isfile
from os import walk
import model_zoo


def read_file(filename):
    with open(filename) as fp:
        return fp.read().strip()


def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


def package_files(directories):
    paths = []
    for item in directories:
        if isfile(item):
            paths.append(join('..', item))
            continue
        for (path, directories, filenames) in walk(item):
            for filename in filenames:
                paths.append(join('..', path, filename))
    return paths


setup(
    name='model-zoo',
    version=model_zoo.version(),
    description='A framework to help you build model much more easily',
    keywords=['model_zoo', 'tensorflow', 'keras'],
    author='germey',
    author_email='cqc@cuiqingcai.com',
    url='http://pypi.python.org/pypi/model-zoo/',
    license='MIT',
    install_requires=read_requirements('requirements.txt'),
    packages=find_packages(),
    package_data={
        '': package_files([
            'model_zoo/VERSION'
        ])
    },
    publish=[
        'sudo python3 setup.py bdist_egg',
        'sudo python3 setup.py sdist',
        'sudo python3 setup.py bdist_egg upload',
        'sudo python3 setup.py sdist upload',
        'twine upload dist/* '
    ]
)
