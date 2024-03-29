import os
import re
from setuptools import setup, find_packages


__pkg_name__ = 'bumblebee'


verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name=__pkg_name__,
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Pay Giesselmann',
    author_email='giesselmann@molgen.mpg.de',
    url='https://github.com/giesselmann/bumblebee',
    entry_points = {
        'console_scripts': [
            '{0} = {0}:main'.format(__pkg_name__)
        ]
    },
)
