import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup( # Finally, pass this all along to distutils to do the heavy lifting.
    name             = 'begepro',
    version          = '0.0.0',
    description      = 'Package for analyzing data from BEGe detectors',
    long_description = read('README.md'),
    author           = 'Andrea Giachero, Alessandro Paonessa, Massimiliano Nastasi ',
    author_email     = 'andrea.giachero@mib.infn.it',
    url              = 'https://github.com/giachero/BEGepro',
    license          = read('LICENSE'),
    install_requires = read('requirements.txt').splitlines(),
    package_dir      = {"": "src"}
    packages         = find_packages(where='src', exclude=('tests', 'notebooks'))
)
