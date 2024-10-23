import re
import os
from setuptools import setup, find_packages


# =============================================================================
# helper functions to extract meta-info from package
# =============================================================================
def read_version_file(*parts):
    return open(os.path.join(*parts), 'r').read()

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def find_version(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

def find_name(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__name__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find name string.")

def find_author(*file_paths):
    version_file = read_version_file(*file_paths)
    version_match = re.search(r"^__author__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find author string.")


# =============================================================================
# setup    
# =============================================================================
setup(
    name = find_name("RCWA_project", "__init__.py"),
    version = find_version("RCWA_project", "__init__.py"),
    author = find_author("RCWA_project", "__init__.py"),
    author_email='denis.langevin@uca.fr',
    license='GPLv3+',
    packages=['RCWA_project'],
    include_package_data=True,   # add data folder containing material dispersions
    description = ("A Rigorous Coupled Wava Analysis library for optical simulations."),
    long_description=read('README.md'),
    url='https://github.com/Milloupe/Code_RCWA',
    keywords=['RCWA','Maxwell','Optics','Multilayers','Plasmonics','Photovoltaics'],
    install_requires=[
          'numpy','matplotlib'
      ],

)
