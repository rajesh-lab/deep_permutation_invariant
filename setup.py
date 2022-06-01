from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setup(name='deep_permutation_invariant',
      version='1.0.0',
      description='A library for deep permutation invariant neural networks',
      long_description='A library for the decomposition of a matrix of signals.'
      url='https://github.com/veronicatozzo/deep_permutation_invariant',
      author='Veronica Tozzo, Lily Zhang',
      author_email='vtozzo@mgh.harvard.edu, lily.h.zhang@nyu.edu',
      license='FreeBSD',
      classifiers={
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Programming Language :: Python',
          'License :: OSI Approved :: MIT License',
          'Operating System :: POSIX',
          'Operating System :: Unix'},
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      )