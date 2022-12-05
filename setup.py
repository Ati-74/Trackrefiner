#!/usr/bin/env python

from setuptools import setup

setup(name='CellProfilerAnalysis',
      version='1.0.0',
      description='analyzing CellProfiler output',
      install_requires=['numpy', 'scipy', 'pandas', 'sklearn', 'scikit-learn'],
      setup_requires=['numpy', 'scipy', 'pandas', 'sklearn', 'scikit-learn'],
      packages=['CellProfilerAnalysis', 'CellProfilerAnalysis.strain'],
      python_requires='>=3',
      url='https://github.com/Ati-74/CellProfilerAnalysis',
      author='Atiyeh Ahmadi',
      author_email='a94ahmad@uwaterloo.ca',
      license=' BSD-3-Clause',
      )
