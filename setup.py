#!/usr/bin/env python

from setuptools import setup

setup(name='TrackRefiner',
      version='1.1.1.b',
      description='analyzing CellProfiler output',
      install_requires=['numpy', 'scipy', 'pandas', 'scikit-learn', 'matplotlib', 'opencv-python'],
      setup_requires=['numpy', 'scipy', 'pandas', 'scikit-learn', 'matplotlib', 'opencv-python'],
      packages=['CellProfilerAnalysis', 'CellProfilerAnalysis.strain', 'CellProfilerAnalysis.strain.correction',
                'CellProfilerAnalysis.strain.correction.action'],
      python_requires='>=3',
      url='https://github.com/Ati-74/CellProfilerAnalysis',
      author='Atiyeh Ahmadi',
      author_email='a94ahmad@uwaterloo.ca',
      license=' BSD-3-Clause',
      )
