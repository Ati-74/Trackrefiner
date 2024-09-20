#!/usr/bin/env python

from setuptools import setup

setup(name='TrackRefiner',
      version='1.1.1.b',
      description='analyzing CellProfiler output',
      install_requires=['numpy==1.26.4', 'scipy', 'pandas >= 2.2.2', 'scikit-learn', 'matplotlib', 'opencv-python',
                        'scikit-image', 'psutil'],
      setup_requires=['numpy==1.26.4', 'scipy', 'pandas >= 2.2.2', 'scikit-learn', 'matplotlib', 'opencv-python',
                      'scikit-image', 'psutil'],
      packages=['Trackrefiner', 'Trackrefiner.strain', 'Trackrefiner.strain.correction',
                'Trackrefiner.strain.correction.action'],
      python_requires='>=3',
      url='https://github.com/Ati-74/CellProfilerAnalysis',
      author='Atiyeh Ahmadi',
      author_email='a94ahmad@uwaterloo.ca',
      license=' BSD-3-Clause',
      )
