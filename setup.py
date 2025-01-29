#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='TrackRefiner',
      version='1.2.1',
      description='analyzing CellProfiler output',
      install_requires=['numpy==1.26.4', 'scipy', 'pandas >= 2.2.2', 'scikit-learn', 'matplotlib', 'opencv-python',
                        'scikit-image', 'psutil', 'seaborn', 'PyQt5'],
      setup_requires=['numpy==1.26.4', 'scipy', 'pandas >= 2.2.2', 'scikit-learn', 'matplotlib', 'opencv-python',
                      'scikit-image', 'psutil', 'seaborn', 'PyQt5'],
      packages=find_packages(include=["Trackrefiner", "Trackrefiner.*"]),
      python_requires='>=3.8',
      url='https://github.com/ingallslab/Trackrefiner',
      author='Atiyeh Ahmadi',
      author_email='a94ahmad@uwaterloo.ca',
      license='BSD-3-Clause',
      entry_points={
          'console_scripts': [
              'trackrefiner-cli=Trackrefiner.cli:main',
              'trackrefiner-gui=Trackrefiner.gui:main',
              'trackrefiner-jitter-remover=Trackrefiner.utils.jitterRemover:main',
          ],
      }, project_urls={
        "Homepage": "https://github.com/ingallslab/Trackrefiner",
        "Repository": "https://github.com/ingallslab/Trackrefiner",
        "Tutorial": "https://github.com/ingallslab/Trackrefiner/wiki",
        "Documentation": "https://github.com/ingallslab/Trackrefiner/tree/main/docs/html",
        "FAQ": "https://github.com/ingallslab/Trackrefiner/issues",
        "Issues": "https://github.com/ingallslab/Trackrefiner/issues",
    },
      classifiers=[
          # Intended Audience
          "Intended Audience :: Science/Research",

          # Topics
          "Topic :: Scientific/Engineering",

          # Environment
          "Environment :: Console",
          "Environment :: X11 Applications :: Qt",

          # Programming Language and Python Versions
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3 :: Only",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Programming Language :: Python :: 3.12",

          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent"
      ],
    )
