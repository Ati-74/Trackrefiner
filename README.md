# CellProfilerAnalysis
Codes used to analyze CellProfiler output


## Converting CellProfiler data similar to CellModeller:
<div align="justify">
The process can be done in the command line.

## How can I install the CellProfilerAnalysis package?
1. Download the most recent version of the package. For beta versions, check the development branch.
2. Extract the files and navigate to the folder where you downloaded the package
3. Execute the following command:
```
python -e .
```

Additionally, you can set it up in a new conda environment using these commands:
```
conda env create -f environment.yml
python -e .
```

## Required dependencies for this package include:
- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- opencv-python
