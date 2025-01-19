# Trackrefiner
Analyzing CellProfiler output
<div align="justify">

## How do I execute the package?
To execute it, run the <a href="runningAnalysis/runPostProcessing.py">runPostProcessing.py</a> file in your terminal. To understand the script's arguments, you can execute the following script:
```
python runPostProcessing.py -h
```

## Inputs required to run this package
1. The output CSV file from "CP" which contains information about measured features of bacteria, such as length, orientation, etc., as well as tracking information. See an example <a href="examples/e.coli/FilterObjects.csv">here</a>.
2. The folder contains files in the `npy` format, which are the results of segmentation, where the pixels of an object are unified in a specific color. See an example <a href="examples/e.coli/objects">here</a>.
3. CSV file containing neighboring data of bacteria. See an example <a href="examples/e.coli/Object%20relationships.csv">here</a>.
