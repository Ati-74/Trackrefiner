# Cell profiler Bacteria Tracking Plot

The `cellProfilerBacteriaTrackingPlot.py` script generates tracking plots for bacteria over different time steps using data from CSV file (output of CellProfiler) and raw images. </br>
The plots visualize the lineage life history of cells in each time step on a background image.

## Usage
Specify the following parameters in the main function:
`cp_output_csv_file`: Path to the CSV file containing bacteria data.
`raw_img_dir`: Directory where the raw images are stored.
`output_dir`: Directory where the output plots will be saved.
`prefix_raw_name`: Prefix for the raw image files.
`postfix_raw_name`: Postfix for the raw image files.
`color`: Color used for plotting.
`font_size`: Font size for annotations.
