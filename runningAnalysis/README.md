# Cell profiler Bacteria Tracking Plot

The `cellProfilerBacteriaTrackingPlot.py` script generates tracking plots for bacteria over different time steps using data from CSV file (output of CellProfiler) and raw images. </br>
The plots visualize the lineage life history of cells in each time step on a background image.

## Usage
Specify the following parameters in the <a href='cellProfilerBacteriaTrackingPlot.py#L198'>main</a> function:
- `cp_output_csv_file`: Path to the CSV file containing bacteria data.
- `raw_img_dir`: Directory where the raw images are stored.
- `output_dir`: Directory where the output plots will be saved.
- `prefix_raw_name`: Prefix for the raw image files.
- `postfix_raw_name`: Postfix for the raw image files.
- `color`: Color used for plotting.
- `font_size`: Font size for annotations.

### Details about prefix_raw_name and postfix_raw_name
The `prefix_raw_name` and `postfix_raw_name` parameters are used to construct the filenames of the raw images that correspond to each time step. </br>The script generates the filenames based on the time step number, formatted to match the naming convention of the raw image files.
</br></br>
For example, if the raw images are named K12_Scene2_C0_T01.tif, K12_Scene2_C0_T02.tif, etc.:
- prefix_raw_name should be K12_Scene2_C0_T
- postfix_raw_name should be .tif
</br>
The script constructs the image filename for a given time step by combining the prefix, the zero-padded time step number (based on the length of the highest time step number), and the postfix.
</br></br>
<b>Note about the order of images</b></br>
The script assumes that the images are ordered sequentially by time step. It constructs the image filenames based on the time step numbers extracted from the CSV file. The filenames must follow a consistent pattern to ensure the correct image is used for each time step.
