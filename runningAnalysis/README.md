# Cell profiler Bacteria Tracking Plot

The `cellProfilerBacteriaTrackingPlot.py` script generates tracking plots for bacteria over different time steps using data from CSV file (output of CellProfiler) and raw images. </br>
The plots visualize the lineage life history of cells in each time step on a background image.

## Usage
The script requires several input arguments, which can be provided through the command line:
- `-t`, `--cellprofiler_csv_output_file`:
Path to the CSV file output generated by Cellprofiler. This file should contain tracking data and other relevant information for each cell at each time step.
- `-r`, `--raw_image_dir`:
Directory containing the raw images corresponding to each time step. The images should be named in a sequential format.</br>
<b>Note about the order of images</b></br>
The script relies on the images being read in the correct sequential order. To ensure this, the image filenames should be formatted in such a way that they are sorted properly by Python's default sorting mechanism. For example, filenames like 01.tif, 02.tif, ..., 10.tif, 11.tif, ... will work correctly. Similarly, if there are 100 or more images, use three digits (001.tif, 002.tif, ..., 100.tif). Avoid naming files in a non-padded format like 1.tif, 2.tif, ..., 10.tif because they may not be sorted in the correct order (e.g., 1.tif, 10.tif, 2.tif). The image naming pattern can vary, but it's essential to ensure they are in the correct order.

- `-o`, `--output`:
Directory where the output tracking plots should be saved. If not provided, the plots will be saved in a subdirectory named `Cellprofiler_tracking_plot` in the same directory as the Cellprofiler CSV file.
- `-c`, `--objectColor`:
Color of the objects (cells) in the tracking plots. Colors should be provided in hexadecimal format.</br>
Default Value: #56e64e (green)
- `-f`, `--font_size`:
Font size for labeling information (cell IDs and parent IDs) on objects in the plots.</br>
Default Value: 1

## How to Run the Script from the Command Line
To run the script, use the following format:</br>
```
python cellProfilerBacteriaTrackingPlot.py -t <cellprofiler_csv_output_file> -r <raw_image_dir> -o <output_dir> -c <objectColor> -f <font_size>
```


# Trackrefiner Bacteria Tracking Plot
The `trackrefinerBacteriaTrackingPlot.py` script generates tracking plots for bacteria over different time steps using data from CSV file (output of Trackrefiner) and raw images. </br>
The plots visualize the lineage life history of cells in each time step on a background image.

## Usage
The script requires several input arguments, which can be provided through the command line:
- `-t`, `--trackrefiner_csv_output_file`:
Path to the CSV file output generated by TrackRefiner. This file should contain tracking data, including cell IDs, parent IDs, orientations, and other relevant information for each cell at each time step.
- `-r`, `--raw_image_dir`:
Directory containing the raw images corresponding to each time step. The images should be named in a sequential format.</br>
<b>Note about the order of images</b></br>
The script relies on the images being read in the correct sequential order. To ensure this, the image filenames should be formatted in such a way that they are sorted properly by Python's default sorting mechanism. For example, filenames like 01.tif, 02.tif, ..., 10.tif, 11.tif, ... will work correctly. Similarly, if there are 100 or more images, use three digits (001.tif, 002.tif, ..., 100.tif). Avoid naming files in a non-padded format like 1.tif, 2.tif, ..., 10.tif because they may not be sorted in the correct order (e.g., 1.tif, 10.tif, 2.tif). The image naming pattern can vary, but it's essential to ensure they are in the correct order.
- `-o`, `--output`:
Directory where the output tracking plots should be saved. If not provided, the plots will be saved in a subdirectory named `Trackrefiner_tracking_plot` in the same directory as the TrackRefiner CSV file.
- `-u`, `--umPerPixel`:
Conversion factor from pixels to micrometers. This value is used to scale the cell sizes and positions correctly in the plots.</br>
Default Value: 0.144
- `-c`, `--objectColor`:
Color of the objects (cells) in the tracking plots. Colors should be provided in hexadecimal format.</br>
Default Value: #56e64e (green)
- `-f`, `--font_size`:
Font size for labeling information (cell IDs and parent IDs) on objects in the plots.</br>
Default Value: 1

## How to Run the Script from the Command Line
To run the script, use the following format:</br>
```
python trackrefinerBacteriaTrackingPlot.py -t <trackrefiner_csv_output_file> -r <raw_image_dir> -o <output_dir> -u <umPerPixel> -c <objectColor> -f <font_size>
```
