import argparse
import os
from Trackrefiner import process_objects_data

if __name__ == '__main__':
    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description='analyzing CellProfiler output')

    # Add arguments
    parser.add_argument('-i', '--cp_output_csv',
                        help='Path to the CP-generated CSV file containing measured bacterial features '
                             '(length, orientation, etc.) and tracking information.')

    parser.add_argument('-s', '--segmentation_results', default=None,
                        help='Path to folder containing .npy files generated from segmentation by CellProfiler, '
                             'where object pixels are unified in a specific color')

    parser.add_argument('-n', '--neighbor_csv',
                        help='Path to CSV file containing bacterial neighbor information')

    parser.add_argument('-t', '--interval_time',
                        help="Time interval between frames, in minutes.")

    parser.add_argument('-d', '--doubling_time',
                        help="Minimum lifespan of bacteria, in minutes")

    parser.add_argument('-e', '--elongation_rate_method', default="Average",
                        help="Method to calculate elongation rate. Options: `Average` or `Linear Regression`. "
                             "Default: `Average`.")

    parser.add_argument('-p', '--pixel_per_micron', default=0.144,
                        help="Conversion factor for pixels to micrometers. Default: `0.144` micrometers per pixel.")

    parser.add_argument('-a', '--assign_cell_type', action='store_false',
                        help="Assign cell type to objects. Default: Enabled. "
                             "To set it to False, include '-a' in the command.")

    parser.add_argument('-l', '--intensity_threshold', default=0.1,
                        help="Threshold for cell intensity used in cell type assignment. Default: `0.1`")

    parser.add_argument('-c', '--classifier', default='LogisticRegression',
                        help='Classifier for track refining. '
                             'Options: `LogisticRegression`, `GaussianProcessClassifier`, '
                             '`C-Support Vector Classifier`. Default: `LogisticRegression`.')

    parser.add_argument('-z', '--num_cpus', default=-1,
                        help="Number of CPUs for parallel processing. Use -1 for all available CPUs. Default: `-1`.")

    parser.add_argument('-b', '--boundary_limits', default=None,
                        help="Define boundary limits to exclude objects outside the image boundary. Specify:"
                             "\n- Lower X Limit, Upper X Limit: Minimum and maximum X coordinates for the boundary."
                             "\n- Lower Y Limit, Upper Y Limit: Minimum and maximum Y coordinates for the boundary."
                             "\nExample: `0, 112, 52, 323` means X: 0–112, Y: 52–323."
                             "\nObjects outside these ranges will be excluded.")

    parser.add_argument('-dy', '--dynamic_boundaries', default=None,
                        help="Define time-dependent boundary limits using a CSV file with the following columns:"
                             "\n- `Time Step`"
                             "\n- `Lower X Limit`, `Upper X Limit`: Minimum and maximum X coordinates for the boundary."
                             "\n- `Lower Y Limit`, `Upper Y Limit`: Minimum and maximum Y coordinates for the boundary."
                             "\nExample Row: `1, 0, 112, 52, 323` means X: 0–112, Y: 52–323 at Time Step 1."
                             "\nObjects outside these ranges at each time step will be excluded.")

    parser.add_argument('-dc', '--disable_tracking_correction', action='store_true',
                        help="Disable tracking correction on CellProfiler output. Default: Disabled. "
                             "To set it to True, include '-dc' in the command.")

    parser.add_argument('-o', '--output', default=None,
                        help="Output folder for results. Default: Same folder as the CellProfiler output file.")

    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Enable warnings and detailed messages. Default: Enabled. "
                             "To set it to True, include '-v' in the command.")

    parser.add_argument('-y', '--save_npy', action='store_true',
                        help="Save results in .npy format. Default: Disabled. "
                             "To enable saving as .npy, include '-y' in the command.")

    # Parse the arguments
    args = parser.parse_args()

    # Dynamically build the command string
    command = f"python {os.path.abspath(__file__)}"
    for arg, value in vars(args).items():
        if isinstance(value, bool):
            if value:  # Add the flag if it's True
                command += f" --{arg.replace('_', '-')}"
        elif value is not None:  # Add the argument and its value if not None
            command += f" --{arg.replace('_', '-')} {value}"

    cp_output_csv_file = args.cp_output_csv
    segmentation_results_dir = args.segmentation_results
    neighbor_csv = args.neighbor_csv
    out_dir = args.output

    # unit: minute
    interval_time = float(args.interval_time)
    doubling_time_of_bacteria = float(args.doubling_time)

    # `Average` or `Linear Regression`
    elongation_rate_method = args.elongation_rate_method

    # convert pixel to um
    pixel_per_micron = float(args.pixel_per_micron)

    assigning_cell_type = args.assign_cell_type

    # This parameter is related to assigning the cell type
    intensity_threshold = float(args.intensity_threshold)

    verbose = args.verbose

    disable_tracking_correction = args.disable_tracking_correction

    boundary_limits = args.boundary_limits
    dynamic_boundaries = args.dynamic_boundaries

    clf = args.classifier.rstrip().lstrip()
    n_cpu = int(args.num_cpus)
    save_npy = args.save_npy

    # run post-processing
    process_objects_data(cp_output_csv=cp_output_csv_file, segmentation_res_dir=segmentation_results_dir,
                         neighbor_csv=neighbor_csv, interval_time=interval_time,
                         elongation_rate_method=elongation_rate_method, pixel_per_micron=pixel_per_micron,
                         intensity_threshold=intensity_threshold,
                         assigning_cell_type=assigning_cell_type, doubling_time=doubling_time_of_bacteria,
                         disable_tracking_correction=disable_tracking_correction, clf=clf, n_cpu=n_cpu,
                         image_boundaries=boundary_limits, dynamic_boundaries=dynamic_boundaries, out_dir=out_dir,
                         save_npy=save_npy, verbose=verbose, command=command)
