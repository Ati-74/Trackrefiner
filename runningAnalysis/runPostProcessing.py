import argparse
from CellProfilerAnalysis.strain.processCellProfilerData import process_data

if __name__ == '__main__':
    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description='analyzing CellProfiler output')

    # Add arguments
    parser.add_argument('-i', '--input', help='This parameter takes the output csv file from "CP" which'
                                              ' contains information about measured features of bacteria,'
                                              ' such as length, orientation, etc., as well as tracking information.')

    parser.add_argument('-np', '--npy', help='This folder contains files in the npy format, which are the results of'
                                             ' segmentation, where the pixels of an object are unified in a specific '
                                             'color.')

    parser.add_argument('-r', '--neighbor', help='CSV file containing neighboring data of bacteria.')

    parser.add_argument('-o', '--output', default=None,
                        help="Where to save package output. Default value: save to the CP output file folder")

    parser.add_argument('-it', '--intervalTime', default=1.5,
                        help="interval time (unit: minute). Default value: 1.5 min")

    parser.add_argument('-m', '--minLifeHistory', default=3,
                        help=" min life history of bacteria. Default value: 3 min")

    parser.add_argument('-g', '--growthRateMethod', default="Average",
                        help="growth rate method. The value of this parameter can be `Average` or `Linear Regression`."
                             " Default value: Average")

    parser.add_argument('-n', '--gap', default=0, help="number of gap. Default value: 0")

    parser.add_argument('-u', '--umPerPixel', default=0.144, help="convert pixel to um. Default value: 0.144")

    parser.add_argument('-a', '--cellType', default=True, help="assigning cell type. Default value: True")

    parser.add_argument('-t', '--intensityThreshold', default=0.1, help="intensity threshold. "
                                                                        "This parameter is related to assigning the"
                                                                        " cell type. Default value: 0.1")

    # Parse the arguments
    args = parser.parse_args()

    input_file = args.input
    npy_files_dir = args.npy
    neighbors_file = args.neighbor
    output_directory = args.output

    # optional parameters
    # unit: minute
    interval_time = args.intervalTime
    min_life_history_of_bacteria = args.minLifeHistory

    # `Average` or `Linear Regression`
    growth_rate_method = args.growthRateMethod

    # useful for fixing tracking errors
    number_of_gap = args.gap

    # convert pixel to um
    um_per_pixel = args.umPerPixel

    assigning_cell_type = args.cellType

    # This parameter is related to assigning the cell type
    intensity_threshold = args.intensityThreshold

    # run post-processing
    process_data(input_file=input_file, npy_files_dir=npy_files_dir, neighbors_file=neighbors_file,
                 output_directory=output_directory, interval_time=interval_time, growth_rate_method=growth_rate_method,
                 number_of_gap=number_of_gap, um_per_pixel=um_per_pixel, intensity_threshold=intensity_threshold,
                 assigning_cell_type=assigning_cell_type, min_life_history_of_bacteria=min_life_history_of_bacteria)