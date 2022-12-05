
from CellProfilerAnalysis.strain.ProcessCellProfilerData import process_data

if __name__ == '__main__':
    input_file = '../examples/K12/K12_CP_output.csv'
    # input_file = '../examples/50-50/50-50_CP_output.csv'
    output_directory = '../examples/K12/outputs'
    # output_directory = '../examples/50-50/outputs'

    # optional parameters
    # unit: minute
    interval_time = 1.5
    growth_rate_method = "Average"
    # useful for fixing transition & more than two daughters errors
    number_of_gap = 0
    # convert pixel to um
    um_per_pixel = 0.144
    # for assign cell type
    intensity_threshold = 0.1

    # run post-processing
    process_data(input_file, output_directory, interval_time, growth_rate_method, number_of_gap, um_per_pixel,
                 intensity_threshold)
