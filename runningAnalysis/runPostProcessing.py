import argparse
from Trackrefiner.strain.processCellProfilerData import process_data

if __name__ == '__main__':
    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description='analyzing CellProfiler output')

    # Add arguments
    parser.add_argument('-i', '--input', help='This parameter takes the output csv file from "CP" which'
                                              ' contains information about measured features of bacteria,'
                                              ' such as length, orientation, etc., as well as tracking information.')

    #  DecisionTreeClassifier, GradientBoostingClassifier, ExtraTreeClassifier, LinearDiscriminantAnalysis
    # RandomForestClassifier, QuadraticDiscriminantAnalysis
    parser.add_argument('-c', '--clf', default='LogisticRegression',
                        help='classifier to be used for track refining. '
                             'classifiers name: LogisticRegression, GaussianProcessClassifier, '
                             'C-Support Vector Classifier')

    parser.add_argument('-np', '--npy',
                        help='This folder contains files in the npy format, which are the results of segmentation, '
                             'where the pixels of an object are unified in a specific color.')

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

    parser.add_argument('-u', '--umPerPixel', default=0.144, help="convert pixel to um. "
                                                                  "Default value: 0.144")

    parser.add_argument('-a', '--cellType', default=True, help="assigning cell type. Default value: True")

    parser.add_argument('-t', '--intensityThreshold', default=0.1, help="intensity threshold. "
                                                                        "This parameter is related to assigning the"
                                                                        " cell type. Default value: 0.1")

    parser.add_argument('-w', '--warn', default='True',
                        help="You will see all warnings if you set it to True. Default value: True "
                             "(Note: The value of this argument should be T (or True) or F (or False).)")

    parser.add_argument('-wi', '--withoutTrackingCorrection', default='False',
                        help="If you want the outputs without correction of tracking, set it to true. "
                             "Default value: False "
                             "(Note: The value of this argument should be T (or True) or F (or False).)")

    parser.add_argument('-z', '--n_cpu', default=-1, help="The number of CPUs employed for parallel computing. "
                                                          "A value of -1 indicates the utilization of all available "
                                                          "CPUs for parallel processing. Default value: -1")

    # Parse the arguments
    args = parser.parse_args()

    input_file = args.input
    npy_files_dir = args.npy
    neighbors_file = args.neighbor
    output_directory = args.output

    # optional parameters
    # unit: minute
    interval_time = float(args.intervalTime)
    min_life_history_of_bacteria = float(args.minLifeHistory)

    # `Average` or `Linear Regression`
    growth_rate_method = args.growthRateMethod

    # useful for fixing tracking errors
    number_of_gap = int(args.gap)

    # convert pixel to um
    um_per_pixel = float(args.umPerPixel)

    assigning_cell_type = args.cellType

    # This parameter is related to assigning the cell type
    intensity_threshold = float(args.intensityThreshold)

    warn = args.warn
    without_tracking_correction = args.withoutTrackingCorrection

    if warn in ['T', 'True']:
        warn = True
    else:
        warn = False

    if without_tracking_correction in ['T', 'True']:
        without_tracking_correction = True
    else:
        without_tracking_correction = False

    clf = args.clf.rstrip().lstrip()
    n_cpu = int(args.n_cpu)

    # run post-processing
    process_data(input_file=input_file, npy_files_dir=npy_files_dir, neighbors_file=neighbors_file,
                 output_directory=output_directory, interval_time=interval_time, growth_rate_method=growth_rate_method,
                 number_of_gap=number_of_gap, um_per_pixel=um_per_pixel, intensity_threshold=intensity_threshold,
                 assigning_cell_type=assigning_cell_type, min_life_history_of_bacteria=min_life_history_of_bacteria,
                 warn=warn, without_tracking_correction=without_tracking_correction, clf=clf, n_cpu=n_cpu)
