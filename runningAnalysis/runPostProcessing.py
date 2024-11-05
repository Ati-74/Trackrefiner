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

    parser.add_argument('-it', '--intervalTime',
                        help="interval time (unit: minute)")

    parser.add_argument('-m', '--minLifeHistory',
                        help=" min life history of bacteria.")

    parser.add_argument('-g', '--growthRateMethod', default="Average",
                        help="growth rate method. The value of this parameter can be `Average` or `Linear Regression`."
                             " Default value: Average")

    parser.add_argument('-u', '--umPerPixel', default=0.144, help="convert pixel to um. "
                                                                  "Default value: 0.144")

    parser.add_argument('-a', '--cellType', default='True', help="assigning cell type. Default value: True."
                                                                 "(Note: The value of this argument should be "
                                                                 "T (or True) or F (or False).")

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

    parser.add_argument('-ba', '--boundary_limits', default=None,
                        help="To identify objects that originate from the walls of an image, "
                             "you only need to define the boundary limits for detection. Here’s how it works:"
                             "Specify the Boundary Limits:"
                             "Lower X Limit: Minimum X coordinate for objects to be considered as bacteria."
                             "Upper X Limit: Maximum X coordinate for objects to be considered as bacteria."
                             "Lower Y Limit: Minimum Y coordinate for objects to be considered as bacteria."
                             "Upper Y Limit: Maximum Y coordinate for objects to be considered as bacteria."
                             "Objects are recognized as walls if their center’s X coordinate is less than the "
                             "lower X limit or greater than the upper X limit."
                             "Similarly, objects are considered walls if their center’s Y coordinate is less than "
                             "the lower Y limit or greater than the upper Y limit."
                             "Example: `0, 112, 52, 323`"
                             "If you set the limits as 0 for the lower X, 112 for the upper X, 52 for the lower Y, "
                             "and 323 for the upper Y, then any object with its center’s X or Y coordinates outside "
                             "these ranges will be identified as originating from the walls of the image.")

    parser.add_argument('-b', '--boundary_limits_per_time_step', default=None,
                        help="To identify objects that originate from the walls of an image, "
                             "you only need to define the boundary limits for each time step to enable detection. "
                             "Here’s how it works:"
                             "For each time step, specify boundary limits in a CSV file with the following columns:"
                             "`Time Step`, `Lower X Limit`, `Upper X Limit`, `Lower Y Limit`, `Upper Y Limit`."
                             "Lower X Limit: Minimum X coordinate for objects to be considered as bacteria."
                             "Upper X Limit: Maximum X coordinate for objects to be considered as bacteria."
                             "Lower Y Limit: Minimum Y coordinate for objects to be considered as bacteria."
                             "Upper Y Limit: Maximum Y coordinate for objects to be considered as bacteria."
                             "Objects are recognized as walls if their center’s X coordinate is less than the "
                             "lower X limit or greater than the upper X limit."
                             "Similarly, objects are considered walls if their center’s Y coordinate is less than "
                             "the lower Y limit or greater than the upper Y limit."
                             "Example: `0, 112, 52, 323`"
                             "If you set the limits as 0 for the lower X, 112 for the upper X, 52 for the lower Y, "
                             "and 323 for the upper Y, then any object with its center’s X or Y coordinates outside "
                             "these ranges will be identified as originating from the walls of the image.")

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

    # convert pixel to um
    um_per_pixel = float(args.umPerPixel)

    assigning_cell_type = args.cellType

    # This parameter is related to assigning the cell type
    intensity_threshold = float(args.intensityThreshold)

    warn = args.warn
    without_tracking_correction = args.withoutTrackingCorrection

    boundary_limits = args.boundary_limits
    boundary_limits_per_time_step = args.boundary_limits_per_time_step

    if assigning_cell_type in ['T', 'True']:
        assigning_cell_type = True
    else:
        assigning_cell_type = False

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
                 um_per_pixel=um_per_pixel, intensity_threshold=intensity_threshold,
                 assigning_cell_type=assigning_cell_type, min_life_history_of_bacteria=min_life_history_of_bacteria,
                 warn=warn, without_tracking_correction=without_tracking_correction, clf=clf, n_cpu=n_cpu,
                 boundary_limits=boundary_limits, boundary_limits_per_time_step=boundary_limits_per_time_step)
