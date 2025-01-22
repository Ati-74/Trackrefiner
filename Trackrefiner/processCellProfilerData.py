import pandas as pd
import time
import pickle
import os
import glob
from Trackrefiner.bacterialLifeHistoryAnalysis import process_bacterial_life_and_family
from Trackrefiner.correction.findFixTrackingErrors import find_fix_tracking_errors
from Trackrefiner.correction.action.helper import identify_important_columns, print_progress_bar
from Trackrefiner.correction.action.drawDistributionPlot import draw_feature_distribution
import psutil
import threading


def monitor_system_usage(stats, interval=1, stop_event=None):
    process = psutil.Process()
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / (1024 * 1024 * 1024)  # Convert memory to GB

        # Store CPU and memory usage in a shared list
        stats['cpu'].append(cpu_usage)
        stats['memory'].append(memory_usage)

        time.sleep(interval)  # Interval to report CPU and memory usage


class Dict2Class(object):

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def write_log_file(log_list, path):

    log_list = [v for v in log_list if v != '']

    # Open a file for writing
    with open(f'{path}log.txt', 'w') as file:
        # Write each element to the file
        for element in log_list:
            file.write(element + '\n')


def create_pickle_files(df, path, assigning_cell_type):

    lineage = {}
    time_steps = sorted(df['stepNum'].unique())
    for t in time_steps:
        lineage_data_frame = df.loc[(df["stepNum"] == t) & df["parent_id"]]
        lineage.update(dict(zip(lineage_data_frame["id"].values, lineage_data_frame["parent_id"].values)))
        data = {}
        data_frame_current_time_step = df.loc[df["stepNum"] == t]
        data["stepNum"] = t
        data["lineage"] = lineage
        # start index from 1
        data_frame_current_time_step.index = data_frame_current_time_step['id']

        # select important columns
        if assigning_cell_type:
            data_frame_current_time_step = data_frame_current_time_step[
                ['ObjectNumber', 'id', 'label', 'cellType', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory',
                 'startVol', 'targetVol', 'pos', 'time', 'radius', 'length', 'dir', 'ends', 'strainRate',
                 'strainRate_rolling']]
        else:
            data_frame_current_time_step = data_frame_current_time_step[
                ['ObjectNumber', 'id', 'label', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol',
                 'targetVol', 'pos', 'time', 'radius', 'length', 'dir', 'ends', 'strainRate', 'strainRate_rolling']]
        # convert to dictionary
        df_to_dict = data_frame_current_time_step.to_dict('index')
        # change dictionary to class
        for dictionary_key in df_to_dict.keys():
            df_to_dict[dictionary_key] = Dict2Class(df_to_dict[dictionary_key])
        data["cellStates"] = df_to_dict
        write_to_pickle_file(data, path, t)


def write_to_pickle_file(data, path, time_step):

    if not os.path.exists(path):
        os.mkdir(path)

    output_file = f"{path}Trackrefiner.step-{time_step:06}.pickle"

    with open(output_file, 'wb') as export:
        pickle.dump(data, export, protocol=-1)


def process_objects_data(cp_output_csv, segmentation_res_dir, neighbor_csv, interval_time, doubling_time,
                         elongation_rate_method, pixel_per_micron, assigning_cell_type, intensity_threshold,
                         disable_tracking_correction, clf, n_cpu, image_boundaries, dynamic_boundaries, out_dir,
                         save_npy, verbose, command):

    """
    Processes CellProfiler output data, performs tracking correction, assigns cell types,
    and calculates additional features related to the life history of bacteria.


    :param str cp_output_csv:
        Path to the CellProfiler output file in CSV format, which contains measured bacterial features.
    :param str segmentation_res_dir:
        Path to the directory containing npy files (segmentation results exported from the CellProfiler pipeline).
    :param str neighbor_csv:
        Path to a CSV file containing information about the neighbors of each object.
    :param str out_dir:
        Path to the directory where processed output files will be saved.
    :param float interval_time:
        Time interval between consecutive images (in minutes).
    :param float doubling_time:
        Minimum life history duration of bacteria (in minutes) for analysis.
    :param str elongation_rate_method:
        Method for calculating the elongation rate. Options:
        - 'Average': Computes average growth rate.
        - 'Linear Regression': Estimates growth rate using linear regression.
    :param float pixel_per_micron:
        Conversion factor for pixels to microns (e.g., default value is 0.144).
    :param bool assigning_cell_type:
        If True, assigns cell types to objects based on intensity thresholds.
    :param float intensity_threshold:
        Threshold to classify objects into specific cell types based on intensity values.
    :param bool disable_tracking_correction:
        If True, disables tracking correction and performs calculations directly on the CellProfiler output data.
    :param str clf:
        Name of the classifier to be used for model training during tracking correction.
        Options: 'LogisticRegression', 'GaussianProcessClassifier', 'C-Support Vector Classifier'.
    :param int n_cpu:
        Number of CPU cores to be used for parallel computing.
        - `-1` indicates that all available CPUs will be utilized.
    :param str image_boundaries:
        Boundary limits defined for all time steps to filter out objects outside the image area.
    :param str dynamic_boundaries:
        Path to a CSV file specifying boundary limits for each time step. The file should contain columns:
        `Time Step`, `Lower X Limit`, `Upper X Limit`, `Lower Y Limit`, `Upper Y Limit`.
    :param bool save_npy:
        If True, results are saved in `.npy` format.
    :param bool verbose:
        If True, displays warnings and additional details during processing.
    :param str command:
        User's Command Statement

    :returns:
        None. Processed data and outputs are saved to the specified ``out_dir``.

    :rtype: None

    """

    log_list = []

    # Shared dictionary to store CPU and memory usage stats
    stats = {
        'cpu': [],  # List to store CPU usage over time
        'memory': []  # List to store memory usage over time
    }

    # Set up an event to stop the monitoring thread when execution is done
    stop_event = threading.Event()

    # Start the monitoring in a separate thread
    monitoring_thread = threading.Thread(target=monitor_system_usage, args=(stats, 1, stop_event))
    monitoring_thread.start()

    try:
        # Recording the start time
        start_time = time.time()
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        start_time_log = f"Trackrefiner Processing started at: {start_time_str}\n"
        print(start_time_log)
        log_list.append(start_time_log)

        command_log = f"User's Command: {command}\n"
        print(command_log)
        log_list.append(command_log)

        # Initial call to print 0% progress
        print_progress_bar(0, prefix='Progress:', suffix='', length=50)

        cp_output_df = pd.read_csv(cp_output_csv)

        (center_coord_cols, all_rel_center_coord_cols, parent_image_number_col, parent_object_number_col,
         label_col) = identify_important_columns(cp_output_df)

        # Check if tracking correction is disabled; if so, skip neighbor file and segmentation files loading

        if not os.path.exists(neighbor_csv):
            raise FileNotFoundError(f"Neighbors CSV file not found: {neighbor_csv}")

        neighbors_df = pd.read_csv(neighbor_csv)
        neighbors_df = neighbors_df.loc[
            neighbors_df['Relationship'] == 'Neighbors'][['First Image Number', 'First Object Number',
                                                          'Second Image Number', 'Second Object Number']]

        if not disable_tracking_correction:

            if segmentation_res_dir is not None:
                sorted_seg_npy_files_list = sorted(glob.glob(segmentation_res_dir + '/*.npy'))
            else:
                sorted_seg_npy_files_list = []

            if len(sorted_seg_npy_files_list) == 0:
                raise FileNotFoundError(f"Segmentation files in npy format not found: {segmentation_res_dir}")
        else:
            sorted_seg_npy_files_list = []

        if ((len(sorted_seg_npy_files_list) > 0 and neighbors_df.shape[0] > 0) or
                (disable_tracking_correction and neighbors_df.shape[0] > 0)):

            if out_dir is not None:
                out_dir = f"{out_dir}/"
            else:
                # Create the directory if it does not exist
                out_dir = os.path.dirname(cp_output_csv)
                out_dir = f"{out_dir}/Trackrefiner/"
                os.makedirs(out_dir, exist_ok=True)

            (cp_output_df, find_fix_errors_log, logs_df, neighbors_df, cell_type_array) = \
                find_fix_tracking_errors(cp_output_df=cp_output_df, sorted_seg_npy_files_list=sorted_seg_npy_files_list,
                                         neighbors_df=neighbors_df, center_coord_cols=center_coord_cols,
                                         all_rel_center_coord_cols=all_rel_center_coord_cols,
                                         parent_image_number_col=parent_image_number_col,
                                         parent_object_number_col=parent_object_number_col, label_col=label_col,
                                         pixel_per_micron=pixel_per_micron, intensity_threshold=intensity_threshold,
                                         assigning_cell_type=assigning_cell_type, interval_time=interval_time,
                                         doubling_time=doubling_time,
                                         disable_tracking_correction=disable_tracking_correction,
                                         clf=clf, n_cpu=n_cpu, image_boundaries=image_boundaries,
                                         dynamic_boundaries=dynamic_boundaries, out_dir=out_dir, verbose=verbose)

            log_list.extend(find_fix_errors_log)

            # process the tracking data
            processed_df, processed_df_with_specific_cols = \
                process_bacterial_life_and_family(cp_output_df, interval_time, elongation_rate_method, assigning_cell_type,
                                                  cell_type_array, label_col, center_coord_cols)

            sel_cols = [col for col in processed_df.columns if col not in
                        ['noise_bac', 'mother_rpl', 'daughter_rpl', 'source_mcl', 'target_mcl', 'prev_index',
                         'checked', 'bad_division_flag', 'ovd_flag', 'bad_daughters_flag', 'pos', 'ends',
                         'prev_id', 'prev_parent_id', 'prev_label', 'age', 'parent_index', 'index',
                         'daughters_index', 'other_daughter_index', 'prev_time_step_index']]

            excluded_patterns = [
                'TrackObjects_Displacement', 'TrackObjects_DistanceTraveled',
                'TrackObjects_FinalAge', 'TrackObjects_IntegratedDistance',
                'TrackObjects_Lifetime', 'TrackObjects_Linearity',
                'TrackObjects_TrajectoryX', 'TrackObjects_TrajectoryY'
            ]

            sel_cols = [col for col in sel_cols if not any(pattern in col for pattern in excluded_patterns)]

            processed_df = processed_df[sel_cols]

            print_progress_bar(10, prefix='Progress:', suffix='', length=50)

            if save_npy:
                create_pickle_files(processed_df_with_specific_cols, out_dir, assigning_cell_type)

            cp_out_base_name = os.path.basename(cp_output_csv).split('.')[0]
            # Common prefix for all output paths
            prefix = f"{out_dir}/Trackrefiner.{cp_out_base_name}_{elongation_rate_method}"

            path = f"{prefix}_analysis"
            path_logs = f"{prefix}_logs"
            path_neighbors = f"{prefix}_neighbors"

            # write to csv
            processed_df.to_csv(f"{path}.csv", index=False)

            if not disable_tracking_correction:
                logs_df.to_csv(f'{path_logs}.csv', index=False)
                neighbors_df.to_csv(f'{path_neighbors}.csv', index=False)

            output_log = f"Output Directory: {out_dir}"
            print(output_log)
            log_list.append(output_log)

            # draw plots
            features_dict = {'LifeHistory': ["Cell Cycle Duration (min)", 'Frequency', 'lifehistory_based'],
                             'startVol': ["Birth Length (um)", 'Frequency', 'lifehistory_based'],
                             'targetVol': ["Final Length (um)", 'Frequency', 'lifehistory_based'],
                             'elongationRate': ["Elongation Rate (um/min)", 'Frequency', 'lifehistory_based'],
                             'velocity': ["Velocity (um/min)", 'Frequency', 'lifehistory_based'],
                             'AverageLength': ["Average Length (um)", 'Frequency', 'lifehistory_based'],
                             'NumberOfDivisionFamily': ["Number of division per family", 'Number of Families',
                                                        'family_based'],
                             'AreaShape_MajorAxisLength': ["Length (um)", 'Frequency', 'bacteria_based']}
            draw_feature_distribution(processed_df, features_dict, label_col, interval_time, doubling_time, out_dir)

        else:
            log = "The segmentation result folder or neighbor file or both are not available!"
            print(log)
            log_list.append(log)

    finally:
        # Stop the monitoring thread after all functions have executed
        stop_event.set()
        monitoring_thread.join()

        # Calculate maximum and average CPU and memory usage
        max_cpu = max(stats['cpu']) if stats['cpu'] else 0
        avg_cpu = sum(stats['cpu']) / len(stats['cpu']) if stats['cpu'] else 0
        max_memory = max(stats['memory']) if stats['memory'] else 0
        avg_memory = sum(stats['memory']) / len(stats['memory']) if stats['memory'] else 0

    end_time = time.time()
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))

    end_time_log = f"Trackrefiner Process completed at: {end_time_str}"
    print(end_time_log)
    log_list.append(end_time_log)

    execution_time = end_time - start_time

    # Conversion to the respective time units
    days, remainder = divmod(execution_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    duration_log = f"Execution Time: {int(days)}:{int(hours)}:{int(minutes)}:{seconds} (day:hours:minutes:seconds)"
    print(duration_log)

    log_list.append(duration_log)

    resource_usage = "\nResource Usage Summary:"
    max_cpu_usage = f"Maximum CPU Usage: {max_cpu:.2f}%"
    avg_cpu_usage = f"Average CPU Usage: {avg_cpu:.2f}%"
    max_mem_usage = f"Maximum Memory Usage: {max_memory:.2f} GB"
    avg_mem_usage = f"Average Memory Usage: {avg_memory:.2f} GB"
    log_list.append(resource_usage)
    log_list.append(max_cpu_usage)
    log_list.append(avg_cpu_usage)
    log_list.append(max_mem_usage)
    log_list.append(avg_mem_usage)

    write_log_file(log_list, out_dir)
