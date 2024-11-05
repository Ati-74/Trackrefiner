import pandas as pd
import time
import pickle
import os
import glob
from Trackrefiner.strain.experimentalDataProcessing import bacteria_analysis_func
from Trackrefiner.strain.correction.findFixErrors import find_fix_errors
from Trackrefiner.strain.correction.action.helperFunctions import checking_columns, print_progress_bar
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


def process_data(input_file, npy_files_dir, neighbors_file, output_directory, interval_time, growth_rate_method,
                 um_per_pixel, intensity_threshold, assigning_cell_type, min_life_history_of_bacteria,
                 warn, without_tracking_correction, clf, n_cpu, boundary_limits, boundary_limits_per_time_step):
    """
    The main function that processes CellProfiler data.
    .pickle Files are exported to the same directory as input_file.
    
    @param interval_time        float   Time between images
    @param growth_rate_method   str     Method for calculating the growth rate
    @param input_file           str    CellProfiler output file in csv format
    @param output_directory     str
    @param um_per_pixel         float   um/pixel
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
        start_time_log = "Started at: " + start_time_str
        print(start_time_log)
        log_list.append(start_time_log)

        # Initial call to print 0% progress
        print_progress_bar(0, prefix='Progress:', suffix='Complete', length=50)

        if not without_tracking_correction:
            # Parsing CellProfiler output
            if npy_files_dir is not None:
                sorted_npy_files_list = sorted(glob.glob(npy_files_dir + '/*.npy'))
            else:
                sorted_npy_files_list = []
        else:
            sorted_npy_files_list = []

        data_frame = pd.read_csv(input_file)

        (center_coordinate_columns, all_center_coordinate_columns, parent_image_number_col, parent_object_number_col,
         label_col) = checking_columns(data_frame)

        if not without_tracking_correction:
            neighbors_df = pd.read_csv(neighbors_file)
            neighbors_df = neighbors_df.loc[
                neighbors_df['Relationship'] == 'Neighbors'][['First Image Number', 'First Object Number',
                                                              'Second Image Number', 'Second Object Number']]
        else:
            neighbors_df = pd.DataFrame()

        if (len(sorted_npy_files_list) > 0 and neighbors_df.shape[0] > 0) or without_tracking_correction:

            if output_directory is not None:
                output_directory = output_directory + "/"
            else:
                # Create the directory if it does not exist
                output_directory = os.path.dirname(input_file)
                os.makedirs(output_directory + '/Trackrefiner', exist_ok=True)
                output_directory = output_directory + '/Trackrefiner/'

            (data_frame, find_fix_errors_log, logs_df, identified_tracking_errors_df, fixed_errors, remaining_errors_df,
             neighbors_df, cell_type_array) = \
                find_fix_errors(data_frame, sorted_npy_files_list, neighbors_df, center_coordinate_columns,
                                all_center_coordinate_columns, parent_image_number_col, parent_object_number_col, label_col,
                                um_per_pixel=um_per_pixel, intensity_threshold=intensity_threshold,
                                check_cell_type=assigning_cell_type, interval_time=interval_time,
                                min_life_history_of_bacteria=min_life_history_of_bacteria, warn=warn,
                                without_tracking_correction=without_tracking_correction,
                                output_directory=output_directory, clf=clf, n_cpu=n_cpu,
                                boundary_limits=boundary_limits,
                                boundary_limits_per_time_step=boundary_limits_per_time_step)

            log_list.extend(find_fix_errors_log)

            start_calc_new_features_time = time.time()
            start_calc_new_features_time_str = \
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_calc_new_features_time))

            start_calc_new_features_time_log = "At " + start_calc_new_features_time_str + \
                                               ", the process of calculating new features for bacteria commenced."

            print(start_calc_new_features_time_log)
            log_list.append(start_calc_new_features_time_log)

            # process the tracking data
            processed_df, processed_df_with_specific_cols = \
                bacteria_analysis_func(data_frame, interval_time, growth_rate_method, assigning_cell_type,
                                       cell_type_array, label_col, center_coordinate_columns)

            end_calc_new_features_time = time.time()
            end_calc_new_features_time_str = \
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_calc_new_features_time))

            end_calc_new_features_time_log = "At " + end_calc_new_features_time_str + \
                                             ", the process of calculating new features for bacteria was completed."

            print(end_calc_new_features_time_log)
            log_list.append(end_calc_new_features_time_log)

            print_progress_bar(10, prefix='Progress:', suffix='Complete', length=50)

            create_pickle_files(processed_df_with_specific_cols, output_directory, assigning_cell_type)

            path = (output_directory + 'Trackrefiner.' + os.path.basename(input_file).split('.')[0] + "_" +
                    growth_rate_method + "_analysis")
            path_logs = (output_directory + 'Trackrefiner.' + os.path.basename(input_file).split('.')[0] + "_" +
                         growth_rate_method + "_logs")
            path_identified_tracking_errors = \
                (output_directory + 'Trackrefiner.' + os.path.basename(input_file).split('.')[0] + "_" +
                 growth_rate_method + "_identified_tracking_errors")
            path_fixed_errors = (output_directory + 'Trackrefiner.' + os.path.basename(input_file).split('.')[0] + "_" +
                                 growth_rate_method + "_fixed_errors")
            path_remaining_errors = (output_directory + 'Trackrefiner.' + os.path.basename(input_file).split('.')[0] + "_" +
                                     growth_rate_method + "_remaining_errors")
            path_neighbors = (output_directory + 'Trackrefiner.' + os.path.basename(input_file).split('.')[0] + "_" +
                              growth_rate_method + "_neighbors")

            # write to csv
            processed_df.to_csv(path + '.csv', index=False)
            logs_df.to_csv(path_logs + '.csv', index=False)
            identified_tracking_errors_df.to_csv(path_identified_tracking_errors + '.csv', index=False)
            fixed_errors.to_csv(path_fixed_errors + '.csv', index=False)
            remaining_errors_df.to_csv(path_remaining_errors + '.csv', index=False)

            if without_tracking_correction:
                neighbors_df.to_csv(path_neighbors + '.csv', index=False)

            output_log = "The outputs are written in the " + output_directory + " directory."
            print(output_log)
            log_list.append(output_log)

        else:
            log = "The npy folder or neighbor file or both are incorrect!"
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

    end_time_log = "Ended at: " + end_time_str
    print(end_time_log)
    log_list.append(end_time_log)

    execution_time = end_time - start_time

    # Conversion to the respective time units
    days, remainder = divmod(execution_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    duration_log = f"Program executed in: {int(days)}:{int(hours)}:{int(minutes)}:{seconds} (day:hours:minutes:seconds)"
    print(duration_log)

    log_list.append(duration_log)

    resource_usage = "Resource Usage Report:"
    max_cpu_usage = f"Maximum CPU Usage: {max_cpu:.2f}%"
    avg_cpu_usage = f"Average CPU Usage: {avg_cpu:.2f}%"
    max_mem_usage = f"Maximum Memory Usage: {max_memory:.2f} GB"
    avg_mem_usage = f"Average Memory Usage: {avg_memory:.2f} GB"
    log_list.append(resource_usage)
    log_list.append(max_cpu_usage)
    log_list.append(avg_cpu_usage)
    log_list.append(max_mem_usage)
    log_list.append(avg_mem_usage)

    write_log_file(log_list, output_directory)


class Dict2Class(object):
    """
        goal:  Change Dictionary Into Class

    """

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def write_log_file(log_list, path):
    log_list = [v for v in log_list if v != '']

    # Open a file for writing
    with open(path + 'log.txt', 'w') as file:
        # Write each element to the file
        for element in log_list:
            file.write(element + '\n')


def create_pickle_files(df, path, assigning_cell_type):
    """
    Saves processed data in a dictionary similar to CellModeller output style and exports as .pickle files
    @param df       data after being processed in experimentalDataProcessing.py
    @param path     str where .pickle files are saved
    """
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
    """
    Writes data to a .pickle file
    
    @param data         *       data to be stored
    @param path         str     path where data will be stored
    @param time_step    float   time between images
    """
    if not os.path.exists(path):
        os.mkdir(path)

    output_file = path + "Trackrefiner.step-" + '0' * (6 - len(str(time_step))) + str(time_step) + ".pickle"

    with open(output_file, 'wb') as export:
        pickle.dump(data, export, protocol=-1)
