import pandas as pd
import numpy as np
import time
import pickle
import os
import glob
from CellProfilerAnalysis.strain.experimentalDataProcessing import bacteria_analysis_func
from CellProfilerAnalysis.strain.correction.findFixErrors import find_fix_errors


def process_data(input_file, npy_files_dir, neighbors_file, output_directory, interval_time, growth_rate_method,
                 number_of_gap, um_per_pixel, intensity_threshold, assigning_cell_type, min_life_history_of_bacteria):
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

    # Recording the start time
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    start_time_log = "Started at: " + start_time_str
    print(start_time_log)
    log_list.append(start_time_log)

    # Parsing CellProfiler output
    if npy_files_dir is not None:
        sorted_npy_files_list = sorted(glob.glob(npy_files_dir + '/*.npy'))
    else:
        sorted_npy_files_list = []

    data_frame = pd.read_csv(input_file)
    neighbors_df = pd.read_csv(neighbors_file)
    neighbors_df = neighbors_df.loc[neighbors_df['Relationship'] == 'Neighbors']

    if len(sorted_npy_files_list) > 0 and neighbors_df.shape[0] > 0:

        data_frame, find_fix_errors_log, logs_df, neighbors_df = \
            find_fix_errors(data_frame, sorted_npy_files_list, neighbors_df,
                                                          number_of_gap=number_of_gap,
                                                          um_per_pixel=um_per_pixel,
                                                          intensity_threshold=intensity_threshold,
                                                          check_cell_type=assigning_cell_type,
                                                          interval_time=interval_time,
                                                          min_life_history_of_bacteria=min_life_history_of_bacteria)

        log_list.extend(find_fix_errors_log)

        start_calc_new_features_time = time.time()
        start_calc_new_features_time_str = \
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_calc_new_features_time))

        start_calc_new_features_time_log = "At " + start_calc_new_features_time_str + \
                                           ", the process of calculating new features for bacteria commenced."

        print(start_calc_new_features_time_log)
        log_list.append(start_calc_new_features_time_log)

        # process the tracking data
        processed_df, processed_df_with_specific_cols = bacteria_analysis_func(data_frame, interval_time,
                                                                               growth_rate_method, assigning_cell_type)

        end_calc_new_features_time = time.time()
        end_calc_new_features_time_str = \
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_calc_new_features_time))

        end_calc_new_features_time_log = "At " + end_calc_new_features_time_str + \
                                         ", the process of calculating new features for bacteria was completed."

        print(end_calc_new_features_time_log)
        log_list.append(end_calc_new_features_time_log)

        if output_directory is not None:
            output_directory = output_directory + "/"
        else:
            # Create the directory if it does not exist
            output_directory = os.path.dirname(input_file)
            os.makedirs(output_directory + '/outputs_package', exist_ok=True)
            output_directory = output_directory + '/outputs_package/'

        create_pickle_files(processed_df_with_specific_cols, output_directory, assigning_cell_type)

        path = output_directory + os.path.basename(input_file).split('.')[0] + "-" + growth_rate_method + "-analysis"
        path_logs = output_directory + os.path.basename(input_file).split('.')[0] + "-" + growth_rate_method + "-logs"
        path_neighbors = (output_directory + os.path.basename(input_file).split('.')[0] + "-" + growth_rate_method +
                     "-neighbors")

        # write to csv
        processed_df.to_csv(path + '.csv', index=False)
        logs_df.to_csv(path_logs + '.csv', index=False)
        neighbors_df.to_csv(path_neighbors + '.csv', index=False)

        output_log = "The outputs are written in the " + output_directory + " directory."
        print(output_log)
        log_list.append(output_log)

    else:
        log = "The npy folder or neighbor file or both are incorrect!"
        print(log)
        log_list.append(log)

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
    @param path     path where .pickle files are saved
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

    output_file = path + "step-" + '0' * (6-len(str(time_step))) + str(time_step) + ".pickle"

    with open(output_file, 'wb') as export:
        pickle.dump(data, export, protocol=-1)
