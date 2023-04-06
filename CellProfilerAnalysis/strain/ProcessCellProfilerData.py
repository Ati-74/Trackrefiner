import pandas as pd
import numpy as np
import pickle
import os
from CellProfilerAnalysis.strain.ExperimentalDataProcessing import bacteria_analysis_func
from CellProfilerAnalysis.strain.correction.Find_Fix_Errors import find_fix_errors


def process_data(input_file, output_directory, interval_time=1, growth_rate_method="Average", number_of_gap=0,
                 um_per_pixel=0.144, intensity_threshold=0.1, assigning_cell_type=True):
    """
    The main function that processes CellProfiler data.
    .pickle Files are exported to the same directory as input_file.
    
    @param interval_time        float   Time between images
    @param growth_rate_method   str     Method for calculating the growth rate
    @param input_file           str    CellProfiler output file in csv format
    @param output_directory     str
    @param um_per_pixel         float   um/pixel
    """

    # Parsing CellProfiler output
    data_frame = pd.read_csv(input_file)
    data_frame = find_fix_errors(data_frame, number_of_gap=number_of_gap, um_per_pixel=um_per_pixel,
                                 intensity_threshold=intensity_threshold, check_cell_type=assigning_cell_type)

    # process the tracking data
    processed_df = bacteria_analysis_func(data_frame, interval_time, growth_rate_method, assigning_cell_type)
    output_directory = output_directory + "/"

    create_pickle_files(processed_df, output_directory, assigning_cell_type)

    path = output_directory + os.path.basename(input_file).split('.')[0] + "-" + growth_rate_method + "-analysis"
    # write to csv
    processed_df.to_csv(path + '.csv', index=False)


class Dict2Class(object):
    """
        goal:  Change Dictionary Into Class

    """

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def create_pickle_files(df, path, assigning_cell_type):
    """
    Saves processed data in a dictionary similar to CellModeller output style and exports as .pickle files
    @param df       data after being processed in ExperimentalDataProcessing.py
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
                ['id', 'label', 'cellType', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol',
                 'targetVol', 'pos', 'time', 'radius', 'length', 'dir', 'ends', 'strainRate', 'strainRate_rolling']]
        else:
            data_frame_current_time_step = data_frame_current_time_step[
                ['id', 'label', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol',
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
