import pandas as pd
import numpy as np
import pickle
import os
from ExperimentalDataProcessing import bacteria_analysis_func
from correction import correction_transition


def process_data(input_file, output_directory, interval_time=1, growth_rate_method="Average", um_per_pixel=0.144):
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
    # remove related rows to bacteria with zero MajorAxisLength
    data_frame = data_frame.loc[data_frame["AreaShape_MajorAxisLength"] != 0].reset_index(drop=True)
    # correction dataframe
    number_of_gap = 1
    # I define a gap number to find parent in other previous time steps
    data_frame = correction_transition(data_frame, number_of_gap)
    # print(data_frame )

    # Currently, I do not think this command is needed. So, I comment it.
    # remove Nan labels and zero MajorAxisLength
    # data_frame=data_frame.loc[data_frame["TrackObjects_Label_50"].notnull()].reset_index(drop=True)

    # this part has been written by Aaron Yip (https://github.com/cheekolegend)
    '''
        start
    '''
    # Convert distances to um (0.144 um/pixel on 63X objective)
    data_frame["AreaShape_MajorAxisLength"] = data_frame["AreaShape_MajorAxisLength"] * um_per_pixel
    data_frame["AreaShape_MinorAxisLength"] = data_frame["AreaShape_MinorAxisLength"] * um_per_pixel
    try:
        data_frame["Location_Center_X"] = data_frame["Location_Center_X"] * um_per_pixel
        data_frame["Location_Center_Y"] = data_frame["Location_Center_Y"] * um_per_pixel
    except:
        data_frame["AreaShape_Center_X"] = data_frame["AreaShape_Center_X"] * um_per_pixel
        data_frame["AreaShape_Center_Y"] = data_frame["AreaShape_Center_Y"] * um_per_pixel

    '''
        end
    '''

    # process the tracking data
    processed_data = bacteria_analysis_func(data_frame, interval_time, growth_rate_method)
    output_directory = output_directory + "/"

    create_pickle_files(processed_data, output_directory)
    processed_data.rename(columns={'ImageNumber': 'stepNum'}, inplace=True)

    path = output_directory + input_file.split('/')[-1].split('.')[0] + "-" + growth_rate_method + "-analysis"
    # write to csv
    processed_data.to_csv(path + '.csv', index=False)


class Dict2Class(object):
    """
        goal:  Change Dictionary Into Class

    """

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def create_pickle_files(df, path):
    """
    Saves processed data in a dictionary similar to CellModeller output style and exports as .pickle files
    @param df       data after being processed in ExperimentalDataProcessing.py
    @param path     path where .pickle files are saved
    """
    lineage = {}
    time_steps = list(set(df['ImageNumber'].values))
    for t in time_steps:
        lineage_data_frame = df.loc[(df["ImageNumber"] == t) & df["lineage"]]
        lineage.update(dict(zip(lineage_data_frame["id"].values, lineage_data_frame["lineage"].values)))
        data = {}
        data_frame_current_time_step = df.loc[df["ImageNumber"] == t]
        data["stepNum"] = t
        data["lineage"] = lineage
        # start index from 1
        data_frame_current_time_step.index = np.arange(1, len(data_frame_current_time_step) + 1)
        # select important columns
        data_frame_current_time_step = data_frame_current_time_step[
            ['id', 'cellType', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol', 'targetVol', 'pos',
             'time', 'radius', 'length', 'orientation']]
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

    output_file = path + "step-" + str(time_step) + ".pickle"

    with open(output_file, 'wb') as export:
        pickle.dump(data, export, protocol=-1)


