import numpy as np
import pandas as pd

from CellProfilerAnalysis.strain.correction.action.helperFunctions import bacteria_in_specific_time_step
from CellProfilerAnalysis.strain.correction.action.assignParent import assign_parent
from CellProfilerAnalysis.strain.correction.action.compareBacteria import optimization_transition_cost


def correction_without_parent(df, neighbors_df, sorted_npy_files_list, number_of_gap, check_cellType, interval_time,
                              min_life_history_of_bacteria, logs_df):

    """
        goal: For bacteria without parent, assign labels, ParentImageNumber, and ParentObjectNumber
        @param df    dataframe   bacteria dataframe
        @param number_of_gap int I define a gap number to find parent in other previous time steps
        in last time step of its life history before transition bacterium to length of candidate parent bacterium
        in investigated time step
        output: df   dataframe   modified dataframe (without any transitions)
    """

    # print("number of without parent bacterium: ")
    without_source_link_bacteria = df.loc[(df["transition"] == True) & (df['noise_bac'] == False)]

    assign_new_link_log = ["The following bacteria have a tracking error: \n ImageNumber\tObjectNumber\tStatus"]

    # print(without_parent_bacterium.shape[0])
    # print("more information: ")
    # print(without_parent_bacterium)

    # min life history of bacteria
    min_life_history_of_bacteria_time_step = np.round_(min_life_history_of_bacteria / interval_time)

    for without_source_link_time_step in without_source_link_bacteria['ImageNumber'].unique():

        # filter next time step bacteria information
        all_bac_in_without_source_time_step = bacteria_in_specific_time_step(df, without_source_link_time_step)

        # filter transition bacteria features value in the next time step (bacteria without parent)
        without_link_to_source_bac = \
            without_source_link_bacteria.loc[without_source_link_bacteria['ImageNumber'] == \
                                             without_source_link_time_step]

        # I define this list to store adjacency matrices
        optimized_cost_df_list = []
        redundant_link_dict_list = []

        for candidate_node_link_time_step in range(max(without_source_link_time_step - number_of_gap - 1, 1),
                                                   without_source_link_time_step):

            # filter consider time step bacteria information
            all_bac_in_source_time_step = bacteria_in_specific_time_step(df, candidate_node_link_time_step)

            # optimized cost dataframe
            # (rows: next time step sudden bacteria, columns: consider time step bacteria)
            optimized_cost_df, redundant_link_dict = \
                optimization_transition_cost(df, sorted_npy_files_list, all_bac_in_without_source_time_step,
                                             without_link_to_source_bac, all_bac_in_source_time_step, check_cellType,
                                             neighbors_df, min_life_history_of_bacteria_time_step)

            optimized_cost_df_list.append(optimized_cost_df)
            redundant_link_dict_list.append(redundant_link_dict)

            # find the parent of each transition bacterium in the next time step
            for without_source_bac_index, without_source_bac in without_link_to_source_bac.iterrows():
                df, assign_new_link_flag = assign_parent(df, without_source_bac_index, optimized_cost_df_list,
                                                         redundant_link_dict_list, neighbors_df)

                logs_df = pd.concat([logs_df, df.iloc[without_source_bac_index].to_frame().transpose()],
                                    ignore_index=True)

                if assign_new_link_flag:
                    assign_new_link_log.append(str(without_source_bac['ImageNumber']) + '\t' + \
                                               str(without_source_bac['ObjectNumber']) + '\tPassed')
                else:
                    assign_new_link_log.append(str(without_source_bac['ImageNumber']) + '\t' + \
                                               str(without_source_bac['ObjectNumber']) + \
                                               '\tFailed to assign a new tracking link.')

    return df, assign_new_link_log, logs_df
