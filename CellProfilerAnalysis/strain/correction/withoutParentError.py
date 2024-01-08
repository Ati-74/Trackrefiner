import numpy as np

from CellProfilerAnalysis.strain.correction.action.helperFunctions import bacteria_in_specific_time_step
from CellProfilerAnalysis.strain.correction.action.assignParent import assign_parent
from CellProfilerAnalysis.strain.correction.checkErrors import check_errors_for_without_parent_link
from CellProfilerAnalysis.strain.correction.action.compareBacteria import optimization_transition_cost


def correction_without_parent(raw_df, sorted_npy_files_list, neighbors_df, number_of_gap=0, check_cellType=True,
                              interval_time=1, um_per_pixel=0.144, min_life_history_of_bacteria=20):
    """
        goal: For bacteria without parent, assign labels, ParentImageNumber, and ParentObjectNumber
        @param raw_df    dataframe   bacteria dataframe
        @param number_of_gap int I define a gap number to find parent in other previous time steps
        in last time step of its life history before transition bacterium to length of candidate parent bacterium
        in investigated time step
        output: df   dataframe   modified dataframe (without any transitions)
    """

    # print("number of without parent bacterium: ")
    without_parent_bacterium = raw_df.loc[raw_df["transition"] == True]

    assign_new_link_log = ["The following bacteria have a tracking error: \n ImageNumber\tObjectNumber\tStatus"]

    # print(without_parent_bacterium.shape[0])
    # print("more information: ")
    # print(without_parent_bacterium)

    # min life history of bacteria
    min_life_history_of_bacteria_time_step = np.round_(min_life_history_of_bacteria / interval_time)

    for without_link_to_prev_time_step in without_parent_bacterium['ImageNumber'].unique():

        # filter next time step bacteria information
        under_invst_timestep_bac = bacteria_in_specific_time_step(raw_df, without_link_to_prev_time_step)

        without_link_to_prev_img_npy = sorted_npy_files_list[without_link_to_prev_time_step - 1]

        # filter transition bacteria features value in the next time step (bacteria without parent)
        without_link_to_prev_bac = \
            without_parent_bacterium.loc[without_parent_bacterium['ImageNumber'] == without_link_to_prev_time_step]

        # I define this list to store adjacency matrices
        optimized_cost_df_list = []

        for candidate_node_link_time_step in range(max(without_link_to_prev_time_step - number_of_gap - 1, 1),
                                                   without_link_to_prev_time_step):

            time_step_under_invest_img_npy = sorted_npy_files_list[candidate_node_link_time_step - 1]

            # filter consider time step bacteria information
            bacteria_time_step_under_invest = bacteria_in_specific_time_step(raw_df, candidate_node_link_time_step)

            candidate_bacteria_time_step_under_invest = \
                check_errors_for_without_parent_link(raw_df, bacteria_time_step_under_invest,
                                                     candidate_node_link_time_step,
                                                     min_life_history_of_bacteria_time_step,
                                                     without_link_to_prev_time_step)

            if candidate_bacteria_time_step_under_invest.shape[0] > 0:
                # optimized cost dataframe
                # (rows: next time step sudden bacteria, columns: consider time step bacteria)
                optimized_cost_df = optimization_transition_cost(raw_df, without_link_to_prev_img_npy,
                                                                 under_invst_timestep_bac, without_link_to_prev_bac,
                                                                 time_step_under_invest_img_npy,
                                                                 bacteria_time_step_under_invest,
                                                                 candidate_bacteria_time_step_under_invest,
                                                                 um_per_pixel, check_cellType, neighbors_df)

                optimized_cost_df_list.append(optimized_cost_df)

            # find the parent of each transition bacterium in the next time step
            for transition_bac_index, transition_bac in without_link_to_prev_bac.iterrows():
                raw_df, assign_new_link_flag = assign_parent(raw_df, transition_bac_index, optimized_cost_df_list)

                if assign_new_link_flag:
                    assign_new_link_log.append(str(transition_bac['ImageNumber']) + '\t' + \
                                               str(transition_bac['ObjectNumber']) + '\tPassed')
                else:
                    assign_new_link_log.append(str(transition_bac['ImageNumber']) + '\t' + \
                                               str(transition_bac['ObjectNumber']) + \
                                               '\tFailed to assign a new tracking link.')

    return raw_df, assign_new_link_log
