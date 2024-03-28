import numpy as np
from Trackrefiner.strain.correction.action.bacteriaModification import bacteria_modification, remove_redundant_link


def find_candidate_parents(without_source_bac_index, optimized_cost_df_list, redundant_link_dict_list):
    """
    this function calls for each target (transition or incorrect daughter) bacterium
    @param without_source_bac_index int row index of target bacterium
    @param optimized_cost_df_list list list of distance of target bacterium from other candidate parents
    in different time steps
    in last time step of its life history before transition bacterium to length of candidate parent bacterium
    in investigated time step
    """
    candidate_parents_index = []
    candidate_parents_cost = []

    lowest_cost = None
    correct_link_bac_indx = None
    redundant_bac_ndx = []

    # was the parent found from previous time steps?
    for optimized_cost_df in optimized_cost_df_list:

        linking_info = optimized_cost_df.loc[optimized_cost_df['without parent index'] == without_source_bac_index]

        if linking_info.shape[0] > 0:
            if linking_info['Cost'].values.tolist()[0] != 999:
                candidate_parents_index.append(linking_info['Candida bacteria index in previous time step'].values.tolist()[0])
                candidate_parents_cost.append(linking_info['Cost'].values.tolist()[0])

    if len(candidate_parents_cost) > 0:
        lowest_cost_indx = np.argmin(candidate_parents_cost)
        lowest_cost = candidate_parents_cost[lowest_cost_indx]
        correct_link_bac_indx = candidate_parents_index[lowest_cost_indx]

        if without_source_bac_index in list(redundant_link_dict_list[0].keys()):
            if correct_link_bac_indx == redundant_link_dict_list[0][without_source_bac_index][0]:
                redundant_bac_ndx = redundant_link_dict_list[0][without_source_bac_index][1]

    return correct_link_bac_indx, redundant_bac_ndx, lowest_cost


def assign_parent(df, without_source_bac_index, optimized_cost_df_list, redundant_link_dict_list, neighbors_df,
                  parent_image_number_col, parent_object_number_col, label_col, center_coordinate_columns):
    """
    we should find the nearest candidate bacterium in the previous time steps that has:
    similar Intensity_MeanIntensity pattern (if exist)

    @param df dataframe features value of bacteria in each time step
    @param without_source_bac_index int row index of target bacterium
    @param optimized_cost_df_list list list of distance of target bacteria from other candidate parents
    in different time steps
    in last time step of its life history before transition bacterium to length of candidate parent bacterium
    in investigated time step
    """

    correct_source_link_bac_indx, redundant_bac_ndx_list, lowest_cost = \
        find_candidate_parents(without_source_bac_index, optimized_cost_df_list, redundant_link_dict_list)
    assign_new_link_flag = False

    if correct_source_link_bac_indx is not None:

        if len(redundant_bac_ndx_list) > 0:
            for redundant_bac_ndx in redundant_bac_ndx_list:
                wrong_daughter_life_history = df.loc[(df['id'] == df.iloc[redundant_bac_ndx]['id']) &
                                                     (df['ImageNumber'] >= df.iloc[without_source_bac_index]['ImageNumber'])]

                df = remove_redundant_link(df, wrong_daughter_life_history, neighbors_df, parent_image_number_col,
                          parent_object_number_col, label_col, center_coordinate_columns)

        # find related bacteria to this transition bacterium
        correct_source_bacterium = df.iloc[correct_source_link_bac_indx]

        without_source_bacterium = df.loc[without_source_bac_index]
        without_source_bacterium_life_history = df.loc[df['id'] == without_source_bacterium['id']]
        all_bac_in_without_source_bac_time_step = df.loc[df['ImageNumber'] == without_source_bacterium['ImageNumber']]

        df = bacteria_modification(df, correct_source_bacterium, without_source_bacterium_life_history,
                                   all_bac_in_without_source_bac_time_step, neighbor_df=neighbors_df,
                                   parent_image_number_col=parent_image_number_col,
                                   parent_object_number_col=parent_object_number_col, label_col=label_col,
                                   center_coordinate_columns=center_coordinate_columns)

        assign_new_link_flag = True

    return df, assign_new_link_flag
