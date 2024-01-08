import numpy as np
from CellProfilerAnalysis.strain.correction.action.bacteriaModification import bacteria_modification


def find_candidate_parents(target_bac_index, optimized_cost_df_list):
    """
    this function calls for each target (transition or incorrect daughter) bacterium
    @param target_bac_index int row index of target bacterium
    @param optimized_cost_df_list list list of distance of target bacterium from other candidate parents
    in different time steps
    in last time step of its life history before transition bacterium to length of candidate parent bacterium
    in investigated time step
    """
    candidate_parents_index = []
    candidate_parents_cost = []

    lowest_cost = None
    correct_link_bac_indx = None

    # was the parent found from previous time steps?
    for optimized_cost_df in optimized_cost_df_list:

        linking_info = optimized_cost_df.loc[optimized_cost_df['without parent index'] == target_bac_index]

        if linking_info.shape[0] > 0:
            if linking_info['Cost'].values.tolist()[0] != 999:
                candidate_parents_index.append(linking_info['Candida bacteria index in previous time step'].values.tolist()[0])
                candidate_parents_cost.append(linking_info['Cost'].values.tolist()[0])

    if len(candidate_parents_cost) > 0:
        lowest_cost_indx = np.argmin(candidate_parents_cost)
        lowest_cost = candidate_parents_cost[lowest_cost_indx]
        correct_link_bac_indx = candidate_parents_index[lowest_cost_indx]

    return correct_link_bac_indx, lowest_cost


def assign_parent(df, target_bac_index, optimized_cost_df_list):
    """
    we should find the nearest candidate bacterium in the previous time steps that has:
    similar Intensity_MeanIntensity pattern (if exist)

    @param df dataframe features value of bacteria in each time step
    @param target_bac_index int row index of target bacterium
    @param optimized_cost_df_list list list of distance of target bacteria from other candidate parents
    in different time steps
    in last time step of its life history before transition bacterium to length of candidate parent bacterium
    in investigated time step
    """

    correct_link_bac_indx, lowest_cost = find_candidate_parents(target_bac_index, optimized_cost_df_list)
    assign_new_link_flag = False

    if correct_link_bac_indx is not None:
        # find related bacteria to this transition bacterium
        correct_bacterium = df.iloc[correct_link_bac_indx]

        target_bacterium = df.iloc[target_bac_index]
        target_bacterium_life_history = df.loc[df['id'] == target_bacterium['id']]
        all_bac_undergo_phase_change = df.loc[df['ImageNumber'] == target_bacterium['ImageNumber']]

        df = bacteria_modification(df, correct_bacterium, target_bacterium_life_history, all_bac_undergo_phase_change)
        assign_new_link_flag = True

    return df, assign_new_link_flag
