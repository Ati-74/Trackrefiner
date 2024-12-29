import pandas as pd
import numpy as np
from Trackrefiner.correction.action.findOutlier import calculate_sum_daughter_to_mother_len_boundary, \
    calculate_max_daughter_to_mother_boundary
from Trackrefiner.correction.action.trackingCost.evaluateDivisionLinkForUB import \
    update_division_cost_and_redundant_links
from Trackrefiner.correction.action.candidateBacteriaFeatureSpace import find_candidates_for_unexpected_beginning
from Trackrefiner.correction.action.trackingCost.calculateMaintainingLinkCost import calc_maintain_exist_link_cost
from Trackrefiner.correction.action.trackingCost.calculateCreateLinkCost import (calc_continuity_link_cost,
                                                                                 calc_division_link_cost)


def compute_create_link_to_unexpected_beginning_cost(df, unexpected_begging_bac_time_step_bacteria,
                                                     unexpected_begging_bacteria, previous_time_step_bacteria,
                                                     neighbors_df, neighbor_list_array, min_life_history_time_step,
                                                     parent_image_number_col, parent_object_number_col,
                                                     center_coord_cols, divided_vs_non_divided_model, non_divided_model,
                                                     division_model, color_array, coordinate_array):
    """
    Computes the cost of linking unexpected beginning bacteria to bacteria from a previous time step,
    evaluating continuity link cost, division costs, and  maintenance the current links of candidate bacteria costs.

    :param pandas.DataFrame df:
        DataFrame containing bacterial measured bacterial features.
    :param pandas.DataFrame unexpected_begging_bac_time_step_bacteria:
        DataFrame containing all bacteria present in the time step of unexpected beginnings.
    :param pandas.DataFrame unexpected_begging_bacteria:
        DataFrame containing unexpected beginning bacteria.
    :param pandas.DataFrame previous_time_step_bacteria:
        DataFrame containing all bacteria from the previous time step.
    :param pandas.DataFrame neighbors_df:
        DataFrame containing neighboring relationships between bacteria, including their image and object numbers.
    :param lil_matrix neighbor_list_array:
        Array storing neighbor connections for bacteria.
    :param int min_life_history_time_step:
        Minimum number of life history frames required for a bacterium to be considered
    :param str parent_image_number_col:
        Column name in the DataFrame for parent image numbers.
    :param str parent_object_number_col:
        Column name in the DataFrame for parent object numbers.
    :param dict center_coord_cols:
        Dictionary containing column names for x and y center coordinates.
    :param sklearn.Model divided_vs_non_divided_model:
        Machine learning model used to compare divided and non-divided states for bacteria
    :param sklearn.Model non_divided_model:
        Machine learning model used to evaluate candidate links for continuity events.
    :param sklearn.Model division_model:
        Machine learning model used to validate division links.
    :param np.ndarray color_array:
        Array representing the colors of objects in the tracking data. This is used for mapping
        bacteria from the dataframe to the coordinate array for spatial analysis.
    :param csr_matrix coordinate_array:
        Array of spatial coordinates used for evaluating candidate links.

    :returns:
        tuple:
            - **continuity_link_cost_df** (*pandas.DataFrame*): Cost DataFrame for creating links to
              continue the life histories of bacteria across time steps.
            - **division_cost_df** (*pandas.DataFrame*): Cost DataFrame for creating division links.
            - **redundant_link_division_df** (*pandas.DataFrame*): DataFrame containing redundant links between
              candidate bacteria and one of their current daughter bacteria.
              These links are flagged for removal to allow unexpected beginning bacteria to establish new link with
              the candidate bacteria.
            - **maintain_exist_link_cost_df** (*pandas.DataFrame*): Cost DataFrame for maintaining the current links of
              candidate bacteria
    """

    max_daughter_len_to_mother_ratio_boundary = calculate_max_daughter_to_mother_boundary(df)
    sum_daughter_len_to_mother_ratio_boundary = calculate_sum_daughter_to_mother_len_boundary(df)

    redundant_link_dict_division = {}
    division_cost_dict = {}

    bac_under_invest_prev_time_step, final_candidate_bac = \
        find_candidates_for_unexpected_beginning(neighbors_df, unexpected_begging_bacteria,
                                                 unexpected_begging_bac_time_step_bacteria,
                                                 previous_time_step_bacteria, center_coord_cols,
                                                 color_array=color_array, coordinate_array=coordinate_array)

    if final_candidate_bac.shape[0] > 0:

        receiver_of_bac_under_invest_link = \
            unexpected_begging_bac_time_step_bacteria.loc[
                unexpected_begging_bac_time_step_bacteria[parent_object_number_col].isin(
                    bac_under_invest_prev_time_step['ObjectNumber'])]

        # now check the cost of maintaining the link
        maintain_exist_link_cost_df = \
            calc_maintain_exist_link_cost(bac_under_invest_prev_time_step, receiver_of_bac_under_invest_link,
                                          center_coord_cols, divided_vs_non_divided_model,
                                          coordinate_array=coordinate_array)

        continuity_link_cost_df = \
            calc_continuity_link_cost(df, neighbors_df, neighbor_list_array, final_candidate_bac.copy(),
                                      center_coord_cols, '_candidate', '', parent_image_number_col,
                                      parent_object_number_col, non_divided_model, divided_vs_non_divided_model,
                                      maintain_exist_link_cost_df, check_maintenance_for='source',
                                      coordinate_array=coordinate_array)

        division_cost_df = calc_division_link_cost(df, neighbors_df, neighbor_list_array, final_candidate_bac.copy(),
                                                   center_coord_cols, '_candidate', '',
                                                   parent_image_number_col, parent_object_number_col, division_model,
                                                   divided_vs_non_divided_model, maintain_exist_link_cost_df,
                                                   check_maintenance_for='source', coordinate_array=coordinate_array)

        if division_cost_df.shape[0] > 0:

            # now check division condition
            source_bac_next_time_step_dict = {}
            source_bac_daughters_dict = {}

            for source_bac_idx in division_cost_df.index.values:
                source_bac = previous_time_step_bacteria.loc[source_bac_idx]

                # Did the bacteria survive or divide?
                source_bac_next_time_step = \
                    unexpected_begging_bac_time_step_bacteria.loc[
                        unexpected_begging_bac_time_step_bacteria['id'] == source_bac['id']]

                source_bac_daughters = \
                    unexpected_begging_bac_time_step_bacteria.loc[
                        unexpected_begging_bac_time_step_bacteria['parent_id'] == source_bac['id']].copy()

                source_bac_next_time_step_dict[source_bac_idx] = source_bac_next_time_step
                source_bac_daughters_dict[source_bac_idx] = source_bac_daughters

            for unexpected_beginning_bac_idx in division_cost_df.columns.values:

                redundant_link_dict_division[unexpected_beginning_bac_idx] = {}
                division_cost_dict[unexpected_beginning_bac_idx] = {}

                unexpected_beginning_bac = \
                    unexpected_begging_bac_time_step_bacteria.loc[unexpected_beginning_bac_idx]

                # unexpected beginning = ub
                candidate_source_bac_ndx_for_this_ub_bac = \
                    division_cost_df[~ division_cost_df[unexpected_beginning_bac_idx].isna()].index.values

                for i, candidate_source_bac_idx in enumerate(candidate_source_bac_ndx_for_this_ub_bac):
                    candidate_source_bac = previous_time_step_bacteria.loc[candidate_source_bac_idx]

                    division_cost_dict, redundant_link_dict_division = \
                        update_division_cost_and_redundant_links(division_cost_df, maintain_exist_link_cost_df,
                                                                 division_cost_dict, candidate_source_bac_idx,
                                                                 candidate_source_bac, unexpected_beginning_bac_idx,
                                                                 unexpected_beginning_bac, redundant_link_dict_division,
                                                                 sum_daughter_len_to_mother_ratio_boundary,
                                                                 max_daughter_len_to_mother_ratio_boundary,
                                                                 min_life_history_time_step,
                                                                 source_bac_next_time_step_dict[
                                                                     candidate_source_bac_idx],
                                                                 source_bac_daughters_dict[candidate_source_bac_idx])

            division_cost_df = pd.DataFrame.from_dict(division_cost_dict, orient='index')
            redundant_link_division_df = pd.DataFrame.from_dict(redundant_link_dict_division, orient='index')

            # now we should transform
            division_cost_df = division_cost_df.transpose()

        else:
            division_cost_df = pd.DataFrame()
            redundant_link_division_df = pd.DataFrame()
    else:
        continuity_link_cost_df = pd.DataFrame()
        division_cost_df = pd.DataFrame()
        redundant_link_division_df = pd.DataFrame()
        maintain_exist_link_cost_df = pd.DataFrame()

    return continuity_link_cost_df, division_cost_df, redundant_link_division_df, maintain_exist_link_cost_df
