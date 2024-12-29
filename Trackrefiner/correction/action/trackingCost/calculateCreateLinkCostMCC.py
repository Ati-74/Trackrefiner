import pandas as pd
import numpy as np
from Trackrefiner.correction.action.findOutlier import calculate_sum_daughter_to_mother_len_boundary, \
    calculate_max_daughter_to_mother_boundary, calculate_upper_statistical_bound, calculate_lower_statistical_bound
from Trackrefiner.correction.action.trackingCost.calculateCreateLinkCost import calc_continuity_link_cost, \
    calc_division_link_cost


def division_detection_cost(df, neighbors_df, neighbor_list_array, candidate_source_daughter_df,
                            min_life_history_of_bacteria_time_step, center_coord_cols, parent_image_number_col,
                            parent_object_number_col, divided_bac_model, divided_vs_non_divided_model,
                            maintain_exist_link_cost_df, coordinate_array):
    """
    Evaluates the cost of maintaining a missing connectivity link (MCC link) while adding a division link
    between the source bacterium and candidate daughter bacteria.

    **Workflow**:
        - Checks if the source bacterium can support a valid division event by:
            - Ensuring biological feasibility based on length ratios.
            - Evaluating statistical boundaries for daughter-to-mother length relationships.
            - Considering the age and unexpected beginning status of the source bacterium.
        - Identifies candidate target bacteria that meet division conditions.
        - Calculates the cost of adding division links using the `calc_division_link_cost` function.

    :param pandas.DataFrame df:
        Main DataFrame containing tracking data for bacteria.
    :param pandas.DataFrame neighbors_df:
        DataFrame containing neighboring relationships between bacteria.
    :param lil_matrix neighbor_list_array:
        Sparse matrix representing neighboring relationships for fast lookups.
    :param pandas.DataFrame candidate_source_daughter_df:
        DataFrame containing features of source bacteria and their potential daughter bacteria.
    :param int min_life_history_of_bacteria_time_step:
        Minimum life history time steps required for a bacterium to be considered valid.
    :param dict center_coord_cols:
        Dictionary specifying column names for bacterial centroid coordinates.
    :param str parent_image_number_col:
        Column name representing parent image numbers.
    :param str parent_object_number_col:
        Column name representing parent object numbers.
    :param sklearn.Model divided_bac_model:
        Machine learning model used to validate division links.
    :param sklearn.Model divided_vs_non_divided_model:
        Machine learning model used to compare divided and non-divided states for bacteria.
    :param pandas.DataFrame maintain_exist_link_cost_df:
        DataFrame containing maintenance costs for existing links.
    :param csr_matrix coordinate_array:
        Array of spatial coordinates used for evaluating candidate links.

    :returns:
        pandas.DataFrame:

        DataFrame containing the cost of maintaining MCC links while adding division links.
    """

    # source has one link, and we check if we can add another link to source bacteria (mother  - daughter relation)

    max_daughter_len_to_mother_ratio_boundary = calculate_max_daughter_to_mother_boundary(df)
    sum_daughter_len_to_mother_ratio_boundary = calculate_sum_daughter_to_mother_len_boundary(df)

    candidate_source_daughter_df['max_target_neighbor_len_to_source'] = \
        np.max([candidate_source_daughter_df['AreaShape_MajorAxisLength_target'],
                candidate_source_daughter_df['AreaShape_MajorAxisLength']], axis=0) / \
        candidate_source_daughter_df['AreaShape_MajorAxisLength_source'].values

    candidate_source_daughter_df['sum_target_neighbor_len_to_source'] = \
        (candidate_source_daughter_df['AreaShape_MajorAxisLength_target'] +
         candidate_source_daughter_df['AreaShape_MajorAxisLength']) / \
        candidate_source_daughter_df['AreaShape_MajorAxisLength_source']

    candidate_source_daughter_df['LengthChangeRatio'] = \
        (candidate_source_daughter_df['AreaShape_MajorAxisLength'] /
         candidate_source_daughter_df['AreaShape_MajorAxisLength_source'])

    upper_bound_max_daughter_len = calculate_upper_statistical_bound(max_daughter_len_to_mother_ratio_boundary)

    lower_bound_sum_daughter_len = calculate_lower_statistical_bound(sum_daughter_len_to_mother_ratio_boundary)

    upper_bound_sum_daughter_len = calculate_upper_statistical_bound(sum_daughter_len_to_mother_ratio_boundary)

    # now check division conditions
    cond1 = ((candidate_source_daughter_df['max_target_neighbor_len_to_source'] < 1) |
             (candidate_source_daughter_df['max_target_neighbor_len_to_source'] <= upper_bound_max_daughter_len))

    cond2 = ((candidate_source_daughter_df['sum_target_neighbor_len_to_source'] >= lower_bound_sum_daughter_len)
             & (candidate_source_daughter_df['sum_target_neighbor_len_to_source'] <= upper_bound_sum_daughter_len))

    cond3 = candidate_source_daughter_df['age_source'] > min_life_history_of_bacteria_time_step

    cond4 = candidate_source_daughter_df['unexpected_beginning_source'] == True

    incorrect_continuity_link_in_this_time_step_with_candidate_neighbors = \
        candidate_source_daughter_df.loc[cond1 & cond2]

    incorrect_continuity_link_in_this_time_step_with_candidate_neighbors = \
        incorrect_continuity_link_in_this_time_step_with_candidate_neighbors.loc[cond3 | cond4]

    if incorrect_continuity_link_in_this_time_step_with_candidate_neighbors.shape[0] > 0:

        cost_df = \
            calc_division_link_cost(df, neighbors_df, neighbor_list_array,
                                    incorrect_continuity_link_in_this_time_step_with_candidate_neighbors.copy(),
                                    center_coord_cols, '_source', '', parent_image_number_col,
                                    parent_object_number_col, divided_bac_model, divided_vs_non_divided_model,
                                    maintain_exist_link_cost_df, check_maintenance_for='target',
                                    coordinate_array=coordinate_array)

    else:
        cost_df = pd.DataFrame()

    return cost_df


def create_continuity_link_cost(df, neighbors_df, neighbor_list_array, mcc_target_neighbors_features,
                                center_coord_cols, parent_image_number_col, parent_object_number_col,
                                non_divided_bac_model, divided_vs_non_divided_model, maintain_exist_link_cost_df,
                                coordinate_array):
    """
    Calculate the cost of creating a new continuity link between a source bacterium with a missing connectivity
    (MCC link) and candidate target bacteria, and replacing the MCC link with this new link.

    :param pandas.DataFrame df:
        Main DataFrame containing tracking data for bacteria.
    :param pandas.DataFrame neighbors_df:
        DataFrame containing neighboring relationships between bacteria.
    :param lil_matrix neighbor_list_array:
        Sparse matrix representing neighboring relationships for fast lookups.
    :param pandas.DataFrame mcc_target_neighbors_features:
        DataFrame containing bacterial measured bacterial features of MCC target neighbors (candidate bacteria)
    :param dict center_coord_cols:
        Dictionary specifying column names for bacterial centroid coordinates.
    :param str parent_image_number_col:
        Column name representing parent image numbers.
    :param str parent_object_number_col:
        Column name representing parent object numbers.
    :param sklearn.Model non_divided_bac_model:
        Machine learning model used to evaluate candidate links for non-divided bacteria.
    :param sklearn.Model divided_vs_non_divided_model:
        Machine learning model used to compare divided and non-divided states for bacteria.
    :param pandas.DataFrame maintain_exist_link_cost_df:
        DataFrame containing maintenance costs for existing links.
    :param csr_matrix coordinate_array:
        Array of spatial coordinates used for evaluating candidate links.

    :returns:
        pandas.DataFrame:

        DataFrame containing the cost of replacing MCC links with new continuity links.
    """

    cost_df = \
        calc_continuity_link_cost(df, neighbors_df, neighbor_list_array,
                                  mcc_target_neighbors_features.copy(),
                                  center_coord_cols, '_source', '', parent_image_number_col,
                                  parent_object_number_col, non_divided_bac_model, divided_vs_non_divided_model,
                                  maintain_exist_link_cost_df, check_maintenance_for='target',
                                  coordinate_array=coordinate_array)

    return cost_df
