import pandas as pd
import numpy as np
from Trackrefiner.core.correction.action.findOutlier import calculate_sum_daughter_to_mother_len_boundary, \
    calculate_max_daughter_to_mother_boundary, calculate_upper_statistical_bound, calculate_lower_statistical_bound
from Trackrefiner.core.correction.action.candidateBacteriaFeatureSpace import find_candidates_for_unexpected_end
from Trackrefiner.core.correction.action.trackingCost.calculateMaintainingLinkCost import calc_maintain_exist_link_cost
from Trackrefiner.core.correction.action.trackingCost.calculateCreateLinkCost import (calc_continuity_link_cost,
                                                                                 calc_division_link_cost)


def compute_create_link_from_unexpected_end_cost(df, neighbors_df, neighbor_list_array,
                                                 unexpected_end_bac_time_step_bacteria, unexpected_end_bacteria,
                                                 next_time_step_bacteria, center_coord_cols, parent_image_number_col,
                                                 parent_object_number_col, min_life_history_time_step,
                                                 divided_vs_non_divided_model, non_divided_bac_model, division_model,
                                                 color_array, coordinate_array):
    """
    Computes the cost of linking unexpected end bacteria to bacteria from a next time step,
    evaluating continuity link cost, division costs, and  maintenance the current links of candidate bacteria costs.

    :param pandas.DataFrame df:
        DataFrame containing bacterial measured bacterial features.
    :param pandas.DataFrame unexpected_end_bac_time_step_bacteria:
        DataFrame containing all bacteria present in the time step of unexpected end.
    :param pandas.DataFrame unexpected_end_bacteria:
        DataFrame containing unexpected end bacteria.
    :param pandas.DataFrame next_time_step_bacteria:
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
    :param sklearn.Model non_divided_bac_model:
        Machine learning model used to evaluate candidate links for continuity events.
    :param sklearn.Model division_model:
        Machine learning model used to validate division links.
    :param np.ndarray color_array:
        Array representing the colors of objects in the tracking data. This is used for mapping
        bacteria from the dataframe to the coordinate array for spatial analysis.
    :param csr_matrix coordinate_array:
        Array of spatial coordinates used for evaluating candidate links.

    :returns:
        tuple: A tuple containing the following DataFrames:
            - **filtered_continuity_link_cost_df** (*pandas.DataFrame*):
                Filtered continuity link costs after applying thresholds and validations.

            - **validated_division_cost_df** (*pandas.DataFrame*):
                A DataFrame containing division costs after resolving conflicts and selecting
                the most plausible division links. Conflicts are resolved in three main scenarios:

                1. **Overlapping Division Scenarios for a Single Source**:
                    - When a single source bacterium is involved in multiple division scenarios with
                      overlapping daughter candidates, the function evaluates each scenario's probabilities.
                      The division pair with the highest likelihood is retained,
                      and other conflicting scenarios are removed.

                2. **Overlapping Daughters Across Different Sources**:
                    - When different source bacteria propose overlapping daughters in their division scenarios,
                      the utils compares the average division probabilities for each source.
                      The highest likely source-daughter link is retained.

                3. **Conflict Between Division and Continuity**:
                    - When a candidate bacterium is involved in both division and continuity scenarios,
                      either for a single source or multiple sources, the function compares probabilities,
                      retains the more likely link (division or continuity),
                      and removes the less likely one, ensuring biologically plausible assignments.

            - **final_division_cost_df** (*pandas.DataFrame*):
                Cleaned and finalized division costs for further analysis.
"""

    max_daughter_len_to_mother_ratio_boundary = calculate_max_daughter_to_mother_boundary(df)
    sum_daughter_len_to_mother_ratio_boundary = calculate_sum_daughter_to_mother_len_boundary(df)

    upper_bound_max_daughter_len = calculate_upper_statistical_bound(max_daughter_len_to_mother_ratio_boundary)

    lower_bound_sum_daughter_len = calculate_lower_statistical_bound(sum_daughter_len_to_mother_ratio_boundary)

    upper_bound_sum_daughter_len = calculate_upper_statistical_bound(sum_daughter_len_to_mother_ratio_boundary)

    bac_under_invest, final_candidate_bac = \
        find_candidates_for_unexpected_end(neighbors_df, unexpected_end_bacteria, unexpected_end_bac_time_step_bacteria,
                                           next_time_step_bacteria, center_coord_cols, color_array, coordinate_array)

    if final_candidate_bac.shape[0] > 0:

        source_of_bac_under_invest_link = \
            unexpected_end_bac_time_step_bacteria.loc[
                unexpected_end_bac_time_step_bacteria['ObjectNumber'].isin(
                    bac_under_invest[parent_object_number_col].values)]

        # now check the cost of maintaining the link
        maintain_exist_link_cost_df = \
            calc_maintain_exist_link_cost(source_of_bac_under_invest_link, bac_under_invest, center_coord_cols,
                                          divided_vs_non_divided_model, coordinate_array=coordinate_array)

        filtered_continuity_link_cost_df = \
            calc_continuity_link_cost(df, neighbors_df, neighbor_list_array, final_candidate_bac.copy(),
                                      center_coord_cols, col_source='', col_target='_candidate',
                                      parent_image_number_col=parent_image_number_col,
                                      parent_object_number_col=parent_object_number_col,
                                      continuity_links_model=non_divided_bac_model,
                                      division_vs_continuity_model=divided_vs_non_divided_model,
                                      maintain_exist_link_cost_df=maintain_exist_link_cost_df,
                                      check_maintenance_for='target', coordinate_array=coordinate_array)

        raw_division_cost_df = calc_division_link_cost(df, neighbors_df, neighbor_list_array,
                                                       final_candidate_bac.copy(), center_coord_cols, '',
                                                       '_candidate', parent_image_number_col,
                                                       parent_object_number_col, division_model,
                                                       divided_vs_non_divided_model,
                                                       maintain_exist_link_cost_df=maintain_exist_link_cost_df,
                                                       check_maintenance_for='target',
                                                       coordinate_array=coordinate_array)

        # now I want to check division chance
        # ba careful the probability reported in raw_division_cost_df is 1 - probability
        # conditions for division: 1. sum daughter length to mother 2. max daughter length to mother
        # 3. min life history of source
        # first I check life history of source bac (unexpected end)
        candidate_unexpected_end_bac_for_division = \
            final_candidate_bac.loc[(final_candidate_bac['age'] > min_life_history_time_step) |
                                    (final_candidate_bac['Unexpected_Beginning'] == True)]

        # unexpected bacteria index is index of raw_division_cost_df
        raw_division_cost_df = \
            raw_division_cost_df.loc[
                raw_division_cost_df.index.isin(candidate_unexpected_end_bac_for_division['index'].values)]

        division_cost_dict = {}

        # < 2: # it means that division can not possible
        filtered_division_cost_df = raw_division_cost_df[
            raw_division_cost_df[raw_division_cost_df < 0.5].count(axis=1) >= 2]

        for row_idx, row in filtered_division_cost_df.iterrows():

            candidate_daughters = row[row < 0.5].index.values

            unexpected_bac_length = df.at[row_idx, 'AreaShape_MajorAxisLength']

            if len(candidate_daughters) == 2:

                daughter1_length = df.at[candidate_daughters[0], 'AreaShape_MajorAxisLength']
                daughter2_length = df.at[candidate_daughters[1], 'AreaShape_MajorAxisLength']

                sum_daughter_to_mother = ((daughter1_length + daughter2_length) / unexpected_bac_length)

                max_daughter_to_mother = max(daughter1_length, daughter2_length) / unexpected_bac_length

                if max_daughter_to_mother < 1 and max_daughter_to_mother <= upper_bound_max_daughter_len and \
                        lower_bound_sum_daughter_len <= sum_daughter_to_mother <= upper_bound_sum_daughter_len:
                    division_cost_dict[row_idx] = {candidate_daughters[0]: row[candidate_daughters[0]],
                                                   candidate_daughters[1]: row[candidate_daughters[1]]}

            elif len(candidate_daughters) > 2:

                num_candidate_daughters = len(candidate_daughters)

                division_cost_for_compare = []
                division_cost_daughters = []

                for i_idx in range(num_candidate_daughters):

                    i = candidate_daughters[i_idx]
                    daughter1_length = df.at[i, 'AreaShape_MajorAxisLength']

                    for j_idx in range(i_idx + 1, num_candidate_daughters):

                        j = candidate_daughters[j_idx]

                        daughter2_length = df.at[j, 'AreaShape_MajorAxisLength']

                        sum_daughter_to_mother = ((daughter1_length + daughter2_length) / unexpected_bac_length)

                        max_daughter_to_mother = max(daughter1_length, daughter2_length) / unexpected_bac_length

                        if (max_daughter_to_mother < 1 and
                                max_daughter_to_mother <= upper_bound_max_daughter_len and
                                lower_bound_sum_daughter_len <=
                                sum_daughter_to_mother <= upper_bound_sum_daughter_len):
                            # daughters
                            # compare basen on average probability (1 - probability)
                            division_cost_for_compare.append(np.average([row[i], row[j]]))
                            division_cost_daughters.append((i, j))

                    if division_cost_daughters:
                        min_div_cost_idx = np.argmin(division_cost_for_compare)
                        selected_daughters = division_cost_daughters[min_div_cost_idx]

                        division_cost_dict[row_idx] = {
                            selected_daughters[0]: row[selected_daughters[0]],
                            selected_daughters[1]: row[selected_daughters[1]]}

        final_division_cost_df = pd.DataFrame.from_dict(division_cost_dict, orient='index')

        # now I want to check is there any column with more than one value? if yes, it means that two different source
        # want to have same daughter
        non_nan_counts = final_division_cost_df.count()
        # Identify columns with more than one non-NaN value
        # it means that this daughter can participate in two different division
        columns_with_multiple_non_nan = non_nan_counts[non_nan_counts > 1].index

        if len(columns_with_multiple_non_nan) > 0:
            for col in columns_with_multiple_non_nan:
                # if final_division_cost_df[col].count() > 1:
                # I want to compare average probability
                division_cost_check = final_division_cost_df.loc[~ final_division_cost_df[col].isna()]

                # find division with lower chance
                avg_division_probability_per_unexpected_end_bac = division_cost_check.mean(axis=1)

                incorrect_division_cond = (avg_division_probability_per_unexpected_end_bac >
                                           avg_division_probability_per_unexpected_end_bac.min())

                # Align the boolean condition index with the original DataFrame
                condition_aligned = incorrect_division_cond.reindex(final_division_cost_df.index, fill_value=False)

                final_division_cost_df.loc[condition_aligned] = np.nan

        # now we should compare division with same cost because one target bac can be daughter of
        common_columns = np.intersect1d(final_division_cost_df.columns, filtered_continuity_link_cost_df.columns)

        if len(common_columns) > 0:

            # Calculate the minimum values for each column in filtered_continuity_link_cost_df
            incorrect_continuity_link_nan_col = []
            incorrect_division_row_ndx = []

            for col in common_columns:
                min_continuity_bac_cost = filtered_continuity_link_cost_df[col].min()

                division_related_to_this_col = final_division_cost_df.loc[~ final_division_cost_df[col].isna()]
                min_div_probability = min(division_related_to_this_col.mean())

                if min_continuity_bac_cost > min_div_probability:
                    # it means 1 - continuity probability is lower than 1 - division probability
                    incorrect_continuity_link_nan_col.append(col)
                else:
                    incorrect_division_row_ndx.extend(division_related_to_this_col.index.values)
                    final_division_cost_df.loc[division_related_to_this_col.index.values] = np.nan

            filtered_continuity_link_cost_df[incorrect_continuity_link_nan_col] = np.nan
            final_division_cost_df.loc[incorrect_division_row_ndx] = np.nan

        # remove rows with all value equal to nan
        filtered_continuity_link_cost_df = filtered_continuity_link_cost_df.dropna(axis=1, how='all')
        final_division_cost_df = final_division_cost_df.dropna(axis=1, how='all')
        validated_division_cost_df = final_division_cost_df.copy()

        un_selected_col = []
        for row_idx, row in validated_division_cost_df.iterrows():
            un_selected_col.extend(row[row > row.min()].index.values.tolist())

        validated_division_cost_df[un_selected_col] = np.nan
        validated_division_cost_df = validated_division_cost_df.dropna(axis=1, how='all')

    else:
        filtered_continuity_link_cost_df = pd.DataFrame()
        final_division_cost_df = pd.DataFrame()
        validated_division_cost_df = pd.DataFrame()

    return filtered_continuity_link_cost_df, validated_division_cost_df, final_division_cost_df
