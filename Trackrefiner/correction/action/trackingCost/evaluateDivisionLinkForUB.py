import numpy as np
from Trackrefiner.correction.action.findOutlier import (calculate_upper_statistical_bound,
                                                        calculate_lower_statistical_bound)


def evaluate_division_link_feasibility(df_source_daughter_cost, division_cost_dict, source_bac_ndx, source_bac,
                                       target_bac_ndx, target_bac, source_bac_next_time_step,
                                       sum_daughter_len_to_mother_ratio_boundary,
                                       max_daughter_len_to_mother_ratio_boundary, min_life_history_of_bacteria):

    """
    Evaluates the feasibility of replacing a continuity link with a division link between a source bacterium
    and a target bacterium by checking specific biological and statistical conditions. Updates the division
    cost dictionary with the evaluated cost.

    **Logic**:
        - Computes the length ratios between the source bacterium and its potential daughters to validate
          whether a division event is biologically plausible.
        - Compares these ratios against statistically derived boundaries to ensure the division adheres to
          expected patterns.
        - Evaluates the age of the source bacterium.
        - If all conditions pass, retrieves the precomputed division cost from the input DataFrame.
        - If any condition fails, assigns a high cost (`1`), indicating that the link is highly improbable.

    :param pandas.DataFrame df_source_daughter_cost:
        DataFrame containing the precomputed division costs between source and daughter bacteria.
    :param dict division_cost_dict:
        Dictionary storing division costs for source-target bacterium pairs.
    :param int source_bac_ndx:
        Index of the source bacterium in the DataFrame.
    :param pandas.Series source_bac:
        Attributes of the source bacterium.
    :param int target_bac_ndx:
        Index of the target bacterium in the DataFrame.
    :param pandas.Series target_bac:
        Attributes of the target bacterium.
    :param pandas.DataFrame source_bac_next_time_step:
        Attributes of the source bacterium in the next time step.
    :param dict sum_daughter_len_to_mother_ratio_boundary:
        Dictionary containing 'avg' (average) and 'std' (standard deviation) for
        the sum of daughter-to-mother length ratios.
    :param dict max_daughter_len_to_mother_ratio_boundary:
        Dictionary containing 'avg' (average) and 'std' (standard deviation) for
        the maximum daughter-to-mother length ratio.
    :param int min_life_history_of_bacteria:
        Minimum required life history frames for a valid bacterium.

    :returns:
        dict:

        Updated `division_cost_dict` with the evaluated division cost for the source-target pair.
    """

    daughters_bac_len_to_source = ((target_bac['AreaShape_MajorAxisLength'] +
                                    source_bac_next_time_step['AreaShape_MajorAxisLength'].values[0]) /
                                   source_bac['AreaShape_MajorAxisLength'])

    max_daughters_bac_len_to_source = (max(target_bac['AreaShape_MajorAxisLength'],
                                           source_bac_next_time_step['AreaShape_MajorAxisLength'].values[0]) /
                                       source_bac['AreaShape_MajorAxisLength'])

    upper_bound_sum_daughter_len = calculate_upper_statistical_bound(sum_daughter_len_to_mother_ratio_boundary)

    lower_bound_sum_daughter_len = calculate_lower_statistical_bound(sum_daughter_len_to_mother_ratio_boundary)

    upper_bound_max_daughter_len = calculate_upper_statistical_bound(max_daughter_len_to_mother_ratio_boundary)

    if (lower_bound_sum_daughter_len <= daughters_bac_len_to_source <= upper_bound_sum_daughter_len and
            max_daughters_bac_len_to_source < 1 and max_daughters_bac_len_to_source <= upper_bound_max_daughter_len):

        if source_bac['age'] > min_life_history_of_bacteria or source_bac['unexpected_beginning'] == True:
            calc_cost = True
        else:
            calc_cost = False

        if calc_cost:
            division_cost = df_source_daughter_cost.at[source_bac_ndx, target_bac_ndx]
        else:
            # prob = 0 so 1 - prob = 1
            division_cost = 1
    else:
        # prob = 0 so 1 - prob = 1
        division_cost = 1

    # for new link cost
    division_cost_dict[target_bac_ndx][source_bac_ndx] = division_cost

    return division_cost_dict


def evaluate_new_division_link(maintenance_cost_df, existing_daughters_idx, target_bac_ndx,
                               source_bac_ndx, new_daughter_prob, redundant_link_dict_division):

    """
    Evaluates whether an existing daughter link should be replaced with a new link to minimize the overall cost.

    **Logic**:
        - Compares the probabilities associated with maintaining the existing daughter links and the new link.
        - Identifies the "worst" existing daughter (the one with the minimum probability).
        - If the "worst" daughter is not the target bacterium, updates the cost to add the new link and marks
          the redundant link in the `redundant_link_dict_division`.
        - If the target bacterium is already the "worst" link, assigns a high cost (`1`) to indicate an
          improbable replacement.

    :param pandas.DataFrame maintenance_cost_df:
        DataFrame containing the maintenance costs for existing daughter links.
    :param list existing_daughters_idx:
        Indices of the existing daughters.
    :param int target_bac_ndx:
        Index of the target bacterium (the new daughter candidate).
    :param int source_bac_ndx:
        Index of the source bacterium.
    :param float new_daughter_prob:
        Probability of establishing a new daughter link with the target bacterium.
    :param dict redundant_link_dict_division:
        Dictionary storing redundant links for division events.

    :returns:
        tuple:
            - float: Cost of adding the new daughter link.
            - dict: Updated `redundant_link_dict_division` with the identified redundant link.
    """

    all_bac_prob_dict = {
        existing_daughters_idx[0]: maintenance_cost_df.at[source_bac_ndx, existing_daughters_idx[0]],
        existing_daughters_idx[1]: maintenance_cost_df.at[source_bac_ndx, existing_daughters_idx[1]],
        target_bac_ndx: 1 - new_daughter_prob,
    }

    # bac with minimum probability
    wrong_daughter_index = list(all_bac_prob_dict.keys())[np.argmin(list(all_bac_prob_dict.values()))]

    if wrong_daughter_index != target_bac_ndx:

        adding_new_daughter_cost = new_daughter_prob
        redundant_link_dict_division[target_bac_ndx][source_bac_ndx] = wrong_daughter_index

    else:
        # probability = 0 so 1 - probability = 1
        adding_new_daughter_cost = 1

    return adding_new_daughter_cost, redundant_link_dict_division


def evaluate_replacement_daughter_link(maintenance_cost_df, source_bac_ndx, source_bac, target_bac_ndx, target_bac,
                                       new_daughter_cost, source_bac_daughters,
                                       max_daughter_len_to_mother_ratio_boundary,
                                       sum_daughter_len_to_mother_ratio_boundary, redundant_link_dict_division):

    """
    Evaluates whether an existing division link can be replaced with a new link to minimize the overall cost,
    while ensuring the replacement adheres to biological constraints.

    **Logic**:
        - Computes the new length ratios for the potential division link, incorporating the source bacterium,
          its existing daughters, and the target bacterium.
        - Compares the new ratios against statistical boundaries (`max_daughter_len_to_mother_ratio_boundary` and
          `sum_daughter_len_to_mother_ratio_boundary`) to ensure the new division link is biologically plausible.
        - If only one daughter satisfies the constraints, compares the probability of the target bacterium
          against the least probable existing daughter link. If the new link is more probable, it replaces the
          least probable link.
        - If multiple daughters satisfy the constraints, delegates the decision to
          `evaluate_replacement_daughter_link` to identify the best replacement.
        - If no daughters meet the constraints, assigns a high cost (`1`), indicating the link is improbable.

    :param pandas.DataFrame maintenance_cost_df:
        DataFrame containing the maintenance costs for existing daughter links.
    :param int source_bac_ndx:
        Index of the source bacterium in the DataFrame.
    :param pandas.Series source_bac:
        Attributes of the source bacterium.
    :param int target_bac_ndx:
        Index of the target bacterium in the DataFrame.
    :param pandas.Series target_bac:
        Attributes of the target bacterium.
    :param float new_daughter_cost:
        Cost of establishing a new daughter link with the target bacterium.
    :param pandas.DataFrame source_bac_daughters:
        DataFrame containing attributes of the source bacterium's existing daughters.
    :param dict max_daughter_len_to_mother_ratio_boundary:
        Dictionary containing 'avg' (average) and 'std' (standard deviation) for
        the maximum daughter-to-mother length ratio.
    :param dict sum_daughter_len_to_mother_ratio_boundary:
        Dictionary containing 'avg' (average) and 'std' (standard deviation) for
        the sum of daughter-to-mother length ratios.
    :param dict redundant_link_dict_division:
        Dictionary storing redundant links for division events.

    :returns:
        tuple:
            - float: Cost of adding the new daughter link.
            - dict: Updated `redundant_link_dict_division` with the identified redundant link.
    """

    upper_bound_max_daughter_len = calculate_upper_statistical_bound(max_daughter_len_to_mother_ratio_boundary)

    upper_bound_sum_daughter_len = calculate_upper_statistical_bound(sum_daughter_len_to_mother_ratio_boundary)

    lower_bound_sum_daughter_len = calculate_lower_statistical_bound(sum_daughter_len_to_mother_ratio_boundary)

    source_bac_daughters['new_sum_daughters_len_to_source'] = \
        (source_bac_daughters['AreaShape_MajorAxisLength'] + target_bac['AreaShape_MajorAxisLength']) / \
        source_bac['AreaShape_MajorAxisLength']

    source_bac_daughters['new_max_daughters_len_to_source'] = \
        np.maximum(source_bac_daughters['AreaShape_MajorAxisLength'], target_bac['AreaShape_MajorAxisLength']) / \
        source_bac['AreaShape_MajorAxisLength']

    filtered_daughters = \
        source_bac_daughters.loc[(source_bac_daughters['new_max_daughters_len_to_source'] < 1) &
                                 (source_bac_daughters['new_max_daughters_len_to_source'] <=
                                  upper_bound_max_daughter_len) &
                                 (lower_bound_sum_daughter_len <=
                                  source_bac_daughters['new_sum_daughters_len_to_source']) &
                                 (source_bac_daughters['new_sum_daughters_len_to_source'] <=
                                  upper_bound_sum_daughter_len)]

    if filtered_daughters.shape[0] > 0:

        if filtered_daughters.shape[0] == 1:

            other_daughter_ndx = [v for v in source_bac_daughters['index'].values if v not in
                                  filtered_daughters['index'].values][0]

            other_daughter_prob = maintenance_cost_df.at[source_bac_ndx, other_daughter_ndx]
            target_bac_prob = 1 - new_daughter_cost

            if target_bac_prob > other_daughter_prob:
                redundant_link_dict_division[target_bac_ndx][source_bac_ndx] = other_daughter_ndx

                adding_new_daughter_cost = new_daughter_cost
            else:
                # probability = 0 so 1 - probability = 1
                adding_new_daughter_cost = 1

        else:

            adding_new_daughter_cost, redundant_link_dict_new_link = \
                evaluate_new_division_link(maintenance_cost_df, filtered_daughters['index'].values.tolist(),
                                           target_bac_ndx, source_bac_ndx, new_daughter_cost,
                                           redundant_link_dict_division)
    else:
        # probability = 0 so 1 - probability = 1
        adding_new_daughter_cost = 1

    return adding_new_daughter_cost, redundant_link_dict_division


def handle_two_daughter_links(division_cost_df, maintenance_cost_df, division_cost_dict, source_bac_ndx,
                              source_bac, target_bac_ndx, target_bac, redundant_link_dict_division,
                              source_bac_daughters, max_daughter_len_to_mother_ratio_boundary,
                              sum_daughter_len_to_mother_ratio_boundary):

    """
    Handles scenarios where a source bacterium already has two daughter links and evaluates the feasibility of
    replacing one of the existing links with a new division link to minimize the overall cost.

    **Logic**:
        - Retrieves the cost of adding a new daughter link from the `division_cost_df`.
        - Delegates the evaluation of replacing an existing link to
          `evaluate_replacement_daughter_link`, which ensures biological constraints are satisfied and
          identifies the most redundant link if applicable.
        - Updates the `division_cost_dict` with the resulting cost for the new link.
        - Updates the `redundant_link_dict_division` to mark any redundant links that were replaced.

    :param pandas.DataFrame division_cost_df:
        DataFrame containing precomputed division costs for source-target links.
    :param pandas.DataFrame maintenance_cost_df:
        DataFrame containing maintenance costs for existing daughter links.
    :param dict division_cost_dict:
        Dictionary storing division costs for source-target bacterium pairs.
    :param int source_bac_ndx:
        Index of the source bacterium in the DataFrame.
    :param pandas.Series source_bac:
        Attributes of the source bacterium.
    :param int target_bac_ndx:
        Index of the target bacterium in the DataFrame.
    :param pandas.Series target_bac:
        Attributes of the target bacterium.
    :param dict redundant_link_dict_division:
        Dictionary storing redundant links for division events.
    :param pandas.DataFrame source_bac_daughters:
        DataFrame containing attributes of the source bacterium's existing daughters.
    :param dict max_daughter_len_to_mother_ratio_boundary:
        Dictionary containing 'avg' (average) and 'std' (standard deviation) for
        the maximum daughter-to-mother length ratio.
    :param dict sum_daughter_len_to_mother_ratio_boundary:
        Dictionary containing 'avg' (average) and 'std' (standard deviation) for
        the sum of daughter-to-mother length ratios.

    :returns:
        tuple:
            - dict: Updated `division_cost_dict` with the cost for the new link.
            - dict: Updated `redundant_link_dict_division` with any redundant links replaced.
    """

    new_daughter_cost = division_cost_df.at[source_bac_ndx, target_bac_ndx]

    (adding_new_daughter_cost, redundant_link_dict_division) = \
        evaluate_replacement_daughter_link(maintenance_cost_df,
                                           source_bac_ndx, source_bac, target_bac_ndx, target_bac,
                                           new_daughter_cost, source_bac_daughters,
                                           max_daughter_len_to_mother_ratio_boundary,
                                           sum_daughter_len_to_mother_ratio_boundary,
                                           redundant_link_dict_division)

    division_cost_dict[target_bac_ndx][source_bac_ndx] = adding_new_daughter_cost

    return division_cost_dict, redundant_link_dict_division


def update_division_cost_and_redundant_links(division_cost_df, maintenance_cost_df, division_cost_dict,
                                             source_bac_ndx, source_bac, target_bac_ndx, target_bac,
                                             redundant_link_dict_division,
                                             sum_daughter_len_to_mother_ratio_boundary,
                                             max_daughter_len_to_mother_ratio_boundary, min_life_history_of_bacteria,
                                             source_bac_next_time_step, source_bac_daughters):

    """
    Evaluates two scenarios for link feasibility:
            1. **Single Link Scenario**: The source bacterium has one existing link. The function evaluates
               the feasibility of adding a new link from the target bacterium to the source bacterium.
            2. **Double Link Scenario**: The source bacterium has two existing links (daughter links).
                The function evaluates the feasibility of removing one of the existing links and
                replacing it with a new link from the target bacterium.

    :param pandas.DataFrame division_cost_df:
        DataFrame containing costs for potential division links.
    :param pandas.DataFrame maintenance_cost_df:
        DataFrame containing costs for maintaining current links of candidate source bacteria.
    :param dict division_cost_dict:
        Dictionary storing updated division costs for all bacteria.
    :param int source_bac_ndx:
        Index of the source bacterium being evaluated.
    :param pandas.Series source_bac:
        Data for the source bacterium being evaluated.
    :param int target_bac_ndx:
        Index of the target bacterium being evaluated.
    :param pandas.Series target_bac:
        Data for the target bacterium being evaluated.
    :param dict redundant_link_dict_division:
        Dictionary storing redundant links for division relationships.
    :param dict sum_daughter_len_to_mother_ratio_boundary:
        Dictionary containing 'avg' (average) and 'std' (standard deviation) for
        the sum of daughter-to-mother length ratios.
    :param dict max_daughter_len_to_mother_ratio_boundary:
        Dictionary containing 'avg' (average) and 'std' (standard deviation) for
        the maximum daughter-to-mother length ratio.
    :param int min_life_history_of_bacteria:
        Minimum number of life history frames required for a bacterium to be considered valid.
    :param pandas.DataFrame source_bac_next_time_step:
        DataFrame containing the next time step's candidate for the source bacterium.
    :param pandas.DataFrame source_bac_daughters:
        DataFrame containing the daughters of the source bacterium.

    :returns:
        tuple:
            - **division_cost_dict** (*dict*): Updated dictionary of division costs for source-target pairs.
            - **redundant_link_dict_division** (*dict*): Updated dictionary of redundant links.
    """

    if source_bac_next_time_step.shape[0] > 0:

        # Maintain the previous link and create a new link (cell division)
        division_cost_dict = \
            evaluate_division_link_feasibility(division_cost_df, division_cost_dict, source_bac_ndx, source_bac,
                                               target_bac_ndx, target_bac, source_bac_next_time_step,
                                               sum_daughter_len_to_mother_ratio_boundary,
                                               max_daughter_len_to_mother_ratio_boundary, min_life_history_of_bacteria)

    elif source_bac_daughters.shape[0] > 0:

        # my idea: at least one of a daughter-mother links can be wrong
        division_cost_dict, redundant_link_dict_division = \
            handle_two_daughter_links(division_cost_df, maintenance_cost_df, division_cost_dict,
                                      source_bac_ndx, source_bac, target_bac_ndx, target_bac,
                                      redundant_link_dict_division, source_bac_daughters,
                                      max_daughter_len_to_mother_ratio_boundary,
                                      sum_daughter_len_to_mother_ratio_boundary)

    return division_cost_dict, redundant_link_dict_division
