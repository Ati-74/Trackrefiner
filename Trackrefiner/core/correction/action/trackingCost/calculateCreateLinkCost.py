from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
from Trackrefiner.core.correction.action.findOutlier import calculate_length_change_ratio_boundary, \
    calculate_lower_statistical_bound
from Trackrefiner.core.correction.action.helper import calculate_angles_between_slopes
from Trackrefiner.core.correction.modelTraning.calculation.iouCalForML import calculate_iou_ml
from Trackrefiner.core.correction.modelTraning.calculation.calcDistanceForML import calc_min_distance_ml
from Trackrefiner.core.correction.modelTraning.calculation.lengthRatio import check_len_ratio
from Trackrefiner.core.correction.modelTraning.calculation.calMotionAlignmentAngle import calc_motion_alignment_angle_ml
from Trackrefiner.core.correction.action.neighborAnalysis import compare_neighbor_sets
from Trackrefiner.core.correction.action.helper import calculate_trajectory_angles


def optimize_assignment_using_hungarian(cost_df):
    """
    Optimizes the assignment of rows to columns in a cost matrix using the Hungarian algorithm.

    :param pandas.DataFrame cost_df:
        A cost matrix represented as a DataFrame

    :returns:
        pandas.DataFrame

        A DataFrame containing the optimized assignments and their associated costs:

    """

    # Convert DataFrame to NumPy array
    cost_matrix = cost_df.values

    # Applying the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Retrieve the original row indices and column names
    original_row_indices = cost_df.index[row_ind]
    original_col_names = cost_df.columns[col_ind]

    # Retrieve the costs for the selected elements
    selected_costs = cost_matrix[row_ind, col_ind]

    # Create a DataFrame for the results
    result_df = pd.DataFrame({
        'idx1': original_row_indices,
        'idx2': original_col_names,
        'cost': selected_costs
    })

    return result_df


def calc_division_link_cost(df, neighbors_df, neighbor_list_array, df_source_daughter, center_coord_cols,
                            col_source, col_target, parent_image_number_col, parent_object_number_col,
                            division_links_model, division_vs_continuity_model, maintain_exist_link_cost_df,
                            check_maintenance_for='target', coordinate_array=None):

    """
    Calculates the cost of establishing division links between a source bacterium and its potential daughter bacteria.
    This function evaluates potential division links by calculating various spatial, geometric, and
    neighborhood-based features. It uses machine learning models to assess the likelihood of a division event
    based on these features.

    **Behavior**:
    - Calculates features like intersection-over-union (IoU), centroid distance, length ratios, and neighbor overlap.
    - Predicts division probabilities using a pre-trained model (`division_links_model`).
    - Compares division and continuity probabilities using a comparison model (`division_vs_continuity_model`).
    - Adjusts costs based on whether maintaining existing links is preferable.

    :param pandas.DataFrame df:
        Full DataFrame containing bacterial features for the current time step.
    :param pandas.DataFrame neighbors_df:
        DataFrame describing relationships between neighboring bacteria.
    :param lil_matrix neighbor_list_array:
        Sparse matrix representing neighbor relationships.
    :param pandas.DataFrame df_source_daughter:
        DataFrame containing the source bacteria and their candidate daughters.
    :param dict center_coord_cols:
        Dictionary specifying column names for bacterial centroid coordinates
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param str col_source:
        Suffix for source bacterium columns.
    :param str col_target:
        Suffix for target bacterium (daughter) columns.
    :param str parent_image_number_col:
        Column name for the parent image number.
    :param str parent_object_number_col:
        Column name for the parent object number.
    :param sklearn.Model division_links_model:
        Machine learning model to predict division probabilities.
    :param sklearn.Model division_vs_continuity_model:
        Machine learning model used to compare divided and non-divided states for bacteria.
    :param pandas.DataFrame maintain_exist_link_cost_df:
        Cost of maintaining current links between bacteria.
    :param str check_maintenance_for:
        Specifies whether maintenance costs are evaluated for the `target` or `source`. Default is `'target'`.
    :param csr_matrix coordinate_array:
        spatial coordinate matrix for neighborhood and overlap calculations.

    :returns:
        pandas.DataFrame:

        A cost matrix for potential division links, with rows as source bacteria and columns as their
        candidate daughters.
    """

    df_source_daughter[f'index_prev{col_target}'] = df_source_daughter[f'index{col_target}']
    df_source_daughter[f'index2{col_target}'] = df_source_daughter.index.values

    df_source_daughter[f'Length_Change_Ratio{col_target}'] = \
        (df_source_daughter[f'AreaShape_MajorAxisLength{col_target}'] /
         df_source_daughter[f'AreaShape_MajorAxisLength{col_source}'])

    df_source_daughter = calculate_iou_ml(df_source_daughter, col_source=f'prev_index{col_source}',
                                          col_target=f'prev_index{col_target}', link_type='div',
                                          coordinate_array=coordinate_array, both=True)

    df_source_daughter = calc_min_distance_ml(df_source_daughter, center_coord_cols, postfix_target=col_target,
                                              postfix_source=col_source, link_type='div')

    df_source_daughter[f'prev_time_step_index{col_target}'] = df_source_daughter[f'index{col_source}'].astype('int64')

    df_source_daughter[f'Neighbor_Difference_Count{col_target}'] = np.nan
    df_source_daughter[f'other_daughter_index{col_target}'] = np.nan
    df_source_daughter[f'parent_id{col_target}'] = df_source_daughter[f'id{col_source}']

    df_source_daughter = compare_neighbor_sets(df_source_daughter, neighbor_list_array, parent_image_number_col,
                                               parent_object_number_col, selected_rows_df=df_source_daughter,
                                               selected_time_step_df=df, return_common_elements=True,
                                               col_target=col_target)

    df_source_daughter[f'diff_common_diff{col_target}'] = \
        df_source_daughter[f'Neighbor_Difference_Count{col_target}'] - \
        df_source_daughter[f'Neighbor_Shared_Count{col_target}']

    df_source_daughter.loc[df_source_daughter[f'diff_common_diff{col_target}'] < 0, f'diff_common_diff{col_target}'] = 0

    df_source_daughter[f'angle_mother_daughter{col_target}'] = \
        calculate_angles_between_slopes(df_source_daughter[f'Bacterium_Slope{col_source}'].values,
                                        df_source_daughter[f'Bacterium_Slope{col_target}'].values)

    df_source_daughter[f'Prev_Center_X{col_target}'] = \
        df_source_daughter[f"{center_coord_cols['x']}{col_source}"]
    df_source_daughter[f'Prev_Center_Y{col_target}'] = \
        df_source_daughter[f"{center_coord_cols['y']}{col_source}"]

    direction_of_motion = calculate_trajectory_angles(df_source_daughter, center_coord_cols, suffix1=col_target)

    df_source_daughter[f'Direction_of_Motion{col_target}'] = direction_of_motion

    # motion alignment
    # calculated for original df and we should calc for new df
    df_source_daughter[f"Motion_Alignment_Angle{col_target}"] = np.nan
    df_source_daughter = calc_motion_alignment_angle_ml(df, neighbors_df, center_coord_cols,
                                                        selected_rows=df_source_daughter, col_target=col_target,
                                                        col_source=col_source)

    df_source_daughter['adjusted_Neighbor_Shared_Count' + col_target] = \
        df_source_daughter['Neighbor_Shared_Count' + col_target].replace(0, 1)

    df_source_daughter['neighbor_ratio' + col_target] = \
        (df_source_daughter['Neighbor_Difference_Count' + col_target] / (
            df_source_daughter['adjusted_Neighbor_Shared_Count' + col_target]))

    raw_features = ['iou', 'iou_continuity', 'min_distance', 'min_distance_continuity',
                    'Neighbor_Difference_Count' + col_target,
                    'neighbor_ratio' + col_target, 'Neighbor_Shared_Count' + col_target,
                    'angle_mother_daughter' + col_target, 'Direction_of_Motion' + col_target,
                    'Motion_Alignment_Angle' + col_target, 'Length_Change_Ratio' + col_target, 'index' + col_source,
                    'index_prev' + col_target, 'divideFlag' + col_target, 'LifeHistory' + col_target,
                    'age' + col_target, 'diff_common_diff' + col_target]

    df_source_daughter = df_source_daughter[raw_features].copy()

    df_source_daughter = df_source_daughter.rename(
        {
            'Neighbor_Difference_Count' + col_target: 'Neighbor_Difference_Count',
            'Neighbor_Shared_Count' + col_target: 'Neighbor_Shared_Count',
            'neighbor_ratio' + col_target: 'neighbor_ratio',
            'angle_mother_daughter' + col_target: 'angle_mother_daughter',
            'Direction_of_Motion' + col_target: 'Direction_of_Motion',
            'Motion_Alignment_Angle' + col_target: 'Motion_Alignment_Angle',
            'Length_Change_Ratio' + col_target: 'Length_Change_Ratio',
            'index_prev' + col_target: 'index_prev',
            'divideFlag' + col_target: 'divideFlag',
            'LifeHistory' + col_target: 'LifeHistory',
            'age' + col_target: 'age',
            'diff_common_diff' + col_target: 'diff_common_diff'
        }, axis=1)

    feature_list_division_links_model = ['iou', 'min_distance', 'neighbor_ratio', 'angle_mother_daughter']

    y_prob_division_links_model = \
        division_links_model.predict_proba(df_source_daughter[feature_list_division_links_model])[:, 1]
    df_source_daughter['prob_divided_bac_model'] = y_prob_division_links_model

    # difference_neighbors
    feature_list_for_compare = \
        ['iou', 'min_distance', 'neighbor_ratio', 'Direction_of_Motion', 'Motion_Alignment_Angle',
         'Length_Change_Ratio']

    # division is class 0
    y_prob_compare = \
        division_vs_continuity_model.predict_proba(df_source_daughter[feature_list_for_compare])[:, 0]
    df_source_daughter['prob_compare'] = y_prob_compare

    if division_vs_continuity_model is not None:

        df_source_daughter = df_source_daughter.loc[(df_source_daughter['prob_divided_bac_model'] > 0.5) &
                                                    (df_source_daughter['prob_compare'] > 0.5)]

        if df_source_daughter.shape[0] > 0:
            # Pivot this DataFrame to get the desired structure
            division_cost_df = \
                df_source_daughter[['index' + col_source, 'index_prev', 'prob_compare']].pivot(
                    index='index' + col_source, columns='index_prev', values='prob_compare')
            division_cost_df.columns.name = None
            division_cost_df.index.name = None

            if check_maintenance_for == 'target':
                # I want to measure is it possible to remove the current link of target bacterium?
                candidate_target_maintenance_cost = \
                    maintain_exist_link_cost_df.loc[:, maintain_exist_link_cost_df.columns.isin(
                        df_source_daughter['index_prev'].unique())]

                # don't need to convert probability to 1 - probability
                for col in candidate_target_maintenance_cost.columns.values:
                    non_na_probability = candidate_target_maintenance_cost[col].dropna().iloc[0]
                    division_cost_df.loc[division_cost_df[col] <= non_na_probability, col] = np.nan

            # for maintenance_to_be_check = source we can not check, and we should check it after

            division_cost_df = 1 - division_cost_df

        else:
            division_cost_df = pd.DataFrame()
    else:
        # Pivot this DataFrame to get the desired structure
        division_cost_df = \
            df_source_daughter[['index' + col_source, 'index_prev', 'prob_divided_bac_model']].pivot(
                index='index' + col_source, columns='index_prev', values='prob_divided_bac_model')
        division_cost_df.columns.name = None
        division_cost_df.index.name = None

    return division_cost_df


def calc_continuity_link_cost(df, neighbors_df, neighbor_list_array, source_bac_with_can_target,
                              center_coord_cols, col_source, col_target, parent_image_number_col,
                              parent_object_number_col, continuity_links_model, division_vs_continuity_model,
                              maintain_exist_link_cost_df, check_maintenance_for='target', coordinate_array=None):

    """
    Calculates the cost of establishing continuity link between a source bacterium and its potential target bacterium.
    This function evaluates continuity links by analyzing spatial, geometric, and neighborhood-based features.
    It determines the likelihood that a bacterium in one time step corresponds to the same bacterium in the next
    time step.

    **Behavior**:
    - Calculates features like intersection-over-union (IoU), centroid distance, length ratios, and neighbor overlap.
    - Predicts continuity probabilities using a pre-trained model (`continuity_links_model`).
    - Compares continuity and division probabilities using a comparison model (`division_vs_continuity_model`).
    - Adjusts costs based on whether maintaining existing links is preferable.

    :param pandas.DataFrame df:
        Full DataFrame containing bacterial features for the current time step.
    :param pandas.DataFrame neighbors_df:
        DataFrame describing relationships between neighboring bacteria.
    :param lil_matrix neighbor_list_array:
        Sparse matrix representing neighbor relationships.
    :param pandas.DataFrame source_bac_with_can_target:
        DataFrame containing the source bacteria and their candidate targets in the next time step.
    :param dict center_coord_cols:
        Dictionary specifying column names for bacterial centroid coordinates
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param str col_source:
        Suffix for source bacterium columns.
    :param str col_target:
        Suffix for target bacterium columns.
    :param str parent_image_number_col:
        Column name for the parent image number.
    :param str parent_object_number_col:
        Column name for the parent object number.
    :param sklearn.Model continuity_links_model:
        Machine learning model to predict continuity probabilities.
    :param sklearn.Model division_vs_continuity_model:
        Machine learning model used to compare divided and non-divided states for bacteria.
    :param pandas.DataFrame maintain_exist_link_cost_df:
        Cost of maintaining current links between bacteria.
    :param str check_maintenance_for:
        Specifies whether maintenance costs are evaluated for the `target` or `source`. Default is `'target'`.
    :param csr_matrix coordinate_array:
        spatial coordinate matrix for neighborhood and overlap calculations.

    :returns:
        pandas.DataFrame:

        A cost matrix for potential continuity links, with rows as source bacteria and columns as their
        candidate targets.
    """

    bac_len_to_bac_ratio_boundary = calculate_length_change_ratio_boundary(df)

    lower_bound_threshold = calculate_lower_statistical_bound(bac_len_to_bac_ratio_boundary)

    source_bac_with_can_target['index_prev' + col_target] = source_bac_with_can_target['index' + col_target]
    source_bac_with_can_target['index2' + col_target] = source_bac_with_can_target.index.values

    source_bac_with_can_target['Length_Change_Ratio' + col_target] = (
            source_bac_with_can_target['AreaShape_MajorAxisLength' + col_target] /
            source_bac_with_can_target['AreaShape_MajorAxisLength' + col_source])

    source_bac_with_can_target = \
        source_bac_with_can_target.loc[source_bac_with_can_target['Length_Change_Ratio' + col_target]
                                       >= lower_bound_threshold].copy()

    if source_bac_with_can_target.shape[0] > 0:

        source_bac_with_can_target = calculate_iou_ml(source_bac_with_can_target, col_source='prev_index' + col_source,
                                                      col_target='prev_index' + col_target, link_type='continuity',
                                                      coordinate_array=coordinate_array)

        source_bac_with_can_target = calc_min_distance_ml(source_bac_with_can_target, center_coord_cols,
                                                          postfix_target=col_target, postfix_source=col_source,
                                                          link_type=None)

        source_bac_with_can_target['prev_time_step_index' + col_target] = \
            source_bac_with_can_target['index' + col_source].astype('int64')

        source_bac_with_can_target['Neighbor_Difference_Count' + col_target] = np.nan
        source_bac_with_can_target['other_daughter_index' + col_target] = np.nan
        source_bac_with_can_target['parent_id' + col_target] = source_bac_with_can_target['id' + col_source]

        source_bac_with_can_target = \
            compare_neighbor_sets(source_bac_with_can_target, neighbor_list_array,
                                  parent_image_number_col, parent_object_number_col,
                                  selected_rows_df=source_bac_with_can_target, selected_time_step_df=df,
                                  return_common_elements=True, col_target=col_target)

        source_bac_with_can_target = check_len_ratio(df, source_bac_with_can_target, col_target=col_target,
                                                     col_source=col_source)

        # motion alignment
        # calculated for original df and we should calc for new df
        source_bac_with_can_target["Motion_Alignment_Angle" + col_target] = np.nan
        source_bac_with_can_target = \
            calc_motion_alignment_angle_ml(df, neighbors_df, center_coord_cols,
                                           selected_rows=source_bac_with_can_target, col_target=col_target,
                                           col_source=col_source)

        source_bac_with_can_target['Prev_Center_X' + col_target] = \
            source_bac_with_can_target[center_coord_cols['x'] + col_source]
        source_bac_with_can_target['Prev_Center_Y' + col_target] = \
            source_bac_with_can_target[center_coord_cols['y'] + col_source]

        direction_of_motion = calculate_trajectory_angles(source_bac_with_can_target,
                                                          center_coord_cols, suffix1=col_target)

        source_bac_with_can_target['Direction_of_Motion' + col_target] = direction_of_motion

        source_bac_with_can_target['adjusted_Neighbor_Shared_Count' + col_target] = \
            source_bac_with_can_target['Neighbor_Shared_Count' + col_target].replace(0, 1)

        source_bac_with_can_target['neighbor_ratio' + col_target] = \
            (source_bac_with_can_target['Neighbor_Difference_Count' + col_target] / (
                source_bac_with_can_target['adjusted_Neighbor_Shared_Count' + col_target]))

        raw_feature_list = ['iou', 'min_distance', 'Neighbor_Difference_Count' + col_target,
                            'Neighbor_Shared_Count' + col_target,
                            'neighbor_ratio' + col_target, 'length_dynamic' + col_target,
                            'Motion_Alignment_Angle' + col_target,
                            'Direction_of_Motion' + col_target, 'Length_Change_Ratio' + col_target,
                            'index' + col_source, 'index_prev' + col_target]

        source_bac_with_can_target = source_bac_with_can_target[raw_feature_list].copy()
        source_bac_with_can_target = source_bac_with_can_target.rename(
            {
                'Neighbor_Shared_Count' + col_target: 'Neighbor_Shared_Count',
                'Neighbor_Difference_Count' + col_target: 'Neighbor_Difference_Count',
                'neighbor_ratio' + col_target: 'neighbor_ratio',
                'length_dynamic' + col_target: 'length_dynamic',
                'Motion_Alignment_Angle' + col_target: 'Motion_Alignment_Angle',
                'Direction_of_Motion' + col_target: 'Direction_of_Motion',
                'Length_Change_Ratio' + col_target: 'Length_Change_Ratio',
                'index_prev' + col_target: 'index_prev',
            }, axis=1)

        # difference_neighbors
        feature_list_for_continuity_links_model = \
            ['iou', 'min_distance', 'neighbor_ratio', 'length_dynamic', 'Motion_Alignment_Angle']

        # difference_neighbors
        feature_list_for_compare = \
            ['iou', 'min_distance', 'neighbor_ratio', 'Direction_of_Motion', 'Motion_Alignment_Angle',
             'Length_Change_Ratio']

        y_prob_continuity_links_model = \
            continuity_links_model.predict_proba(source_bac_with_can_target[
                                                    feature_list_for_continuity_links_model])[:, 1]
        source_bac_with_can_target['prob_non_divided_bac_model'] = y_prob_continuity_links_model

        # non divided class 1
        y_prob_compare = \
            division_vs_continuity_model.predict_proba(source_bac_with_can_target[
                                                                  feature_list_for_compare])[:, 1]
        source_bac_with_can_target['prob_compare'] = y_prob_compare

        source_bac_with_can_target = \
            source_bac_with_can_target.loc[(source_bac_with_can_target['prob_non_divided_bac_model'] > 0.5) &
                                           (source_bac_with_can_target['prob_compare'] > 0.5) &
                                           (source_bac_with_can_target['Neighbor_Difference_Count'] <=
                                            source_bac_with_can_target['Neighbor_Shared_Count'])]

        # Pivot this DataFrame to get the desired structure
        continuity_link_cost_df = \
            source_bac_with_can_target[['index' + col_source, 'index_prev', 'prob_compare']].pivot(
                index='index' + col_source, columns='index_prev', values='prob_compare')
        continuity_link_cost_df.columns.name = None
        continuity_link_cost_df.index.name = None

        if check_maintenance_for == 'target':
            # I want to measure is it possible to remove the current link of target bacterium?
            candidate_target_maintenance_cost = \
                maintain_exist_link_cost_df.loc[:, maintain_exist_link_cost_df.columns.isin(
                    source_bac_with_can_target['index_prev'].unique())]

            # don't need to convert probability to 1 - probability
            for col in candidate_target_maintenance_cost.columns.values:
                this_target_bac_maintenance_cost = candidate_target_maintenance_cost[col].dropna().iloc[0]
                continuity_link_cost_df.loc[
                    continuity_link_cost_df[col] <= this_target_bac_maintenance_cost, col] = np.nan

        elif check_maintenance_for == 'source':

            # I want to measure is it possible to remove the current link of target bacterium?
            candidate_target_maintenance_cost = \
                maintain_exist_link_cost_df.loc[maintain_exist_link_cost_df.index.isin(
                    source_bac_with_can_target['index' + col_source].unique())]

            selected_candidate_target_maintenance_cost = \
                candidate_target_maintenance_cost[candidate_target_maintenance_cost.min(axis=1) > 0.5]

            for source_idx in selected_candidate_target_maintenance_cost.index.values:
                max_probability_value = candidate_target_maintenance_cost.loc[source_idx].max()

                continuity_link_cost_df.loc[
                    source_idx, continuity_link_cost_df.loc[source_idx] <= max_probability_value] = np.nan

        continuity_link_cost_df = 1 - continuity_link_cost_df

    else:
        continuity_link_cost_df = pd.DataFrame()

    return continuity_link_cost_df


def calc_division_link_cost_for_restoring_links(df, neighbors_df, neighbor_list_array, df_source_daughter,
                                                center_coord_cols, col_source, col_target, parent_image_number_col,
                                                parent_object_number_col, division_links_model, coordinate_array):

    """
    Calculates the cost of restoring division links between a source bacterium and its potential daughter
    bacteria (UB bacteria). This function evaluates potential division links by calculating various spatial,
    geometric, and neighborhood-based features.
    It uses machine learning model to assess the likelihood of a division event based on these features.

    **Behavior**:
    - Calculates features like intersection-over-union (IoU), centroid distance, length ratios, and neighbor overlap.
    - Predicts division probabilities using a pre-trained model (`division_links_model`).

    :param pandas.DataFrame df:
        Full DataFrame containing bacterial features for the current time step.
    :param pandas.DataFrame neighbors_df:
        DataFrame describing relationships between neighboring bacteria.
    :param lil_matrix neighbor_list_array:
        Sparse matrix representing neighbor relationships.
    :param pandas.DataFrame df_source_daughter:
        DataFrame containing the source bacteria and their candidate daughters.
    :param dict center_coord_cols:
        Dictionary specifying column names for bacterial centroid coordinates
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param str col_source:
        Suffix for source bacterium columns.
    :param str col_target:
        Suffix for target bacterium (daughter) columns.
    :param str parent_image_number_col:
        Column name for the parent image number.
    :param str parent_object_number_col:
        Column name for the parent object number.
    :param sklearn.Model division_links_model:
        Machine learning model to predict division probabilities.
    :param csr_matrix coordinate_array:
        spatial coordinate matrix for neighborhood and overlap calculations.

    :returns:
        pandas.DataFrame:

        A cost matrix for potential division links, with rows as source bacteria and columns as their
        candidate daughters.
    """

    df_source_daughter['index_prev' + col_target] = df_source_daughter['index' + col_target]
    df_source_daughter['index2' + col_target] = df_source_daughter.index.values

    df_source_daughter['Length_Change_Ratio' + col_target] = \
        (df_source_daughter['AreaShape_MajorAxisLength' + col_target] /
         df_source_daughter['AreaShape_MajorAxisLength' + col_source])

    df_source_daughter = calculate_iou_ml(df_source_daughter, col_source='prev_index' + col_source,
                                          col_target='prev_index' + col_target, link_type='div',
                                          coordinate_array=coordinate_array, both=True)

    df_source_daughter = calc_min_distance_ml(df_source_daughter, center_coord_cols, postfix_target=col_target,
                                              postfix_source=col_source, link_type='div')

    df_source_daughter['prev_time_step_index' + col_target] = \
        df_source_daughter['index' + col_source].fillna(-1).astype('int64')

    df_source_daughter['Neighbor_Difference_Count' + col_target] = np.nan
    df_source_daughter['other_daughter_index' + col_target] = np.nan
    df_source_daughter['parent_id' + col_target] = df_source_daughter['id' + col_source]

    df_source_daughter = \
        compare_neighbor_sets(df_source_daughter, neighbor_list_array,
                              parent_image_number_col, parent_object_number_col,
                              selected_rows_df=df_source_daughter, selected_time_step_df=df,
                              return_common_elements=True, col_target=col_target)

    df_source_daughter['diff_common_diff' + col_target] = \
        df_source_daughter['Neighbor_Difference_Count' + col_target] - \
        df_source_daughter['Neighbor_Shared_Count' + col_target]

    df_source_daughter.loc[df_source_daughter['diff_common_diff' + col_target] < 0, 'diff_common_diff' + col_target] = 0

    df_source_daughter['angle_mother_daughter' + col_target] = \
        calculate_angles_between_slopes(df_source_daughter['Bacterium_Slope' + col_source].values,
                                        df_source_daughter['Bacterium_Slope' + col_target].values)

    df_source_daughter['Prev_Center_X' + col_target] = \
        df_source_daughter[center_coord_cols['x'] + col_source]
    df_source_daughter['Prev_Center_Y' + col_target] = \
        df_source_daughter[center_coord_cols['y'] + col_source]

    direction_of_motion = calculate_trajectory_angles(df_source_daughter,
                                                      center_coord_cols, suffix1=col_target)

    df_source_daughter['Direction_of_Motion' + col_target] = direction_of_motion

    # motion alignment
    # calculated for original df and we should calc for new df
    df_source_daughter["Motion_Alignment_Angle" + col_target] = np.nan
    df_source_daughter = \
        calc_motion_alignment_angle_ml(df, neighbors_df, center_coord_cols,
                                       selected_rows=df_source_daughter, col_target=col_target,
                                       col_source=col_source)

    df_source_daughter['adjusted_Neighbor_Shared_Count' + col_target] = np.where(
        df_source_daughter['Neighbor_Shared_Count' + col_target] == 0,
        df_source_daughter['Neighbor_Shared_Count' + col_target] + 1,
        df_source_daughter['Neighbor_Shared_Count' + col_target]
    )

    df_source_daughter['neighbor_ratio' + col_target] = \
        (df_source_daughter['Neighbor_Difference_Count' + col_target] / (
            df_source_daughter['adjusted_Neighbor_Shared_Count' + col_target]))

    raw_features = ['iou', 'iou_continuity', 'min_distance', 'min_distance_continuity',
                    'Neighbor_Difference_Count' + col_target,
                    'Neighbor_Shared_Count' + col_target, 'neighbor_ratio' + col_target,
                    'angle_mother_daughter' + col_target, 'Direction_of_Motion' + col_target,
                    'Motion_Alignment_Angle' + col_target, 'Length_Change_Ratio' + col_target, 'index' + col_source,
                    'index_prev' + col_target, 'divideFlag' + col_target, 'LifeHistory' + col_target,
                    'age' + col_target, 'diff_common_diff' + col_target]

    df_source_daughter = df_source_daughter[raw_features].copy()

    df_source_daughter = df_source_daughter.rename(
        {
            'Neighbor_Difference_Count' + col_target: 'Neighbor_Difference_Count',
            'Neighbor_Shared_Count' + col_target: 'Neighbor_Shared_Count',
            'neighbor_ratio' + col_target: 'neighbor_ratio',
            'angle_mother_daughter' + col_target: 'angle_mother_daughter',
            'Direction_of_Motion' + col_target: 'Direction_of_Motion',
            'Motion_Alignment_Angle' + col_target: 'Motion_Alignment_Angle',
            'Length_Change_Ratio' + col_target: 'Length_Change_Ratio',
            'index_prev' + col_target: 'index_prev',
            'divideFlag' + col_target: 'divideFlag',
            'LifeHistory' + col_target: 'LifeHistory',
            'age' + col_target: 'age',
            'diff_common_diff' + col_target: 'diff_common_diff'
        }, axis=1)

    # 'Neighbor_Difference_Count'
    feature_list_division_links_model = ['iou', 'min_distance', 'neighbor_ratio', 'angle_mother_daughter']

    y_prob_division_links_model = \
        division_links_model.predict_proba(df_source_daughter[feature_list_division_links_model])[:, 1]
    df_source_daughter['prob_divided_bac_model'] = y_prob_division_links_model

    df_source_daughter = df_source_daughter.loc[df_source_daughter['prob_divided_bac_model'] > 0.5]

    if df_source_daughter.shape[0] > 0:
        # Pivot this DataFrame to get the desired structure
        division_cost_df = \
            df_source_daughter[['index' + col_source, 'index_prev', 'prob_divided_bac_model']].pivot(
                index='index' + col_source, columns='index_prev', values='prob_divided_bac_model')
        division_cost_df.columns.name = None
        division_cost_df.index.name = None

        division_cost_df = 1 - division_cost_df

    else:
        division_cost_df = pd.DataFrame()

    return division_cost_df


def calc_continuity_link_cost_for_restoring_links(df, neighbors_df, neighbor_list_array, source_bac_with_can_target,
                                                  center_coord_cols, col_source, col_target,
                                                  parent_image_number_col, parent_object_number_col,
                                                  continuity_links_model, coordinate_array=None):

    """
    Calculates the cost of restoring continuity link between a source bacterium and its potential UB target bacterium.
    This function evaluates continuity links by analyzing spatial, geometric, and neighborhood-based features.
    It determines the likelihood that a bacterium in one time step corresponds to the same bacterium in the next
    time step.

    **Behavior**:
    - Calculates features like intersection-over-union (IoU), centroid distance, length ratios, and neighbor overlap.
    - Predicts continuity probabilities using a pre-trained model (`continuity_links_model`).

    :param pandas.DataFrame df:
        Full DataFrame containing bacterial features for the current time step.
    :param pandas.DataFrame neighbors_df:
        DataFrame describing relationships between neighboring bacteria.
    :param lil_matrix neighbor_list_array:
        Sparse matrix representing neighbor relationships.
    :param pandas.DataFrame source_bac_with_can_target:
        DataFrame containing the source bacteria and their candidate UB target in the next time step.
    :param dict center_coord_cols:
        Dictionary specifying column names for bacterial centroid coordinates
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param str col_source:
        Suffix for source bacterium columns.
    :param str col_target:
        Suffix for target bacterium columns.
    :param str parent_image_number_col:
        Column name for the parent image number.
    :param str parent_object_number_col:
        Column name for the parent object number.
    :param sklearn.Model continuity_links_model:
        Machine learning model to predict continuity probabilities.`.
    :param csr_matrix coordinate_array:
        spatial coordinate matrix for neighborhood and overlap calculations.

    :returns:
        pandas.DataFrame:

        A cost matrix for potential continuity links, with rows as source bacteria and columns as their
        candidate targets (UB bacteria).
    """

    source_bac_with_can_target['index_prev' + col_target] = source_bac_with_can_target['index' + col_target]
    source_bac_with_can_target['index2' + col_target] = source_bac_with_can_target.index.values

    source_bac_with_can_target['Length_Change_Ratio' + col_target] = (
            source_bac_with_can_target['AreaShape_MajorAxisLength' + col_target] /
            source_bac_with_can_target['AreaShape_MajorAxisLength' + col_source])

    if source_bac_with_can_target.shape[0] > 0:

        source_bac_with_can_target = calculate_iou_ml(source_bac_with_can_target, col_source='prev_index' + col_source,
                                                      col_target='prev_index' + col_target, link_type='continuity',
                                                      coordinate_array=coordinate_array)

        source_bac_with_can_target = calc_min_distance_ml(source_bac_with_can_target, center_coord_cols,
                                                          postfix_target=col_target, postfix_source=col_source,
                                                          link_type=None)

        source_bac_with_can_target['prev_time_step_index' + col_target] = \
            source_bac_with_can_target['index' + col_source].fillna(-1).astype('int64')

        source_bac_with_can_target['Neighbor_Difference_Count' + col_target] = np.nan
        source_bac_with_can_target['other_daughter_index' + col_target] = np.nan
        source_bac_with_can_target['parent_id' + col_target] = source_bac_with_can_target['id' + col_source]

        source_bac_with_can_target = \
            compare_neighbor_sets(source_bac_with_can_target, neighbor_list_array,
                                  parent_image_number_col, parent_object_number_col,
                                  selected_rows_df=source_bac_with_can_target, selected_time_step_df=df,
                                  return_common_elements=True, col_target=col_target)

        source_bac_with_can_target = check_len_ratio(df, source_bac_with_can_target, col_target=col_target,
                                                     col_source=col_source)

        # motion alignment
        # calculated for original df and we should calc for new df
        source_bac_with_can_target["Motion_Alignment_Angle" + col_target] = np.nan
        source_bac_with_can_target = \
            calc_motion_alignment_angle_ml(df, neighbors_df, center_coord_cols,
                                           selected_rows=source_bac_with_can_target, col_target=col_target,
                                           col_source=col_source)

        source_bac_with_can_target['Prev_Center_X' + col_target] = \
            source_bac_with_can_target[center_coord_cols['x'] + col_source]
        source_bac_with_can_target['Prev_Center_Y' + col_target] = \
            source_bac_with_can_target[center_coord_cols['y'] + col_source]

        direction_of_motion = calculate_trajectory_angles(source_bac_with_can_target,
                                                          center_coord_cols, suffix1=col_target)

        source_bac_with_can_target['Direction_of_Motion' + col_target] = direction_of_motion

        source_bac_with_can_target['adjusted_Neighbor_Shared_Count' + col_target] = np.where(
            source_bac_with_can_target['Neighbor_Shared_Count' + col_target] == 0,
            source_bac_with_can_target['Neighbor_Shared_Count' + col_target] + 1,
            source_bac_with_can_target['Neighbor_Shared_Count' + col_target]
        )

        source_bac_with_can_target['neighbor_ratio' + col_target] = \
            (source_bac_with_can_target['Neighbor_Difference_Count' + col_target] / (
                source_bac_with_can_target['adjusted_Neighbor_Shared_Count' + col_target]))

        raw_feature_list = ['iou', 'min_distance', 'Neighbor_Difference_Count' + col_target,
                            'Neighbor_Shared_Count' + col_target,
                            'length_dynamic' + col_target,
                            'neighbor_ratio' + col_target,
                            'Motion_Alignment_Angle' + col_target,
                            'Direction_of_Motion' + col_target, 'Length_Change_Ratio' + col_target,
                            'index' + col_source, 'index_prev' + col_target]

        source_bac_with_can_target = source_bac_with_can_target[raw_feature_list].copy()
        source_bac_with_can_target = source_bac_with_can_target.rename(
            {
                'Neighbor_Shared_Count' + col_target: 'Neighbor_Shared_Count',
                'Neighbor_Difference_Count' + col_target: 'Neighbor_Difference_Count',
                'neighbor_ratio' + col_target: 'neighbor_ratio',
                'length_dynamic' + col_target: 'length_dynamic',
                'Motion_Alignment_Angle' + col_target: 'Motion_Alignment_Angle',
                'Direction_of_Motion' + col_target: 'Direction_of_Motion',
                'Length_Change_Ratio' + col_target: 'Length_Change_Ratio',
                'index_prev' + col_target: 'index_prev',
            }, axis=1)

        # difference_neighbors
        feature_list_for_continuity_links_model = \
            ['iou', 'min_distance', 'neighbor_ratio', 'length_dynamic', 'Motion_Alignment_Angle']

        y_prob_continuity_links_model = \
            continuity_links_model.predict_proba(source_bac_with_can_target[
                                                    feature_list_for_continuity_links_model])[:, 1]
        source_bac_with_can_target['prob_non_divided_bac_model'] = y_prob_continuity_links_model

        source_bac_with_can_target = \
            source_bac_with_can_target.loc[(source_bac_with_can_target['prob_non_divided_bac_model'] > 0.5) &
                                           (source_bac_with_can_target['Neighbor_Difference_Count'] <=
                                            source_bac_with_can_target['Neighbor_Shared_Count'])]

        # Pivot this DataFrame to get the desired structure
        continuity_link_cost_df = \
            source_bac_with_can_target[['index' + col_source, 'index_prev', 'prob_non_divided_bac_model']].pivot(
                index='index' + col_source, columns='index_prev', values='prob_non_divided_bac_model')
        continuity_link_cost_df.columns.name = None
        continuity_link_cost_df.index.name = None

        continuity_link_cost_df = 1 - continuity_link_cost_df

    else:
        continuity_link_cost_df = pd.DataFrame()

    return continuity_link_cost_df
