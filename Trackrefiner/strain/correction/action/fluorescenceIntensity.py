import numpy as np
from Trackrefiner.strain.correction.action.helperFunctions import k_nearest_neighbors


def check_fluorescent_intensity(unexpected_beginning_bac, candidate_parent_bac):
    """
    goal: does the candidate parent has an appropriate
    fluorescent intensity pattern to be the parent of the unexpected_beginning bacterium?
    """
    appropriate_intensity_pattern = False

    if len(set(candidate_parent_bac['cellType'])) == 1:
        # means: all elements value = 0 or 1
        appropriate_intensity_pattern = True
    elif len(set(unexpected_beginning_bac['cellType'])) == 1:
        # means: all elements value = 0 or 1
        appropriate_intensity_pattern = True
    else:
        sum_intensity = [sum(x) for x in zip(unexpected_beginning_bac['cellType'], candidate_parent_bac['cellType'])]
        if 2 in sum_intensity:
            # bacteria have at least one candidate cell type in common
            appropriate_intensity_pattern = True

    return appropriate_intensity_pattern


def probability_cell_type(df, cell_type_array, cols, intensity_threshold):
    """
    @param df dataframe features value of bacteria in each time step
    @param cols list intensity columns name
    @param intensity_threshold float min intensity value of channel
    """

    for col_idx, col in enumerate(cols):
        condition = df[col] > intensity_threshold
        cell_type_array[condition, col_idx] = 1

    # for bac_row_index, bac_row in df[cols].iterrows():
        # initial value: all intensity column values are <= intensity_threshold
    #    probability_list = [0] * len(cols)

    #    candidate_cell_type_index = [i for i, elem in enumerate(bac_row.values.tolist()) if elem > intensity_threshold]

    #    for index in candidate_cell_type_index:
    #        probability_list[index] = 1

    #    df.at[bac_row_index, 'cellType'] = probability_list

    return cell_type_array


def check_intensity(dataframe_col):
    """
    If the CSV file has two mean intensity columns, the cosine similarity is calculated
    @param dataframe_col dataframe features value of bacteria in each time step
    """
    # fluorescence intensity columns
    fluorescence_intensities_col = \
        sorted(dataframe_col[dataframe_col.str.contains('Intensity_MeanIntensity_')].values.tolist())

    return fluorescence_intensities_col


def assign_cell_type(dataframe, intensity_threshold):
    # If the CSV file has two mean intensity columns, the cosine similarity is calculated
    intensity_col_names = check_intensity(dataframe.columns)

    # cell type
    if len(intensity_col_names) >= 1:
        # dataframe['cellType'] = [[0] * len(intensity_col_names)] * len(dataframe)
        cell_type_array = np.zeros((dataframe.shape[0], len(intensity_col_names)))
        # check  fluorescence intensity
        cell_type_array = probability_cell_type(dataframe, cell_type_array, intensity_col_names, intensity_threshold)
    else:
        # dataframe['cellType'] = [[0] * (len(intensity_col_names) + 1)] * len(dataframe)
        cell_type_array = np.zeros((dataframe.shape[0], 1))

    return cell_type_array


def fix_cell_type_error(dataframe, center_coordinate_columns, label_col):
    df_bacteria_cell_type_errors = dataframe.loc[dataframe['unknown_cell_type']]
    bacteria_labels = df_bacteria_cell_type_errors[label_col].unique()

    for label in bacteria_labels:
        bacteria_family_tree = dataframe.loc[dataframe[label_col] == label]
        root_bacterium = bacteria_family_tree.iloc[[0]]

        other_same_time_step_bacteria = \
            dataframe.loc[(dataframe['ImageNumber'] == root_bacterium.iloc[0]['ImageNumber']) &
                          (dataframe['unknown_cell_type'] == False)]

        nearest_bacteria_index = k_nearest_neighbors(root_bacterium, other_same_time_step_bacteria,
                                                     center_coordinate_columns, k=1, distance_check=False)[0]

        final_cell_type_value = dataframe.iloc[nearest_bacteria_index]['cellType']
        for idx in bacteria_family_tree.index:
            dataframe.at[idx, 'cellType'] = final_cell_type_value

    dataframe.drop(labels='unknown_cell_type', axis=1, inplace=True)

    return dataframe


def final_cell_type(dataframe, cell_type_array):

    # dataframe['unknown_cell_type'] = False
    num_intensity_cols = len(check_intensity(dataframe.columns))

    if num_intensity_cols > 1:

        dataframe['cellType'] = 0

        num_1_value_per_bac = np.sum(cell_type_array, axis=1)
        num_0_value_per_bac = cell_type_array.shape[1] - num_1_value_per_bac

        cond1_more_than_2_1_value = num_1_value_per_bac >= 2
        cond2_more_than_2_0_value = num_0_value_per_bac >= 2

        cond3 = ~ cond1_more_than_2_1_value & ~ cond2_more_than_2_0_value

        indices_of_ones_in_columns = np.where(cell_type_array[cond3] == 1)[0]
        cond3_value = indices_of_ones_in_columns + 1

        dataframe.loc[cond1_more_than_2_1_value, 'cellType'] = 3
        dataframe.loc[cond3, 'cellType'] = cond3_value

        # for bac_idx in dataframe['index'].values:
        #    bac_cell_type_list = cell_type_array[bac_idx]

        #    if bac_cell_type_list.count(1) >= 2:
        #        dataframe.at[bac_idx, 'cellType'] = 3
        #    elif bac_cell_type_list.count(0) >= 2:
        #        dataframe.at[bac_idx, 'cellType'] = 0
        #    else:
        #       dataframe.at[bac_idx, 'cellType'] = bac_cell_type_list.index(1) + 1
    else:
        dataframe['cellType'] = 1

    return dataframe
