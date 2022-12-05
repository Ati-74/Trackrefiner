import numpy as np
from CellProfilerAnalysis.strain.correction.action.processing import k_nearest_neighbors


def check_fluorescent_intensity(transition_bac, candidate_parent_bac):
    """
    goal: does the candidate parent has an appropriate
    fluorescent intensity pattern to be the parent of the transition bacterium?
    """
    appropriate_intensity_pattern = False

    if len(set(candidate_parent_bac['cellType'])) == 1:
        # means: all elements value = 0 or 1
        appropriate_intensity_pattern = True
    else:
        sum_intensity = [sum(x) for x in zip(transition_bac['cellType'], candidate_parent_bac['cellType'])]
        if 2 in sum_intensity:
            # bacteria have at least one candidate cell type in common
            appropriate_intensity_pattern = True

    return appropriate_intensity_pattern


def probability_cell_type(df, cols, intensity_threshold):
    """
    @param df dataframe features value of bacteria in each time step
    @param cols list intensity columns name
    @param intensity_threshold float min intensity value of channel
    """

    for bac_row_index, bac_row in df[cols].iterrows():
        # initial value: all intensity column values are <= intensity_threshold
        probability_list = [0] * len(cols)

        candidate_cell_type_index = [i for i, elem in enumerate(bac_row.values.tolist()) if elem > intensity_threshold]

        for index in candidate_cell_type_index:
            probability_list[index] = 1

        df.at[bac_row_index, 'cellType'] = probability_list

    return df


def check_intensity(dataframe_col):

    """
    If the CSV file has two mean intensity columns, the cosine similarity is calculated
    @param dataframe_col dataframe features value of bacteria in each time step
    """
    # fluorescence intensity columns
    fluorescence_intensities_col = [col for col in dataframe_col if col.startswith('Intensity_MeanIntensity')]

    return fluorescence_intensities_col


def assign_cell_type(dataframe, intensity_threshold):
    # If the CSV file has two mean intensity columns, the cosine similarity is calculated
    intensity_col_names = check_intensity(dataframe.columns)
    # cell type
    if len(intensity_col_names) >= 1:
        dataframe['cellType'] = [[0] * len(intensity_col_names)] * len(dataframe)
        # check  fluorescence intensity
        dataframe = probability_cell_type(dataframe, intensity_col_names, intensity_threshold)
    else:
        dataframe['cellType'] = [[0] * (len(intensity_col_names) + 1)] * len(dataframe)

    return dataframe


def fix_cell_type_error(dataframe):

    df_bacteria_cell_type_errors = dataframe.loc[dataframe['unknown_cell_type'] == True]
    bacteria_labels = df_bacteria_cell_type_errors['TrackObjects_Label_50'].unique()

    for label in bacteria_labels:
        bacteria_family_tree = dataframe.loc[dataframe['TrackObjects_Label_50'] == label]
        root_bacterium = bacteria_family_tree.iloc[[0]]

        other_same_time_step_bacteria = dataframe.loc[(dataframe['ImageNumber'] == root_bacterium.iloc[0]['ImageNumber']) &
                                                      (dataframe['unknown_cell_type'] == False)]

        nearest_bacteria_index = k_nearest_neighbors(1, root_bacterium, other_same_time_step_bacteria,
                                                     distance_check=False)[0]
        final_cell_type_value = dataframe.iloc[nearest_bacteria_index]['cellType']
        for idx in bacteria_family_tree.index:
            dataframe.at[idx, 'cellType'] = final_cell_type_value

    dataframe.drop(labels='unknown_cell_type', axis=1, inplace=True)

    return dataframe


def final_cell_type(dataframe):

    dataframe['unknown_cell_type'] = False
    num_intensity_cols = len(check_intensity(dataframe.columns))

    if num_intensity_cols >1:

        bacteria_labels = dataframe['TrackObjects_Label_50'].unique()

        for label in bacteria_labels:
            bacteria_family_tree = dataframe.loc[dataframe['TrackObjects_Label_50'] == label]
            bacteria_family_tree_cell_types = bacteria_family_tree['cellType'].values.tolist()
            frequency = []
            for i in range(num_intensity_cols):
                frequency_ith_channel = len([elem for elem in bacteria_family_tree_cell_types if elem[i] == 1])
                frequency.append(frequency_ith_channel)
            if len(set(frequency)) > 1:
                for idx in bacteria_family_tree.index:
                    dataframe.at[idx, 'cellType'] = np.argmax(frequency) + 1
            else:
                for idx in bacteria_family_tree.index:
                    dataframe.at[idx, 'unknown_cell_type'] = True

        dataframe = fix_cell_type_error(dataframe)
    else:
        dataframe['cellType'] = 1

    return dataframe