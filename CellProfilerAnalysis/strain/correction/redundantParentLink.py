import numpy as np
from CellProfilerAnalysis.strain.correction.action.helperFunctions import distance_normalization
from CellProfilerAnalysis.strain.correction.action.compareBacteria import compare_bacteria
from CellProfilerAnalysis.strain.correction.action.findOutlier import find_daughter_len_to_mother_ratio_outliers, \
    find_bac_len_to_bac_ratio_boundary
from CellProfilerAnalysis.strain.correction.action.bacteriaModification import remove_redundant_link


def detect_and_remove_redundant_parent_link(dataframe, sorted_npy_files_list, um_per_pixel):

    # note: it's only for division (mother - daughters relation)
    # check daughter length (sum daughters length or max daughter length) to mother length
    daughters_to_mother_ratio_list_outliers = find_daughter_len_to_mother_ratio_outliers(dataframe)

    bacteria_with_redundant_parent_link_error = \
        dataframe.loc[(dataframe["daughter_length_to_mother"].isin(
            daughters_to_mother_ratio_list_outliers['daughter_length_to_mother'])) |
                      (dataframe["max_daughter_len_to_mother"].isin(
                          daughters_to_mother_ratio_list_outliers['max_daughter_len_to_mother']))
                      ]

    # print("number of redundant parent link error: ")
    # print(bacteria_with_redundant_parent_link_error.shape[0])
    # print("more information: ")
    # print(bacteria_with_redundant_parent_link_error)

    bac_len_to_bac_ratio_boundary = find_bac_len_to_bac_ratio_boundary(dataframe)

    for parent_indx, parent_with_redundant_link in bacteria_with_redundant_parent_link_error.iterrows():

        current_img_npy = sorted_npy_files_list[parent_with_redundant_link['ImageNumber'] - 1]

        bacteria_in_current_time_step = dataframe.loc[
            dataframe['ImageNumber'] == parent_with_redundant_link['ImageNumber']]

        daughters_at_first_time_step_of_life_history = \
            dataframe.loc[(dataframe['parent_id'] == dataframe.iloc[parent_indx]['id']) &
                          (dataframe['ImageNumber'] == parent_with_redundant_link['ImageNumber'] + 1)]

        bac_len_to_bac_ratio_df = daughters_at_first_time_step_of_life_history['AreaShape_MajorAxisLength'] / \
                                  parent_with_redundant_link['AreaShape_MajorAxisLength']

        candidate_daughters_at_first_time_step_of_life_history = \
            daughters_at_first_time_step_of_life_history[bac_len_to_bac_ratio_df >=
                                                         bac_len_to_bac_ratio_boundary['avg'] -
                                                         1.96 * bac_len_to_bac_ratio_boundary['std']]

        incorrect_daughters_at_first_time_step_of_life_history = \
            daughters_at_first_time_step_of_life_history[bac_len_to_bac_ratio_df <
                                                         bac_len_to_bac_ratio_boundary['avg'] -
                                                         1.96 * bac_len_to_bac_ratio_boundary['std']]

        if candidate_daughters_at_first_time_step_of_life_history.shape[0] > 1:

            bacteria_in_next_time_step = dataframe.loc[dataframe['ImageNumber'] ==
                                                       parent_with_redundant_link['ImageNumber'] + 1]
            next_img_npy = sorted_npy_files_list[parent_with_redundant_link['ImageNumber']]

            # check the cost of daughters to mother
            overlap_df, distance_df = compare_bacteria(current_img_npy, bacteria_in_current_time_step,
                                                       parent_with_redundant_link.to_frame().transpose(), next_img_npy,
                                                       bacteria_in_next_time_step,
                                                       daughters_at_first_time_step_of_life_history, um_per_pixel)

            normalized_distance_df = distance_normalization(dataframe, distance_df)

            cost_df = np.sqrt((1 - overlap_df)**2 + normalized_distance_df**2)

            wrong_daughter_index = cost_df.max().idxmax()

            wrong_daughter_life_history = dataframe.loc[dataframe['id'] == dataframe.iloc[wrong_daughter_index]['id']]

        else:
            wrong_daughter_life_history = dataframe.loc[dataframe['id'] ==
                                                        incorrect_daughters_at_first_time_step_of_life_history['id'].values.tolist()[0]]

        dataframe = remove_redundant_link(dataframe, wrong_daughter_life_history)

    return dataframe
