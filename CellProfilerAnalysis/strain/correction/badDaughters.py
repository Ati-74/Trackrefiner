from CellProfilerAnalysis.strain.correction.action.bacteriaModification import remove_redundant_link
from CellProfilerAnalysis.strain.correction.action.compareBacteria import compare_bacteria
from CellProfilerAnalysis.strain.correction.action.helperFunctions import distance_normalization
from CellProfilerAnalysis.strain.correction.action.helperFunctions import calculate_orientation_angle
import numpy as np


def detect_remove_bad_daughters_to_mother_link(df, sorted_npy_files_list, um_per_pixel):
    """
        goal: modification of bad daughters (try to assign bad daughters to new parent)
        @param df    dataframe   bacteria dataframe
        in last time step of its life history before transition bacterium to length of candidate parent bacterium
        in investigated time step
        output: df   dataframe   modified dataframe (without bad daughters)
    """

    bad_daughters_list = df.loc[df['bad_division_flag'] == True]['daughters_index'].drop_duplicates()

    # print("number of bad daughters: ")
    # print(bad_daughters_list.shape[0])

    # if bad_daughters_list.shape[0] > 0:
    #     print('more information: ')
    #    print(bad_daughters_list)

    for bad_daughters_index_list in bad_daughters_list:

        daughters_df = df.iloc[bad_daughters_index_list]
        mother_df = df.loc[df['id'] == daughters_df['parent_id'].tolist()[0]]

        mother_last_time_step = mother_df.loc[mother_df['ImageNumber'] == mother_df['ImageNumber'].max()]

        last_time_step_mother_img_npy = sorted_npy_files_list[mother_df['ImageNumber'].max() - 1]

        bacteria_in_mother_last_time_step = df.loc[df['ImageNumber'] == mother_df['ImageNumber'].max()]

        bacteria_in_daughter_time_step = df.loc[df['ImageNumber'] == daughters_df['ImageNumber'].values.tolist()[0] - 1]
        daughter_time_step_img_npy = sorted_npy_files_list[daughters_df['ImageNumber'].values.tolist()[0] - 1]

        # check the cost of daughters to mother
        overlap_df, distance_df = compare_bacteria(last_time_step_mother_img_npy, bacteria_in_mother_last_time_step,
                                                   mother_last_time_step, daughter_time_step_img_npy,
                                                   bacteria_in_daughter_time_step,
                                                   daughters_df, um_per_pixel)

        normalized_distance_df = distance_normalization(df, distance_df)

        cost_df = np.sqrt((1 - overlap_df) ** 2 + normalized_distance_df ** 2)

        # Calculate orientation angle between mother and each daughter
        for daughter_ndx, daughter in daughters_df.iterrows():
            # Calculate orientation angle
            orientation_angle = calculate_orientation_angle(mother_last_time_step['bacteria_slope'].values.tolist()[0],
                                                            daughter['bacteria_slope'])
            cost_df[daughter_ndx] += orientation_angle

        while cost_df.shape[1] > 2:
            wrong_daughter_index = cost_df.max().idxmax()
            cost_df.drop(wrong_daughter_index, axis=1, inplace=True)

            wrong_daughter_life_history = df.loc[df['id'] == df.iloc[wrong_daughter_index]['id']]

            df = remove_redundant_link(df, wrong_daughter_life_history)

    # change the value of bad division flag
    df['bad_division_flag'] = False

    return df
