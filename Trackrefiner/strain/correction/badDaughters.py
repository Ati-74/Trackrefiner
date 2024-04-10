import pandas as pd
import numpy as np
from Trackrefiner.strain.correction.action.bacteriaModification import remove_redundant_link
from Trackrefiner.strain.correction.action.compareBacteria import make_initial_distance_matrix
from Trackrefiner.strain.correction.action.helperFunctions import distance_normalization
from Trackrefiner.strain.correction.action.helperFunctions import calculate_orientation_angle



def detect_remove_bad_daughters_to_mother_link(df, neighbor_df, parent_image_number_col, parent_object_number_col,
                                               label_col, center_coordinate_columns, logs_df):
    """
        goal: modification of bad daughters (try to assign bad daughters to new parent)
        @param df    dataframe   bacteria dataframe
        in last time step of its life history before transition bacterium to length of candidate parent bacterium
        in investigated time step
        output: df   dataframe   modified dataframe (without bad daughters)
    """

    bad_daughters_list = df.loc[(df['bad_division_flag'] == True) & (df['noise_bac'] == False)]['daughters_index'].drop_duplicates()

    for bad_daughters_index_list in bad_daughters_list:

        daughters_df = df.iloc[bad_daughters_index_list]
        mother_df = df.loc[df['id'] == daughters_df['parent_id'].tolist()[0]]

        mother_last_time_step = mother_df.loc[mother_df['ImageNumber'] == mother_df['ImageNumber'].max()]

        bacteria_in_mother_last_time_step = df.loc[df['ImageNumber'] == mother_df['ImageNumber'].max()]

        bacteria_in_daughter_time_step = df.loc[df['ImageNumber'] == daughters_df['ImageNumber'].values.tolist()[0]]

        # check the cost of daughters to mother
        overlap_df, distance_df = make_initial_distance_matrix(bacteria_in_mother_last_time_step,
                                                               mother_last_time_step, bacteria_in_daughter_time_step,
                                                               daughters_df, center_coordinate_columns)

        normalized_distance_df = distance_normalization(df, distance_df)

        cost_df = np.sqrt((1 - overlap_df) ** 2 + normalized_distance_df ** 2)

        # Calculate orientation angle between mother and each daughter
        for daughter_ndx, daughter in daughters_df.iterrows():
            # Calculate orientation angle
            orientation_angle = calculate_orientation_angle(mother_last_time_step['bacteria_slope'].values.tolist()[0],
                                                            daughter['bacteria_slope'])
            cost_df[daughter_ndx] = np.sqrt(np.power(cost_df[daughter_ndx], 2) + np.power(orientation_angle, 2))

        while cost_df.shape[1] > 2:
            wrong_daughter_index = cost_df.max().idxmax()
            cost_df.drop(wrong_daughter_index, axis=1, inplace=True)

            wrong_daughter_life_history = df.loc[df['id'] == df.iloc[wrong_daughter_index]['id']]

            df = remove_redundant_link(df, wrong_daughter_life_history, neighbor_df, parent_image_number_col,
                          parent_object_number_col, label_col, center_coordinate_columns)
            logs_df = pd.concat([logs_df, wrong_daughter_life_history.iloc[0].to_frame().transpose()],
                                ignore_index=True)

    # change the value of bad division flag
    df['bad_division_flag'] = False

    return df, logs_df
