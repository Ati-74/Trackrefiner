from CellProfilerAnalysis.strain.correction.action.processing import bacteria_in_specific_time_step, adjacency_matrix
from CellProfilerAnalysis.strain.correction.action.AssignParent import assign_parent


def correction_bad_daughters(df, number_of_gap=0, distance_threshold=5, proportion_of_length_threshold=0.85,
                             check_cellType=True):
    """
        goal: modification of bad daughters (try to assign bad daughters to new parent)
        @param df    dataframe   bacteria dataframe
        @param number_of_gap int I define a gap number to find parent in other previous time steps
        @param distance_threshold float(unit: um) maximum distance between candidate parent and target bacterium
        @param proportion_of_length_threshold float proportion length of candidate parent bacterium
        in last time step of its life history before transition bacterium to length of candidate parent bacterium
        in investigated time step
        output: df   dataframe   modified dataframe (without bad daughters)
    """

    df['drop'] = False

    bad_daughters_list = df.loc[(df['bad_division_flag'] == True) & (df['transition_drop'] == False)][
        'bad_daughters_index'].unique()

    for bad_daughters in bad_daughters_list:
        bad_daughters_df = df.iloc[bad_daughters]
        bad_daughters_time_step = bad_daughters_df['ImageNumber'].values.tolist()[0]

        # I define this list to store adjacency matrices
        distance_df_list = []

        for time_step_under_invest in range(max(bad_daughters_time_step - number_of_gap - 1, 1), bad_daughters_time_step):
            # filter consider time step bacteria information
            bacteria_time_step_under_invest = bacteria_in_specific_time_step(df, time_step_under_invest)
            bacteria_time_step_under_invest = bacteria_time_step_under_invest.loc[
                    (bacteria_time_step_under_invest['division_time'] == 0) |
                    (bacteria_time_step_under_invest['division_time'] > bad_daughters_time_step + 1)]

            # create distance matrix (rows: bad daughters, columns: consider time step bacteria)
            try:
                distance_df = adjacency_matrix(bad_daughters_df, bacteria_time_step_under_invest,
                                               'Location_Center_X', 'Location_Center_Y')
            except TypeError:
                distance_df = adjacency_matrix(bad_daughters_df, bacteria_time_step_under_invest,
                                               'AreaShape_Center_X', 'AreaShape_Center_Y')
            distance_df_list.append(distance_df)

            # find the parent of each transition bacterium in the next time step
            for daughter_bac_index in bad_daughters:
                df = assign_parent(df, daughter_bac_index, distance_df_list, distance_threshold,
                                   proportion_of_length_threshold, check_cellType)

    # rename drop column
    df.rename(columns={'drop': 'bad_daughter_drop'}, inplace=True)

    return df
