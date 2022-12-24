from CellProfilerAnalysis.strain.correction.action.processing import bacteria_in_specific_time_step, adjacency_matrix
from CellProfilerAnalysis.strain.correction.action.AssignParent import assign_parent


def correction_transition(raw_df, number_of_gap=0, distance_threshold=5, proportion_of_length_threshold=0.85):
    """
        goal: For bacteria without parent, assign labels, ParentImageNumber, and ParentObjectNumber
        @param raw_df    dataframe   bacteria dataframe
        @param number_of_gap int I define a gap number to find parent in other previous time steps
        @param distance_threshold float(unit: um) maximum distance between candidate parent and target bacterium
        @param proportion_of_length_threshold float proportion length of candidate parent bacterium
        in last time step of its life history before transition bacterium to length of candidate parent bacterium
        in investigated time step
        output: df   dataframe   modified dataframe (without any transitions)
    """

    # add new column to dataframe
    # During processing, I need it to find bacteria that have no parents (and remove them at the end)
    raw_df["drop"] = False

    # now, I try to resolve transition problem
    # list of unique time steps
    time_steps_list = sorted(raw_df['ImageNumber'].unique())

    for current_time_step in time_steps_list[:-1]:
        # time step range: 1 to (last - 1)

        # filter next time step bacteria information
        next_time_step = current_time_step + 1
        next_timestep_bac = bacteria_in_specific_time_step(raw_df, next_time_step)

        # filter transition bacteria features value in the next time step (bacteria without parent)
        next_timestep_transition_bac = next_timestep_bac.loc[next_timestep_bac["transition"] == True]

        if next_timestep_transition_bac.shape[0] > 0:

            # I define this list to store adjacency matrices
            distance_df_list = []

            for time_step_under_invest in range(max(current_time_step - number_of_gap, 1), current_time_step + 1):

                # filter consider time step bacteria information
                bacteria_time_step_under_invest = bacteria_in_specific_time_step(raw_df, time_step_under_invest)
                bacteria_time_step_under_invest = bacteria_time_step_under_invest.loc[
                    (bacteria_time_step_under_invest['division_time'] == 0) |
                    (bacteria_time_step_under_invest['division_time'] > next_time_step + 1)]

                # create distance matrix (rows: next time step sudden bacteria, columns: consider time step bacteria)
                try:
                    distance_df = adjacency_matrix(next_timestep_transition_bac, bacteria_time_step_under_invest,
                                                   'Location_Center_X', 'Location_Center_Y')
                except TypeError:
                    distance_df = adjacency_matrix(next_timestep_transition_bac, bacteria_time_step_under_invest,
                                                   'AreaShape_Center_X', 'AreaShape_Center_Y')

                distance_df_list.append(distance_df)

            # find the parent of each transition bacterium in the next time step
            for transition_bac_index, transition_bac in next_timestep_transition_bac.iterrows():

                raw_df = assign_parent(raw_df, transition_bac_index, distance_df_list, distance_threshold,
                                       proportion_of_length_threshold)

    # rename drop column
    raw_df.rename(columns={'drop': 'transition_drop'}, inplace=True)
    return raw_df
