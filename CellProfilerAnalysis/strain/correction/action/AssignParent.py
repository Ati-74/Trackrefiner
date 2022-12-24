import numpy as np
from CellProfilerAnalysis.strain.correction.action.FluorescenceIntensity import check_fluorescent_intensity
from CellProfilerAnalysis.strain.correction.action.processing import find_related_bacteria


def find_parent_in_desire_time_step(df, transition_bac, sorted_distance, distance_threshold,
                                    proportion_of_length_threshold):
    """
    goal: finding parent bacterium in previous time steps
    @param df dataframe bacteria information in each time steps
    @param transition_bac series transition bacterium features values
    @param sorted_distance series it shows the distance of bacteria from
    the considered time step to transition bacterium
    @param distance_threshold float maximum distance between candidate parent and transition bacterium
    @param proportion_of_length_threshold float proportion length of candidate parent bacterium
    in last time step of its life history before transition bacterium to length of candidate parent bacterium
    in investigated time step
    """
    was_parent_found = False
    # index of candidate parent bacteria
    bacteria_index = sorted_distance.index.values.tolist()
    # first index of bacteria_index list
    element_index = 0

    candidate_parent_stat = {'was_parent_found': False, 'candidate_parent_index': None,
                             'candidate_parent_index_in_last_time_step': None,
                             'nearest_candidate_parent_distance': None}

    nearest_candidate_parent_distance = sorted_distance.loc[bacteria_index[element_index]]

    while was_parent_found is False and element_index < len(sorted_distance) and \
            nearest_candidate_parent_distance <= distance_threshold:
        # candidate parent bacterium
        candidate_parent_bacterium = df.iloc[bacteria_index[element_index]]
        # check intensity (flag)
        appropriate_candidate_parent = check_fluorescent_intensity(transition_bac, candidate_parent_bacterium)

        if appropriate_candidate_parent:

            # candidate parent bacterium in the last time step of its life history
            # before the birth of the transition bacterium
            candidate_bac_life_history_before_transition = df.loc[(df['id'] == candidate_parent_bacterium['id']) &
                                                                  (df['ImageNumber'] <= transition_bac['ImageNumber']) &
                                                                  (df['ImageNumber'] >= candidate_parent_bacterium[
                                                                      'ImageNumber'])]

            if candidate_bac_life_history_before_transition['ImageNumber'].iloc[-1] != \
                    candidate_parent_bacterium['ImageNumber']:
                proportion_of_length = candidate_bac_life_history_before_transition['AreaShape_MajorAxisLength'].iloc[
                                           -1] / \
                                       candidate_parent_bacterium['AreaShape_MajorAxisLength']

                if proportion_of_length < proportion_of_length_threshold:
                    # it means that division has occurred but one daughter doesn't detected by Cellprofiler
                    appropriate_candidate_parent = True
            else:
                # it means that the life history of candidate parent bacterium has been continued
                appropriate_candidate_parent = True

        if not appropriate_candidate_parent:
            # check next candidate bacterium
            element_index += 1
            # row number of the nearest bacterium (candidate parent) to transition bacterium
            nearest_candidate_parent_distance = sorted_distance.loc[bacteria_index[element_index]]
        elif appropriate_candidate_parent:
            if candidate_bac_life_history_before_transition['ImageNumber'].iloc[-1] == transition_bac['ImageNumber']:
                last_parent_index_before_transition = candidate_bac_life_history_before_transition.index[-2]
            else:
                last_parent_index_before_transition = candidate_bac_life_history_before_transition.index[-1]
            was_parent_found = True
            candidate_parent_stat['was_parent_found'] = True
            candidate_parent_stat['candidate_parent_index'] = bacteria_index[element_index]
            candidate_parent_stat['candidate_parent_index_in_last_time_step'] = last_parent_index_before_transition
            candidate_parent_stat['nearest_candidate_parent_distance'] = nearest_candidate_parent_distance

    return candidate_parent_stat


def find_candidate_parents(df, target_bac_index, distance_df_list, distance_threshold, proportion_of_length_threshold):
    """
    this function calls for each target (transition or incorrect daughter) bacterium
    @param df dataframe features value of bacteria in each time step
    @param target_bac_index int row index of target bacterium
    @param distance_df_list list list of distance of target bacterium from other candidate parents
    in different time steps
    @param distance_threshold float maximum distance between candidate parent and target bacterium
    @param proportion_of_length_threshold float proportion length of candidate parent bacterium
    in last time step of its life history before transition bacterium to length of candidate parent bacterium
    in investigated time step
    """
    candidate_parents_index = []
    candidate_parents_index_in_last_time_step = []
    candidate_parents_distance_from_transition_bac = []

    transition_bacterium = df.iloc[target_bac_index]

    # was the parent found from previous time steps?
    for distance_df in distance_df_list:
        sorted_distance = distance_df.loc[target_bac_index].sort_values()

        candidate_parent_stat = find_parent_in_desire_time_step(df, transition_bacterium, sorted_distance,
                                                                distance_threshold, proportion_of_length_threshold)

        if candidate_parent_stat['was_parent_found']:
            candidate_parents_index.append(candidate_parent_stat['candidate_parent_index'])
            candidate_parents_index_in_last_time_step.append(
                candidate_parent_stat['candidate_parent_index_in_last_time_step'])
            candidate_parents_distance_from_transition_bac.append(
                candidate_parent_stat['nearest_candidate_parent_distance'])

    return candidate_parents_index, candidate_parents_index_in_last_time_step, \
           candidate_parents_distance_from_transition_bac


def assign_parent(df, target_bac_index, distance_df_list, distance_threshold, proportion_of_length_threshold):
    """
    we should find the nearest candidate bacterium in the previous time steps that has:
    similar Intensity_MeanIntensity pattern (if exist)

    @param df dataframe features value of bacteria in each time step
    @param target_bac_index int row index of target bacterium
    @param distance_df_list list list of distance of target bacteria from other candidate parents
    in different time steps
    @param distance_threshold float maximum distance between candidate parent and target bacterium
    @param proportion_of_length_threshold float proportion length of candidate parent bacterium
    in last time step of its life history before transition bacterium to length of candidate parent bacterium
    in investigated time step
    """

    candidate_parents_index, candidate_parents_index_in_last_time_step, distance_from_target_bac = \
        find_candidate_parents(df, target_bac_index, distance_df_list, distance_threshold,
                               proportion_of_length_threshold)
    # find related bacteria to this transition bacterium
    target_bacterium = df.iloc[target_bac_index]
    related_bacteria_index = find_related_bacteria(df, target_bacterium, target_bac_index, bacteria_index_list=None)

    if len(candidate_parents_index) == 0:
        # it means that no parent has been found for this bacterium
        # remove this bacterium and related bacteria
        for idx in related_bacteria_index:
            df.at[idx, "drop"] = True
    else:
        # sort distance and find the nearest parent
        nearest_parent_index = np.argmin(distance_from_target_bac)
        selected_parent_list_index = candidate_parents_index[nearest_parent_index]
        selected_parent = df.iloc[selected_parent_list_index]
        parent_life_history = df.loc[df['id'] == selected_parent['id']]
        target_bacterium_life_history = df.loc[df['id'] == target_bacterium['id']]
        target_bacterium_daughters = df.loc[df['parent_id'] == target_bacterium['id']]
        # modify info

        if parent_life_history['ImageNumber'].iloc[-1] < target_bacterium['ImageNumber']:
            # it means parent life history ended before transition bacterium was birth
            # division flag = False
            for parent_index in parent_life_history.index:
                df.at[parent_index, 'unexpected_end'] = target_bacterium['unexpected_end']
                df.at[parent_index, 'divideFlag'] = target_bacterium['divideFlag']
                df.at[parent_index, 'daughters_index'] = target_bacterium['daughters_index']
                df.at[parent_index, 'bad_division_flag'] = target_bacterium['bad_division_flag']
                df.at[parent_index, 'bad_daughters_index'] = target_bacterium['bad_daughters_index']
                df.at[parent_index, 'division_time'] = target_bacterium['division_time']
                df.at[parent_index, 'LifeHistory'] = target_bacterium["LifeHistory"] + selected_parent["LifeHistory"]

            for bac_index in target_bacterium_life_history.index:
                df.at[bac_index, 'parent_id'] = selected_parent['parent_id']
                df.at[bac_index, 'id'] = selected_parent['id']
                df.at[bac_index, "LifeHistory"] = target_bacterium["LifeHistory"] + selected_parent["LifeHistory"]

            for bac_index in target_bacterium_daughters.index:
                df.at[bac_index, 'parent_id'] = selected_parent['id']
        else:
            # this part of life history should define as new daughter
            new_daughter_life_history = \
                parent_life_history.loc[parent_life_history['ImageNumber'] >= target_bacterium['ImageNumber']]
            parent_bac_before_division = \
                parent_life_history.loc[parent_life_history['ImageNumber'] < target_bacterium['ImageNumber']]

            parent_daughters = df.loc[(df['parent_id'] == selected_parent['id'])]

            daughter1_index = new_daughter_life_history.index[0]
            daughter2_index = target_bac_index

            new_id = sorted(df['id'].unique())[-1] + 1

            for new_daughter_bac_index in new_daughter_life_history.index:
                df.at[new_daughter_bac_index, 'id'] = new_id
                df.at[new_daughter_bac_index, 'parent_id'] = selected_parent['id']
                df.at[new_daughter_bac_index, 'LifeHistory'] = new_daughter_life_history.shape[0]

            for bac_index in parent_daughters.index:
                df.at[bac_index, 'parent_id'] = new_id

            for parent_bac_indx in parent_bac_before_division.index:
                df.at[parent_bac_indx, 'divideFlag'] = True
                df.at[parent_bac_indx, 'daughters_index'] = [daughter1_index, daughter2_index]
                df.at[parent_bac_indx, 'bad_division_flag'] = False
                df.at[parent_bac_indx, 'bad_daughters_index'] = ''
                df.at[parent_bac_indx, 'unexpected_end'] = False
                df.at[parent_bac_indx, 'division_time'] = target_bacterium['ImageNumber']
                df.at[parent_bac_indx, 'LifeHistory'] = parent_bac_before_division.shape[0]

            for transition_bac_index in target_bacterium_life_history.index:
                df.at[transition_bac_index, 'parent_id'] = selected_parent['id']

        # change parent image number & parent object number of root bacterium
        df.at[target_bac_index, 'TrackObjects_ParentImageNumber_50'] = selected_parent["ImageNumber"]
        df.at[target_bac_index, 'TrackObjects_ParentObjectNumber_50'] = selected_parent["ObjectNumber"]
        # update distance traveled
        df.at[target_bac_index, 'TrackObjects_DistanceTraveled_50'] = \
            distance_from_target_bac[nearest_parent_index]

        df.at[target_bac_index, "transition"] = False

        for idx in related_bacteria_index:
            # change bacteria label to parent label
            df.at[idx, "TrackObjects_Label_50"] = df.iloc[selected_parent_list_index]["TrackObjects_Label_50"]

    return df
