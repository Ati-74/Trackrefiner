import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix


def incorrect_df(data_frame):
    data_frame = data_frame.loc[(data_frame["drop"] == True) |
                                (data_frame["TrackObjects_Label_50"].isnull())].reset_index(drop=True)
    return data_frame


def find_related_bacteria(df, root_bac_index_in_df, next_parent_image_number, next_parent_object_number, time_step,
                          timestep_between_parent_daughter, bac_index_list=None):
    """
    goal:  From the bacteria dataframe, find the row index of bacteria related to a specific bacterium (bacterium
    without a parent) input: root_bac_index_in_df   number   The row index of the corresponding bacterium output:
    bac_index_list   list   row index of related bacteria

        """

    if bac_index_list is None:  # create a new result if no intermediate was given
        bac_index_list = [root_bac_index_in_df]

    # related bacteria in next timestep
    next_timestep = time_step + 1
    if next_timestep <= df.iloc[-1]['ImageNumber']:
        navigate_timestep_between_parent_daughter = 0
        family_tree_check = False
        while (navigate_timestep_between_parent_daughter <= timestep_between_parent_daughter) and \
                family_tree_check is False:
            bac_in_next_timestep = df.loc[(df["ImageNumber"] == next_timestep) &
                                          (df["TrackObjects_ParentImageNumber_50"] == next_parent_image_number) &
                                          (df["TrackObjects_ParentObjectNumber_50"] == next_parent_object_number)]

            if bac_in_next_timestep.shape[0] > 0:
                family_tree_check = True
                bac_in_next_timestep_index = bac_in_next_timestep.index.tolist()
                bac_in_next_timestep = bac_in_next_timestep.reset_index(drop=True)
                bac_index_list.extend(bac_in_next_timestep_index)

                # next time step
                time_step = next_timestep
                for bac_index in range(len(bac_in_next_timestep_index)):
                    next_bac_index_in_df = bac_in_next_timestep_index[bac_index]
                    next_bacteria_next_parent_image_number = next_timestep
                    next_bacteria_next_parent_object_number = bac_in_next_timestep.iloc[bac_index]["ObjectNumber"]
                    find_related_bacteria(df, next_bac_index_in_df, next_bacteria_next_parent_image_number,
                                          next_bacteria_next_parent_object_number,
                                          time_step, timestep_between_parent_daughter, bac_index_list)

                for gap_number in range(timestep_between_parent_daughter):
                    time_step += 1
                    find_related_bacteria(df, root_bac_index_in_df, next_parent_image_number, next_parent_object_number,
                                          time_step, timestep_between_parent_daughter, bac_index_list)
            else:
                navigate_timestep_between_parent_daughter += 1
                next_timestep += 1

    return bac_index_list


def remove_rows(df):
    """
        goal: remove bacteria that have no parents
        input: df    dataframe   bacteria dataframe
        output: df   dataframe   modified dataframe

    """

    df = df.loc[df["drop"] == False].reset_index(drop=True)
    return df


def lineage_bacteria_after_this_time_step(data_frame, bacteria):
    """
        goal: find family tree of corresponding bacterium
        input: dataframe   dataframe   bacteria information dataframe
        input: Bacteria    dataframe   Associated dataframe row with bacterium
        output: dataFrameOfLineage   dataframe   dataframe of related bacteria (family tree) to corresponding bacterium
    """

    data_frame_of_lineage = data_frame.loc[(data_frame["TrackObjects_Label_50"] == bacteria["TrackObjects_Label_50"]) &
                                           (data_frame["ImageNumber"] >= bacteria["ImageNumber"])]
    return data_frame_of_lineage


def find_incorrect_daughter(data_frame_of_lineage, bacteria, number_of_gap):
    """
        goal: find index row of incorrect daughter in dataframe
        @param data_frame_of_lineage   dataframe   dataframe of related bacteria (family tree) to corresponding bacterium
        @param  bacteria    dataframe   Associated dataframe row with bacterium
        @param number_of_gap int  maximum number of gaps between parent and next bacterium in its family tree
        output incorrect_daughter_row_index   number row index of incorrect daughter bacterium in bacteria dataframe
    """

    root_image_number = bacteria["ImageNumber"]
    root_object_number = bacteria["ObjectNumber"]
    previous_bacteria_image_number_in_this_family_tree = [root_image_number]
    previous_bacteria_object_number_in_this_family_tree = [root_object_number]

    next_time_step = root_image_number + 1

    # maximum number of time steps between parent bacterium and same bacterium
    maximum_time_steps_between_parent_same_bacterium = number_of_gap + 1
    time_step_distance_between_parent_daughter = 0

    division_occ = False
    last_time_step = data_frame_of_lineage["ImageNumber"].iloc[-1]

    incorrect_daughter_row_index = []

    while (division_occ is False) and (next_time_step <= last_time_step) and \
            (time_step_distance_between_parent_daughter <= maximum_time_steps_between_parent_same_bacterium):

        bacteria_in_next_timestep = data_frame_of_lineage.loc[(data_frame_of_lineage["ImageNumber"] == next_time_step)]

        parents_detail_df = pd.DataFrame(list(zip(previous_bacteria_image_number_in_this_family_tree,
                                                  previous_bacteria_object_number_in_this_family_tree)),
                                         columns=["TrackObjects_ParentImageNumber_50",
                                                  "TrackObjects_ParentObjectNumber_50"])

        relative_bacteria_in_next_time_step = pd.merge(bacteria_in_next_timestep, parents_detail_df, how='inner',
                                                       on=["TrackObjects_ParentImageNumber_50",
                                                           "TrackObjects_ParentObjectNumber_50"])

        number_of_relative_bacteria = relative_bacteria_in_next_time_step.shape[0]

        if number_of_relative_bacteria == 2:  # the division has been found
            division_occ = True

        elif number_of_relative_bacteria == 1:
            previous_bacteria_image_number_in_this_family_tree.append(bacteria_in_next_timestep.iloc[0]["ImageNumber"])
            previous_bacteria_object_number_in_this_family_tree.append(
                bacteria_in_next_timestep.iloc[0]["ObjectNumber"])
            next_time_step += 1
            time_step_distance_between_parent_daughter = 1

        elif number_of_relative_bacteria == 0:
            #  maybe the family tree of the corresponding bacterium has been continued (because of the gap)
            next_time_step += 1
            time_step_distance_between_parent_daughter += 1
            pass
        else:
            daughter_bacteria_traveled_distance = relative_bacteria_in_next_time_step[
                "AreaShape_MajorAxisLength"].values
            sorted_daughters_distance_traveled = sorted(daughter_bacteria_traveled_distance, reverse=True)
            for daughter_distance_traveled in \
                    sorted_daughters_distance_traveled[0:len(sorted_daughters_distance_traveled) - 2]:
                max_distance_traveled = daughter_distance_traveled
                incorrect_daughter = np.where(daughter_bacteria_traveled_distance == max_distance_traveled)
                bad_daughter = data_frame_of_lineage.loc[(data_frame_of_lineage['ImageNumber'] ==
                                                          relative_bacteria_in_next_time_step.iloc[incorrect_daughter]
                                                          ['ImageNumber'].values[0]) &
                                                         (data_frame_of_lineage['ObjectNumber'] ==
                                                          relative_bacteria_in_next_time_step.iloc[incorrect_daughter]
                                                          ['ObjectNumber'].values[0])]
                incorrect_daughter_row_index.append(bad_daughter.index.values[0])
            division_occ = True

    return incorrect_daughter_row_index


def bacteria_in_specific_time_step(df, t):
    """
    goal: find bacteria in specific time step
    @param df dataframe bacteria information dataframe
    @param t int timestep
    """
    correspond_bacteria = df.loc[(df["ImageNumber"] == t) & (df["drop"] == False)]
    correspond_bacteria_index = correspond_bacteria.index.values.tolist()
    correspond_bacteria = correspond_bacteria.reset_index(drop=True)

    return correspond_bacteria, correspond_bacteria_index


def adjacency_matrix(next_timestep_sudden_bac, next_timestep_sudden_bac_index, another_timestep_bac,
                     another_timestep_bac_index):
    # create distance matrix (rows: next time step sudden bacteria, columns: another time step bacteria)
    try:
        distance_df = pd.DataFrame(
            distance_matrix(next_timestep_sudden_bac[["Location_Center_X", "Location_Center_Y"]].values,
                            another_timestep_bac[["Location_Center_X", "Location_Center_Y"]].values),
            index=next_timestep_sudden_bac_index, columns=another_timestep_bac_index)
    except:
        distance_df = pd.DataFrame(
            distance_matrix(next_timestep_sudden_bac[["AreaShape_Center_X", "AreaShape_Center_Y"]].values,
                            another_timestep_bac[["AreaShape_Center_X", "AreaShape_Center_Y"]].values),
            index=next_timestep_sudden_bac_index, columns=another_timestep_bac_index)

    return distance_df


def find_parent_in_consider_time_step(df, sorted_distance_consider_time_step, next_time_steps_bac_list):
    """
    goal: finding parent bacterium in previous time steps
    @param df dataframe bacteria information in each time steps
    @param sorted_distance_consider_time_step list it shows the distance of bacteria from
    the considered time step to transition bacterium
    @param next_time_steps_bac_list list list of dataframe of bacteria in variant time steps before
    after parent bacteria candidate time step
    """
    was_parent_found = False
    # first index of tuple list
    element_index_in_sorted_distance = 0
    # distance of parent bacterium to transition bacterium
    parent_distance_from_transition_bacterium = None
    # maximum distance between candidate parent and transition bacterium
    distance_threshold = 25
    same_bacterium_found = False
    division_found = False

    while was_parent_found is False and element_index_in_sorted_distance < len(sorted_distance_consider_time_step):
        # row number of the nearest bacterium (candidate parent) to transition bacterium
        nearest_bac_index_in_consider_time_step = sorted_distance_consider_time_step[element_index_in_sorted_distance][
            1]
        # find number of related bacteria to this bacterium in current time step
        parent_image_number = df.iloc[nearest_bac_index_in_consider_time_step]['ImageNumber']
        parent_object_number = df.iloc[nearest_bac_index_in_consider_time_step]['ObjectNumber']
        same_bacterium_found = False
        division_found = False

        for next_time_steps_bac in next_time_steps_bac_list:
            related_bac_in_next_timestep = next_time_steps_bac.loc[
                (next_time_steps_bac["TrackObjects_ParentImageNumber_50"] == parent_image_number) &
                (next_time_steps_bac["TrackObjects_ParentObjectNumber_50"] == parent_object_number)].reset_index(
                drop=True)

            if related_bac_in_next_timestep.shape[0] < 2:
                if related_bac_in_next_timestep.shape[0] == 1:
                    candidate_parent_length = df.iloc[nearest_bac_index_in_consider_time_step][
                        'AreaShape_MajorAxisLength']
                    next_bac_in_family_tree_length = related_bac_in_next_timestep.iloc[0]['AreaShape_MajorAxisLength']
                    proportion_of_length = next_bac_in_family_tree_length / candidate_parent_length
                    # check travel distance
                    nearest_candidate_parent_distance = \
                        sorted_distance_consider_time_step[element_index_in_sorted_distance][0]
                    if proportion_of_length < 0.85 or nearest_candidate_parent_distance <= distance_threshold:
                        # it means that division has been occurrence but one daughter doesn't detect bt Cellprofiler
                        pass
                    else:
                        # it means that the life history of candidate parent bacterium has been continued
                        same_bacterium_found = True
                        break
                else:
                    pass
            else:
                division_found = True
                break
        if division_found is True or same_bacterium_found is True:
            # check next candidate bacterium
            element_index_in_sorted_distance += 1
        else:
            was_parent_found = True
            parent_distance_from_transition_bacterium = \
                sorted_distance_consider_time_step[element_index_in_sorted_distance][0]

    return was_parent_found, nearest_bac_index_in_consider_time_step, parent_distance_from_transition_bacterium


def candidate_parents(df, sudden_bac_index, distance_list, consider_time_step_bac_list,
                      consider_timestep_bac_index_list):
    was_parent_found_in_consider_time_step_list = []
    candidate_parent_index_in_consider_time_step_list = []
    consider_parent_distance_from_transition_bacterium_list = []

    # was the parent found from previous time steps?
    for indx, consider_distance_df in enumerate(distance_list):
        correspond_distance_df_row = consider_distance_df.iloc[sudden_bac_index].values.tolist()
        # [(distance, bacterium index),(),(), ...]
        sorted_distance_consider_time_step = [(distance, bac_index) for distance, bac_index in
                                              sorted(zip(correspond_distance_df_row,
                                                         consider_timestep_bac_index_list[indx]))]

        was_parent_found_in_consider_time_step, candidate_parent_index_in_consider_time_step, \
        consider_parent_distance_from_transition_bacterium = \
            find_parent_in_consider_time_step(df, sorted_distance_consider_time_step,
                                              consider_time_step_bac_list[indx + 1:])

        if was_parent_found_in_consider_time_step is not False:
            was_parent_found_in_consider_time_step_list.append(was_parent_found_in_consider_time_step)
            candidate_parent_index_in_consider_time_step_list.append(
                candidate_parent_index_in_consider_time_step)
            consider_parent_distance_from_transition_bacterium_list.append(
                consider_parent_distance_from_transition_bacterium)

    return was_parent_found_in_consider_time_step_list, candidate_parent_index_in_consider_time_step_list, \
           consider_parent_distance_from_transition_bacterium_list


def find_appropriate_parent_for_incorrect_daughter(df, bacterium, number_of_gap, next_timestep_bac):
    bacteria_time_step = bacterium['ImageNumber'].values.tolist()[0]

    # I define this list to stora adjacency matrices
    distance_list = []
    # list of bacteria information dataframe in previous time steps, and curren time step
    consider_time_step_bac_list = []
    # list of bacteria index list in previous time steps
    consider_timestep_bac_index_list = []

    for i in range(max(bacteria_time_step - number_of_gap - 1, 1), bacteria_time_step):
        # consider time step
        consider_time_step = i
        # filter consider time step bacteria information
        consider_time_step_bac, consider_timestep_bac_index = bacteria_in_specific_time_step(df,
                                                                                             consider_time_step)
        consider_time_step_bac_list.append(consider_time_step_bac)
        consider_timestep_bac_index_list.append(consider_timestep_bac_index)
        # create distance matrix (rows: next time step sudden bacteria, columns: consider time step bacteria)
        consider_distance_df = adjacency_matrix(bacterium, bacterium.index.values.tolist(),
                                                consider_time_step_bac, consider_timestep_bac_index)
        distance_list.append(consider_distance_df)

    # append next time step bacteria information
    consider_time_step_bac_list.append(next_timestep_bac)

    # find the parent of sudden bacteria in next time step
    was_parent_found_in_consider_time_step_list, candidate_parent_index_in_consider_time_step_list, \
    consider_parent_distance_from_transition_bacterium_list = candidate_parents(df, 0,
                                                                                distance_list,
                                                                                consider_time_step_bac_list,
                                                                                consider_timestep_bac_index_list)

    return was_parent_found_in_consider_time_step_list, candidate_parent_index_in_consider_time_step_list, \
           consider_parent_distance_from_transition_bacterium_list


def correction_transition(df, number_of_gap):
    """
        goal: For bacteria without parent, assign labels, ParentImageNumber, and ParentObjectNumber
        @param df    dataframe   bacteria dataframe
        @param number_of_gap int I define a gap number to find parent in other previous time steps
        output: df   dataframe   modified dataframe (without any transitions)

    """

    # add new column to dataframe
    # During processing, I need it to find bacteria that have no parents (and remove them at the end)
    df["drop"] = False

    # now, I try to resolve transition problem
    max_timestep_between_parent_daughter = number_of_gap + 1

    # list of unique time steps
    time_steps = list(set(df['ImageNumber'].values))
    # remove last time step
    time_steps.pop()

    for t in time_steps:

        # filter next time step bacteria information
        next_time_step = t + 1
        next_timestep_bac, next_timestep_bac_index = bacteria_in_specific_time_step(df, next_time_step)
        # print(next_timestep_bac)
        # filter next time step sudden bacteria information (bacteria without parent)
        next_timestep_sudden_bac = next_timestep_bac.loc[next_timestep_bac["TrackObjects_ParentImageNumber_50"] == 0]
        next_timestep_sudden_bac_index = [next_timestep_bac_index[i] for i in
                                          next_timestep_sudden_bac.index.values.tolist()]
        next_timestep_sudden_bac = next_timestep_sudden_bac.reset_index(drop=True)

        if next_timestep_sudden_bac.shape[0] > 0:

            # I define this list to stora adjacency matrices
            distance_list = []
            # list of bacteria information dataframe in previous time steps, and curren time step
            consider_time_step_bac_list = []
            # list of bacteria index list in previous time steps
            consider_timestep_bac_index_list = []

            for i in range(max(t - number_of_gap, 1), t + 1):
                # consider time step
                consider_time_step = i
                # filter consider time step bacteria information
                consider_time_step_bac, consider_timestep_bac_index = bacteria_in_specific_time_step(df,
                                                                                                     consider_time_step)
                consider_time_step_bac_list.append(consider_time_step_bac)
                consider_timestep_bac_index_list.append(consider_timestep_bac_index)
                # create distance matrix (rows: next time step sudden bacteria, columns: consider time step bacteria)
                consider_distance_df = adjacency_matrix(next_timestep_sudden_bac, next_timestep_sudden_bac_index,
                                                        consider_time_step_bac, consider_timestep_bac_index)
                distance_list.append(consider_distance_df)

            # append next time step bacteria information
            consider_time_step_bac_list.append(next_timestep_bac)

            # find the parent of sudden bacteria in next time step
            for sudden_bac_index in range(len(next_timestep_sudden_bac_index)):

                was_parent_found_in_consider_time_step_list, candidate_parent_index_in_consider_time_step_list, \
                consider_parent_distance_from_transition_bacterium_list = candidate_parents(df, sudden_bac_index,
                                                                                            distance_list,
                                                                                            consider_time_step_bac_list,
                                                                                            consider_timestep_bac_index_list)
                # find related bacteria to this bacterium
                root_bac_index_in_df = next_timestep_sudden_bac_index[sudden_bac_index]
                next_parent_image_number = t + 1
                next_parent_object_number = df.iloc[root_bac_index_in_df]["ObjectNumber"]
                time_step = t + 1
                related_bacteria = find_related_bacteria(df, root_bac_index_in_df, next_parent_image_number,
                                                         next_parent_object_number, time_step,
                                                         max_timestep_between_parent_daughter)

                if not was_parent_found_in_consider_time_step_list:
                    # it means that no parent has been found for this bacterium
                    # remove this bacterium and related bacteria
                    for idx in related_bacteria:
                        # change bacteria label to parent label
                        df.at[idx, "drop"] = True
                else:
                    # sort distance and find the nearest parent
                    nearest_parent_index = np.argmin(consider_parent_distance_from_transition_bacterium_list)
                    selected_parent_list_index = candidate_parent_index_in_consider_time_step_list[nearest_parent_index]

                    # change parent image number & parent object number of root bacterium
                    df.at[root_bac_index_in_df, 'TrackObjects_ParentImageNumber_50'] = \
                        df.iloc[selected_parent_list_index]["ImageNumber"]
                    df.at[root_bac_index_in_df, 'TrackObjects_ParentObjectNumber_50'] = \
                        df.iloc[selected_parent_list_index]["ObjectNumber"]
                    # update distance traveled
                    df.at[root_bac_index_in_df, 'TrackObjects_DistanceTraveled_50'] = \
                        min(consider_parent_distance_from_transition_bacterium_list)

                    for list_index, idx in enumerate(related_bacteria):
                        # change bacteria label to parent label
                        df.at[idx, "TrackObjects_Label_50"] = df.iloc[selected_parent_list_index][
                            "TrackObjects_Label_50"]

    # remove bacteria that have no parents
    incorrect_bacteria_after_transition = incorrect_df(df)
    correct_bacteria_after_transition = remove_rows(df)
    # remove nan labels
    correct_bacteria_after_transition = correct_bacteria_after_transition.loc[df["TrackObjects_Label_50"].notnull()].reset_index(drop=True)
    # remove the smallest daughter, and related bacteria
    correct_bacteria, incorrect_bacteria_after_tracking = correction_tracking(correct_bacteria_after_transition,
                                                                              number_of_gap)
    incorrect_bacteria = pd.concat([incorrect_bacteria_after_transition, incorrect_bacteria_after_tracking],
                                   ignore_index=True, sort=True)
    return correct_bacteria, incorrect_bacteria


def correction_tracking(df, number_of_gap):
    timestep_between_parent_daughter = number_of_gap + 1
    """
            goal:  Find bacteria with three daughters, remove the smallest daughter, and related bacteria
            input: df   dataframe   Bacteria information dataframe without transitions
            output df   dataframe   modified dataframe (without incorrect daughters, and related bacteria to them)

        """
    for index, row in df.iterrows():
        if not df.iloc[index]["drop"]:

            data_frame_of_lineage = lineage_bacteria_after_this_time_step(df, row)

            incorrect_daughter_row_index = find_incorrect_daughter(data_frame_of_lineage, row, number_of_gap)
            if incorrect_daughter_row_index:
                for incorrect_bac_index in incorrect_daughter_row_index:
                    # it means that: incorrect daughter has been found
                    # now, we should find related bacteria to the incorrect daughter
                    root_bac_index_in_df = incorrect_bac_index
                    next_parent_image_number = df.iloc[incorrect_bac_index]["ImageNumber"]
                    next_parent_object_number = df.iloc[incorrect_bac_index]["ObjectNumber"]
                    time_step = next_parent_image_number
                    related_bacteria = find_related_bacteria(df, root_bac_index_in_df, next_parent_image_number,
                                                             next_parent_object_number, time_step,
                                                             timestep_between_parent_daughter)

                    # assign new parent to incorrect daughter
                    incorrect_daughter = df.loc[df.index == incorrect_bac_index]
                    same_daughter_time_step_bac = df.loc[
                        df["ImageNumber"] == df.iloc[incorrect_bac_index]["ImageNumber"]]
                    was_parent_found_in_consider_time_step_list, candidate_parent_index_in_consider_time_step_list, \
                    consider_parent_distance_from_transition_bacterium_list = \
                        find_appropriate_parent_for_incorrect_daughter(df, incorrect_daughter, number_of_gap,
                                                                       same_daughter_time_step_bac)

                    if not was_parent_found_in_consider_time_step_list:
                        # remove this bacterium and related bacteria
                        for idx in related_bacteria:
                            # change bacteria label to parent label
                            df.at[idx, "drop"] = True
                    else:
                        # sort distance and find the nearest parent
                        nearest_parent_index = np.argmin(consider_parent_distance_from_transition_bacterium_list)
                        selected_parent_list_index = candidate_parent_index_in_consider_time_step_list[
                            nearest_parent_index]

                        # change parent image number & parent object number of root bacterium
                        df.at[incorrect_bac_index, 'TrackObjects_ParentImageNumber_50'] = \
                            df.iloc[selected_parent_list_index]["ImageNumber"]
                        df.at[incorrect_bac_index, 'TrackObjects_ParentObjectNumber_50'] = \
                            df.iloc[selected_parent_list_index]["ObjectNumber"]
                        # update distance traveled
                        df.at[incorrect_bac_index, 'TrackObjects_DistanceTraveled_50'] = \
                            min(consider_parent_distance_from_transition_bacterium_list)

                        for list_index, idx in enumerate(related_bacteria):
                            # change bacteria label to parent label
                            df.at[idx, "TrackObjects_Label_50"] = df.iloc[selected_parent_list_index][
                                "TrackObjects_Label_50"]

    # remove incorrect daughter bacteria , and related bacteria to them
    incorrect_bacteria_after_tracking = incorrect_df(df)
    correct_bacteria_after_tracking = remove_rows(df)
    # remove 'drop' column
    correct_bacteria_after_tracking.drop('drop', axis=1, inplace=True)
    # correct_bacteria_after_tracking.to_csv('file.csv', index=False)
    print('dataframe has been modified')
    return correct_bacteria_after_tracking, incorrect_bacteria_after_tracking


def finding_incorrect_bacteria(df, number_of_gap):

    correct_bacteria, incorrect_bacteria = correction_transition(df, number_of_gap)

    return correct_bacteria, incorrect_bacteria
