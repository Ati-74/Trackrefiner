import pandas as pd

from CellProfilerAnalysis.strain.correction.action.compareBacteria import adding_new_link_to_unexpected
from CellProfilerAnalysis.strain.correction.action.compareBacteria import optimize_assignment
from CellProfilerAnalysis.strain.correction.action.bacteriaModification import bacteria_modification
import numpy as np


def adding_new_link(df, neighbors_df, unexpected_end_bac_index, unexpected_end_bac_bac, new_link_result_df,
                    optimized_daughter_cost):
    new_link_cost = new_link_result_df.loc[new_link_result_df['without parent index'] == unexpected_end_bac_index]

    if new_link_cost.shape[0] > 0:
        source_target_cost = new_link_cost['Cost'].values.tolist()[0]

        if source_target_cost != 999:
            # dose it have a daughter?
            if optimized_daughter_cost is not None:
                if new_link_cost['without parent index'].values.tolist()[0] in \
                        optimized_daughter_cost['without parent index'].values.tolist():
                    another_daughter = optimized_daughter_cost.loc[optimized_daughter_cost['without parent index'] ==
                                                                   new_link_cost[
                                                                       'without parent index'].values.tolist()[0]]

                    target_bac = df.iloc[
                        new_link_cost['Candida bacteria index in previous time step'].values.tolist()[0]]
                    another_target_bac = df.iloc[
                        another_daughter['Candida bacteria index in previous time step'].values.tolist()[0]]

                    target_bac_life_history = df.loc[df['id'] == target_bac['id']]

                    another_target_bac_life_history = df.loc[df['id'] == another_target_bac['id']]

                    all_bac_in_target_time_step = df.loc[df['ImageNumber'] == target_bac['ImageNumber']]

                    df = bacteria_modification(df, unexpected_end_bac_bac, target_bac_life_history,
                                               all_bac_in_target_time_step, neighbors_df)
                    df = bacteria_modification(df, unexpected_end_bac_bac, another_target_bac_life_history,
                                               all_bac_in_target_time_step, neighbors_df)
                else:
                    target_bac = df.iloc[
                        new_link_cost['Candida bacteria index in previous time step'].values.tolist()[0]]
                    target_bac_life_history = df.loc[df['id'] == target_bac['id']]
                    all_bac_in_target_time_step = df.loc[df['ImageNumber'] == target_bac['ImageNumber']]

                    df = bacteria_modification(df, unexpected_end_bac_bac, target_bac_life_history,
                                               all_bac_in_target_time_step, neighbors_df)

            else:
                target_bac = df.iloc[new_link_cost['Candida bacteria index in previous time step'].values.tolist()[0]]
                target_bac_life_history = df.loc[df['id'] == target_bac['id']]
                all_bac_in_target_time_step = df.loc[df['ImageNumber'] == target_bac['ImageNumber']]

                df = bacteria_modification(df, unexpected_end_bac_bac, target_bac_life_history,
                                           all_bac_in_target_time_step,
                                           neighbors_df)

    return df


def unexpected_end_bacteria(df, neighbors_df, sorted_npy_files_list, min_life_history_of_bacteria, interval_time,
                            logs_df):

    num_incorrect_same_links = None
    prev_bacteria_with_wrong_same_link = None
    n_iterate = 0

    # min life history of bacteria
    min_life_history_of_bacteria_time_step = np.round_(min_life_history_of_bacteria / interval_time)

    while num_incorrect_same_links != 0:
        unexpected_end_bacteria = df.loc[(df['unexpected_end'] == True) & (df['noise_bac'] == False)]

        if n_iterate > 0:
            if prev_bacteria_with_wrong_same_link.values.all() == unexpected_end_bacteria.values.all():
                num_incorrect_same_links = 0
            else:
                num_incorrect_same_links = unexpected_end_bacteria.shape[0]

        prev_bacteria_with_wrong_same_link = unexpected_end_bacteria

        if unexpected_end_bacteria.shape[0] > 0:
            for unexpected_end_bacteria_time_step in unexpected_end_bacteria['ImageNumber'].unique():
                unexpected_end_bac_in_current_time_step_df = \
                    unexpected_end_bacteria.loc[
                        unexpected_end_bacteria['ImageNumber'] == unexpected_end_bacteria_time_step]

                all_bac_in_current_time_step = df.loc[df['ImageNumber'] == unexpected_end_bacteria_time_step]
                all_bac_in_next_time_step = df.loc[df['ImageNumber'] == unexpected_end_bacteria_time_step + 1]

                new_link_result_df, candidate_new_bac_daughter_list_id, cost_df, maintenance_cost_df = \
                    adding_new_link_to_unexpected(df, neighbors_df, sorted_npy_files_list,
                                                  unexpected_end_bac_in_current_time_step_df,
                                                  all_bac_in_current_time_step, all_bac_in_next_time_step,
                                                  min_life_history_of_bacteria_time_step)

                if len(candidate_new_bac_daughter_list_id.keys()) > 0:
                    daughter_ndx = [item for sublist in list(candidate_new_bac_daughter_list_id.values()) for item in
                                    sublist]
                    daughter_cost = cost_df[list(set(daughter_ndx))]

                    for col in daughter_cost.columns:
                        related_bac = [v for v in candidate_new_bac_daughter_list_id.keys() if col in
                                       candidate_new_bac_daughter_list_id[v]]

                        daughter_cost.loc[~ daughter_cost.index.isin(related_bac), :] = 999

                    # Create a mask where all values in a row are 999
                    mask = (daughter_cost == 999).all(axis=1)

                    # Use the mask to select rows where not all values are 999
                    daughter_cost_filtered = daughter_cost.loc[~mask]

                    optimized_daughter_cost = optimize_assignment(daughter_cost_filtered)

                else:
                    optimized_daughter_cost = None

                # try to modify the links
                for unexpected_end_bac_index, unexpected_end_bac in (
                        unexpected_end_bac_in_current_time_step_df.iterrows()):
                    unexpected_bac = df.iloc[unexpected_end_bac_index]

                    df = adding_new_link(df, neighbors_df, unexpected_end_bac_index, unexpected_bac,
                                         new_link_result_df, optimized_daughter_cost)

                    logs_df = pd.concat([logs_df, df.iloc[unexpected_end_bac_index].to_frame().transpose()],
                                        ignore_index=True)

                    # df.to_csv(str(unexpected_end_bac_index) + '.csv')

        n_iterate += 1
    return df, logs_df
