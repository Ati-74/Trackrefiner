import pandas as pd
import numpy as np
from Trackrefiner.core.correction.action.featuresCalculation.calculateOverlap import (track_object_overlaps_to_next_frame,
                                                                                 calculate_frame_existence_links)


def calc_distance_term_for_maintain_continuity_links(continuity_df, center_coord_cols):

    """
    Calculates a distance matrix to maintain continuity links.

    **Distance Calculation**:
        For each source-target pair, the following distances are calculated:
            - The Euclidean distance between the centroids of the source object and the target object.
            - The Euclidean distance between the first endpoints of the source and target objects
              (located on the same side).
            - The Euclidean distance between the second endpoints of the source and target objects
              (located on the opposite side).

    For each source-target pair, the smallest of these three distances is selected as the min_distance.

    :param pandas.DataFrame continuity_df:
        A DataFrame containing information about objects in consecutive time steps. The DataFrame must
        include centroid and endpoint coordinates for calculating distances between source and target objects.
    :param dict center_coord_cols:
        A dictionary specifying the column names for the x and y coordinates of object centroids
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).

    :returns:
        pandas.DataFrame: A pivoted distance matrix where:
            - Rows correspond to source objects (`index_1`).
            - Columns correspond to target objects (`index_2`).
            - Values represent the minimum distance between a source and target.
    """

    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
    continuity_df['center_distance'] = \
        np.linalg.norm(
            continuity_df[[f"{center_coord_cols['x']}_1", f"{center_coord_cols['y']}_1"]].values -
            continuity_df[[f"{center_coord_cols['x']}_2", f"{center_coord_cols['y']}_2"]].values,
            axis=1)

    continuity_df['endpoint1_1_distance'] = \
        np.linalg.norm(continuity_df[['Endpoint1_X_1', 'Endpoint1_Y_1']].values -
                       continuity_df[['Endpoint1_X_2', 'Endpoint1_Y_2']].values, axis=1)

    continuity_df['endpoint2_2_distance'] = \
        np.linalg.norm(continuity_df[['Endpoint2_X_1', 'Endpoint2_Y_1']].values -
                       continuity_df[['Endpoint2_X_2', 'Endpoint2_Y_2']].values, axis=1)

    continuity_df['min_distance'] = \
        continuity_df[['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance']].min(axis=1)

    # Pivot this DataFrame to get the desired structure
    continuity_df_distance_df = \
        continuity_df[['index_1', 'index_2', 'min_distance']].pivot(index='index_1', columns='index_2',
                                                                    values='min_distance')
    continuity_df_distance_df.columns.name = None
    continuity_df_distance_df.index.name = None

    return continuity_df_distance_df


def calc_distance_term_for_maintain_division(division_df, center_coord_cols):

    """
    Calculates a distance matrix to maintain relationship between parent and daughter objects in a division event.

    **Distance Calculation**:
        For each parent-daughter pair, the following distances are calculated:
            - The Euclidean distance between the centroids of the parent and daughter objects (`center_distance`).
            - The Euclidean distance between the first endpoints of the parent and daughter objects
              (`endpoint1_1_distance`).
            - The Euclidean distance between the second endpoints of the parent and daughter objects
              (`endpoint2_2_distance`).
            - The Euclidean distance between the first endpoint of the parent and the second endpoint of the daughter
              (`endpoint12_distance`).
            - The Euclidean distance between the second endpoint of the parent and the first endpoint of the daughter
              (`endpoint21_distance`).
            - The Euclidean distance between the centroid of the parent and the first endpoint of the daughter
              (`center_endpoint1_distance`).
            - The Euclidean distance between the centroid of the parent and the second endpoint of the daughter
              (`center_endpoint2_distance`).

    For each parent-daughter pair, the smallest of these distances is selected as the `min_distance`.

    :param pandas.DataFrame division_df:
        A DataFrame containing information about parent-daughter relationships in division events. The DataFrame
        must include centroid and endpoint coordinates for both parent and daughter objects.
    :param dict center_coord_cols:
        A dictionary specifying the column names for the x and y coordinates of object centroids
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).

    :returns:
        pandas.DataFrame: A pivoted distance matrix where:
            - Rows correspond to parent objects (`index_parent`).
            - Columns correspond to daughter objects (`index_daughter`).
            - Values represent the minimum distance between a parent and daughter.
    """

    # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
    division_df['center_distance'] = \
        np.linalg.norm(division_df[[f"{center_coord_cols['x']}_parent", f"{center_coord_cols['y']}_parent"]].values -
                       division_df[[f"{center_coord_cols['x']}_daughter",
                                    f"{center_coord_cols['y']}_daughter"]].values, axis=1)

    division_df['endpoint1_1_distance'] = \
        np.linalg.norm(division_df[['Endpoint1_X_parent', 'Endpoint1_Y_parent']].values -
                       division_df[['Endpoint1_X_daughter', 'Endpoint1_Y_daughter']].values, axis=1)

    division_df['endpoint2_2_distance'] = \
        np.linalg.norm(division_df[['Endpoint2_X_parent', 'Endpoint2_Y_parent']].values -
                       division_df[['Endpoint2_X_daughter', 'Endpoint2_Y_daughter']].values, axis=1)

    # relation: mother & daughter
    division_df['endpoint12_distance'] = \
        np.linalg.norm(division_df[['Endpoint1_X_parent', 'Endpoint1_Y_parent']].values -
                       division_df[['Endpoint2_X_daughter', 'Endpoint2_Y_daughter']].values, axis=1)

    division_df['endpoint21_distance'] = \
        np.linalg.norm(division_df[['Endpoint2_X_parent', 'Endpoint2_Y_parent']].values -
                       division_df[['Endpoint1_X_daughter', 'Endpoint1_Y_daughter']].values, axis=1)

    division_df['center_endpoint1_distance'] = \
        np.linalg.norm(division_df[[f"{center_coord_cols['x']}_parent",
                                    f"{center_coord_cols['y']}_parent"]].values -
                       division_df[['Endpoint1_X_daughter', 'Endpoint1_Y_daughter']].values, axis=1)

    division_df['center_endpoint2_distance'] = \
        np.linalg.norm(division_df[[f"{center_coord_cols['x']}_parent",
                                    f"{center_coord_cols['y']}_parent"]].values -
                       division_df[['Endpoint2_X_daughter', 'Endpoint2_Y_daughter']].values, axis=1)

    division_df['min_distance'] = (
        division_df)[['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance', 'endpoint12_distance',
                      'endpoint21_distance', 'center_endpoint1_distance', 'center_endpoint2_distance']].min(axis=1)

    # Pivot this DataFrame to get the desired structure
    division_distance_df = \
        division_df[['index_parent', 'index_daughter', 'min_distance']].pivot(index='index_parent',
                                                                              columns='index_daughter',
                                                                              values='min_distance')
    division_distance_df.columns.name = None
    division_distance_df.index.name = None

    return division_distance_df


def calc_distance_and_overlap_matrices(source_time_step_df, sel_source_bacteria, bacteria_in_target_time_step,
                                       sel_target_bacteria, center_coord_cols, color_array, daughter_flag=False,
                                       maintain=False, coordinate_array=None):

    """
    Constructs initial distance and overlap matrices for tracking bacteria across consecutive time steps.

    **Distance Calculation**:
        - When `maintain=True`:
            - Calculates distances to maintain both division and continuity links:
                - Division links: Evaluates spatial relationships between parent and daughter bacteria.
                - Continuity links: Tracks life history continuity of the same bacterium across time steps.
        - When `maintain=False`:
            - Computes overlaps and distances between bacteria in the source and target time steps:
                - Centroid distances (`center_distance`).
                - Endpoint distances:
                    - Between first endpoints (`endpoint1_1_distance`).
                    - Between second endpoints (`endpoint2_2_distance`).
                - Additional distances for daughter tracking:
                    - Between mismatched endpoints (`endpoint1_2_distance` and `endpoint2_1_distance`).
                    - Between centroids and endpoints (`center_endpoint1_distance` and `center_endpoint2_distance`).
            - Determines the minimum distance for each source-target pair and pivots the results into a distance matrix.

    :param pandas.DataFrame source_time_step_df:
        DataFrame containing tracking data for bacteria in the source time step.
    :param pandas.DataFrame sel_source_bacteria:
        Subset of source bacteria selected for analysis.
    :param pandas.DataFrame bacteria_in_target_time_step:
        DataFrame containing tracking data for bacteria in the target objects time step.
    :param pandas.DataFrame sel_target_bacteria:
        Subset of target bacteria selected for analysis.
    :param dict center_coord_cols:
        Dictionary specifying column names for bacterial centroid coordinates
        (e.g., `{"x": "Center_X", "y": "Center_Y"}`).
    :param np.ndarray color_array:
        Array representing the colors of objects in the tracking data. This is used for mapping
        bacteria from the dataframe to the coordinate array for spatial analysis.
    :param bool daughter_flag:
        Whether to include additional distances related to parent-daughter relationships.
    :param bool maintain:
        If `True`, computes distance terms for maintaining division and continuity links.
    :param csr_matrix coordinate_array:
        Array of spatial coordinates used for evaluating candidate links.

    :returns:
        tuple: (overlap_df, distance_df)
            - `overlap_df`: Overlap matrix indicating the existence of relationships between bacteria.
            - `distance_df`: Distance matrix with minimum distances for source-target pairs.
    """

    if sel_target_bacteria.shape[0] > 0 and sel_source_bacteria.shape[0] > 0:

        if maintain:

            division_df = \
                sel_source_bacteria.merge(sel_target_bacteria, left_on='id', right_on='parent_id', how='inner',
                                          suffixes=('_parent', '_daughter'))
            continuity_df = sel_source_bacteria.merge(sel_target_bacteria, on='id', how='inner', suffixes=('_1', '_2'))

            division_distance_df = calc_distance_term_for_maintain_division(division_df, center_coord_cols)
            continuity_distance_df = calc_distance_term_for_maintain_continuity_links(continuity_df, center_coord_cols)

            distance_df = pd.concat([division_distance_df, continuity_distance_df], axis=0)
            overlap_df = calculate_frame_existence_links(division_df, continuity_df, coordinate_array)

            distance_df = distance_df.fillna(999)
            overlap_df = overlap_df.fillna(0)

        else:

            overlap_df, product_df = \
                track_object_overlaps_to_next_frame(source_time_step_df, sel_source_bacteria,
                                                    bacteria_in_target_time_step, sel_target_bacteria,
                                                    center_coord_cols,
                                                    color_array=color_array,
                                                    daughter_flag=daughter_flag,
                                                    coordinate_array=coordinate_array)

            # create distance matrix (row: parent bacterium, column: candidate daughters in next time step)
            dis_cols = ['center_distance', 'endpoint1_1_distance', 'endpoint2_2_distance']

            # _current', '_next
            product_df['center_distance'] = \
                np.linalg.norm(product_df[[f"{center_coord_cols['x']}_current",
                                           f"{center_coord_cols['y']}_current"]].values -
                               product_df[[f"{center_coord_cols['x']}_next",
                                           f"{center_coord_cols['y']}_next"]].values, axis=1)

            product_df['endpoint1_1_distance'] = \
                np.linalg.norm(product_df[['Endpoint1_X_current', 'Endpoint1_Y_current']].values -
                               product_df[['Endpoint1_X_next', 'Endpoint1_Y_next']].values, axis=1)

            product_df['endpoint2_2_distance'] = \
                np.linalg.norm(product_df[['Endpoint2_X_current', 'Endpoint2_Y_current']].values -
                               product_df[['Endpoint2_X_next', 'Endpoint2_Y_next']].values, axis=1)

            if daughter_flag:
                dis_cols.extend(['endpoint1_2_distance', 'endpoint2_1_distance', 'center_endpoint1_distance',
                                 'center_endpoint2_distance'])

                product_df['endpoint1_2_distance'] = \
                    np.linalg.norm(product_df[['Endpoint1_X_current', 'Endpoint1_Y_current']].values -
                                   product_df[['Endpoint2_X_next', 'Endpoint2_Y_next']].values, axis=1)

                product_df['endpoint2_1_distance'] = \
                    np.linalg.norm(product_df[['Endpoint2_X_current', 'Endpoint2_Y_current']].values -
                                   product_df[['Endpoint1_X_next', 'Endpoint1_Y_next']].values, axis=1)

                product_df['center_endpoint1_distance'] = \
                    np.linalg.norm(product_df[[f"{center_coord_cols['x']}_current",
                                               f"{center_coord_cols['y']}_current"]].values -
                                   product_df[['Endpoint1_X_next', 'Endpoint1_Y_next']].values, axis=1)

                product_df['center_endpoint2_distance'] = \
                    np.linalg.norm(product_df[[f"{center_coord_cols['x']}_current",
                                               f"{center_coord_cols['y']}_current"]].values -
                                   product_df[['Endpoint2_X_next', 'Endpoint2_Y_next']].values, axis=1)

            product_df['min_distance'] = product_df[dis_cols].min(axis=1)

            # Pivot this DataFrame to get the desired structure
            distance_df = \
                product_df[['index_current', 'index_next', 'min_distance']].pivot(
                    index='index_current', columns='index_next', values='min_distance')
            distance_df.columns.name = None
            distance_df.index.name = None

    else:
        if sel_target_bacteria.shape[0] > 0:
            overlap_df = pd.DataFrame(columns=[sel_target_bacteria.index], index=[0], data=999)
            distance_df = pd.DataFrame(columns=[sel_target_bacteria.index], index=[0], data=999)
        elif sel_source_bacteria.shape[0] > 0:
            overlap_df = pd.DataFrame(columns=[0], index=sel_source_bacteria.index, data=999)
            distance_df = pd.DataFrame(columns=[0], index=sel_source_bacteria.index, data=999)

    overlap_df = overlap_df.apply(pd.to_numeric)
    distance_df = distance_df.apply(pd.to_numeric)

    return overlap_df, distance_df
