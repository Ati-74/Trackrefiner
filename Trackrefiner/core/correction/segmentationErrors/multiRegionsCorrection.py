import numpy as np
import pandas as pd


def modify_existing_object(df, regions_color, faulty_row, img_array, regions_coordinates, coordinate_array,
                           color_array):
    """
    Modifies attributes of an existing object in the image and updates the relevant data structures.

    :param pandas.DataFrame df:
        Input DataFrame containing object data, including tracking and feature information.
    :param list regions_color:
        list representing the colors assigned to different regions.
    :param dict faulty_row:
        A dictionary containing details about the faulty object, including indices used for updates.
    :param numpy.ndarray img_array:
        Image array where object regions are represented by their respective colors.
    :param list of numpy.ndarray regions_coordinates:
        A list of arrays where element indices are region indices, and
        elements are arrays of coordinates for each region.
    :param csr_matrix coordinate_array:
        Boolean array tracking encoded spatial coordinates for each object.
    :param numpy.ndarray color_array:
        Array storing the color of each object.

    :returns:
        tuple:

        Updated DataFrame (`df`), image array (`img_array`), coordinate array (`coordinate_array`),
        and color array (`color_array`).

    """

    # It means that the Cellprofiler detected both multi-regions and particles
    this_region_color = regions_color[faulty_row['row idx par']]

    # Update the img_array with the new color for this region
    img_array[tuple(zip(*regions_coordinates[faulty_row['row idx par']]))] = this_region_color

    color_array[faulty_row['cp index']] = this_region_color
    coordinates_array = regions_coordinates[faulty_row['row idx par']]
    x, y = coordinates_array[:, 0], coordinates_array[:, 1]
    # Cantor Pairing Function
    encoded_numbers = (x + y) * (x + y + 1) // 2 + y
    coordinate_array[faulty_row['cp index'], encoded_numbers] = True

    return df, img_array, coordinate_array, color_array


def remove_object_from_image(df, faulty_row, img_array, regions_coordinates):
    """
    Removes an object from the image by removing its region in the image array and marking it as noise in the DataFrame.

    :param pandas.DataFrame df:
        Input DataFrame containing object data, including tracking and feature information.
    :param dict faulty_row:
        A dictionary containing details about the faulty object, including indices used for updates.
    :param numpy.ndarray img_array:
        Image array where object regions are represented by their respective colors.
    :param list of numpy.ndarray regions_coordinates:
        A list of arrays where element indices are region indices, and
        elements are arrays of coordinates for each region.

    :returns:
        tuple: Updated DataFrame (`df`) and image array (`img_array`).

    """

    # Update the img_array with the new color for this region
    img_array[tuple(zip(*regions_coordinates[faulty_row['row idx par']]))] = (0, 0, 0)
    df.at[faulty_row['cp index'], 'noise_bac'] = True

    return df, img_array


def update_object_records(df, faulty_row, img_array, regions_coordinates, regions_color, regions_features,
                          all_center_coord_cols, parent_image_number_col, parent_object_number_col,
                          coordinate_array, color_array):
    """
    Updates the records and attributes of a multi-region object after determining the best-fitting region.

    :param pandas.DataFrame df:
        Input DataFrame containing object data, including tracking and feature information.
    :param dict faulty_row:
        A dictionary containing details about the faulty object, including indices used for updates.
    :param numpy.ndarray img_array:
        Image array where object regions are represented by their respective colors.
    :param list of numpy.ndarray regions_coordinates:
        A list of Arrays where element indices are region indices, and
        elements are arrays of coordinates for each region.
    :param list regions_color:
        list representing the colors assigned to different regions.
    :param list of dict regions_features:
        A list of dictionaries where keys are region indices, and values contain feature data for each region
        (e.g., orientation, major/minor axis lengths).
    :param dict all_center_coord_cols:
        A dictionary mapping coordinate types (`x`, `y`) to their respective column names in the DataFrame.
    :param str parent_image_number_col:
        Column name in the DataFrame representing the parent image number.
    :param str parent_object_number_col:
        Column name in the DataFrame representing the parent object number.
    :param csr_matrix coordinate_array:
        Boolean array tracking encoded spatial coordinates for each object.
    :param numpy.ndarray color_array:
        Array storing the color of each object.

    :returns:
        tuple: Updated DataFrame (`df`), image array (`img_array`), coordinate array (`coordinate_array`),
        and color array (`color_array`).

        Updates include:

        - **Center Coordinates**: Updates the x and y center positions (`center_x`, `center_y`) in the DataFrame.
        - **Shape Attributes**: Updates the major axis length (`AreaShape_MajorAxisLength`),
          minor axis length (`AreaShape_MinorAxisLength`), and orientation (`AreaShape_Orientation`) in radians.
        - **Parent Information**: Resets the parent image number and parent object number to `0`.
    """

    this_region_color = regions_color[faulty_row['row idx par']]

    # Update the img_array with the new color for this region
    img_array[tuple(zip(*regions_coordinates[faulty_row['row idx par']]))] = \
        this_region_color

    # update dataframe
    updates = {
        **{x_center_col: regions_features[faulty_row['row idx par']]['center_x']
           for x_center_col in all_center_coord_cols['x']},
        **{y_center_col: regions_features[faulty_row['row idx par']]['center_y']
           for y_center_col in all_center_coord_cols['y']},
        "AreaShape_MajorAxisLength": regions_features[faulty_row['row idx par']]['major'],
        "AreaShape_MinorAxisLength": regions_features[faulty_row['row idx par']]['minor'],
        "AreaShape_Orientation":
            -(regions_features[faulty_row['row idx par']]['orientation'] + 90) * np.pi / 180,
        parent_image_number_col: 0,
        parent_object_number_col: 0,
    }

    this_bac_ndx = faulty_row['cp index']

    coordinates_array = regions_coordinates[faulty_row['row idx par']]
    x, y = coordinates_array[:, 0], coordinates_array[:, 1]
    # Cantor Pairing Function
    encoded_numbers = (x + y) * (x + y + 1) // 2 + y
    coordinate_array[this_bac_ndx, encoded_numbers] = True
    color_array[this_bac_ndx] = this_region_color

    # Update the DataFrame for the given index (bac_ndx) with all updates
    df.loc[this_bac_ndx, updates.keys()] = updates.values()

    return df, img_array, coordinate_array, color_array


def multi_region_correction(df, img_array, distance_df_cp_connected_regions, img_number,
                            regions_center_particles_stat, regions_center_particles, regions_color,
                            regions_coordinates, regions_features, all_rel_center_coord_cols,
                            parent_image_number_col, parent_object_number_col, min_distance_prev_objects_df,
                            coordinate_array, color_array):
    """
    Corrects multi-region objects by identifying mismatched or faulty associations and updating or removing objects
    as necessary. Ensures consistency between regions and objects while maintaining accurate data records.

    :param pandas.DataFrame df:
        Input DataFrame containing object data.
    :param numpy.ndarray img_array:
        Image array representing the current state of the image.
    :param pandas.DataFrame distance_df_cp_connected_regions:
        DataFrame containing pairwise distances between objects (based on CellProfiler records) and regions.
    :param int img_number:
        Current image number
    :param list regions_center_particles_stat:
        List of status for regions' center points.
    :param list regions_center_particles:
        List of coordinates for regions' center points.
    :param list regions_color:
        list of region colors in current time step.
    :param list of numpy.ndarray regions_coordinates:
        A list of arrays where each region index maps to an array of its coordinates.
    :param list of dict regions_features:
        A list of dictionaries where each region index maps to its features (e.g., orientation, axis lengths).
    :param dict all_rel_center_coord_cols:
        Dictionary containing x and y coordinate column names.
    :param str parent_image_number_col:
        Name of the parent image number column in the DataFrame.
    :param str parent_object_number_col:
        Name of the parent object number column in the DataFrame.
    :param pandas.DataFrame min_distance_prev_objects_df:
        DataFrame containing previous object distance information.
    :param csr_matrix coordinate_array:
        Boolean array tracking encoded spatial coordinates for each object.
    :param numpy.ndarray color_array:
        Array storing the colors of objects.

    :returns:
        tuple: Updated DataFrame (`df`), coordinate array (`coordinate_array`), and color array (`color_array`).
    """

    rows_with_duplicates = distance_df_cp_connected_regions.T.apply(lambda row: (row == row.min()).sum() > 1, axis=1)

    # two same regions from multi regions
    if rows_with_duplicates.any():

        cols_should_remove = []
        rows_should_remove = []

        for cp_idx in distance_df_cp_connected_regions.columns[rows_with_duplicates]:

            cols_should_remove.append(cp_idx)

            df.at[cp_idx, 'noise_bac'] = True

            row_idxs = \
                distance_df_cp_connected_regions.loc[
                    distance_df_cp_connected_regions[cp_idx] == distance_df_cp_connected_regions[cp_idx].min()
                ].index.values

            rows_should_remove.extend(row_idxs)
            # Update the img_array with the new color for this region
            for row_idx in row_idxs:
                img_array[tuple(zip(*regions_coordinates[row_idx]))] = (0, 0, 0)

        # Drop from distance_df_cp_connected_regions
        distance_df_cp_connected_regions = distance_df_cp_connected_regions.drop(
            index=rows_should_remove,
            columns=cols_should_remove
        )

        distance_df_cp_connected_regions.to_csv('distance_df_cp_connected_regions_after.csv')

        print(
            f"Duplicate regions detected in multi-region correction. "
            f"Time step: {img_number}. This situation must be resolved."
        )

    distance_df_particles_min_val_idx = distance_df_cp_connected_regions.idxmin()
    distance_df_particles_min_val = distance_df_cp_connected_regions.min()

    # Extracting corresponding values
    regions_center_min_particles_stat = [regions_center_particles_stat[i] for i in
                                         distance_df_particles_min_val_idx]
    regions_center_particles_x = [regions_center_particles[i][0] for i in distance_df_particles_min_val_idx]
    regions_center_particles_y = [regions_center_particles[i][1] for i in distance_df_particles_min_val_idx]

    min_distance_particles_df = pd.DataFrame({
        'cp index': distance_df_cp_connected_regions.columns,
        'row idx par': distance_df_particles_min_val_idx,
        'Cost par': distance_df_particles_min_val,
        'stat par': regions_center_min_particles_stat,
        'center_x_par': regions_center_particles_x,
        'center_y_par': regions_center_particles_y,
    })

    merged_distance_df = pd.merge(min_distance_prev_objects_df, min_distance_particles_df, on='cp index')

    # if both stats are `natural` --> compare stat is True
    merged_distance_df['compare_stat'] = merged_distance_df['stat prev'] == merged_distance_df['stat par']

    # how it can be possible stat = False:
    # 1. prev: Multi - particle: N
    # 2. prev: Multi - particle: particle
    # 3. prev: natural - particle: particle
    faulty_rows_df = merged_distance_df.loc[merged_distance_df['compare_stat'] == False]

    correct_rows_df = merged_distance_df.loc[merged_distance_df['compare_stat'] == True]

    # noise regions
    par_not_in_min_df = [idx for idx in distance_df_cp_connected_regions.index if idx not in
                         min_distance_particles_df['row idx par'].values]

    for faulty_row_ndx, faulty_row in faulty_rows_df.iterrows():

        number_of_occ_par = \
            merged_distance_df.loc[(merged_distance_df['row idx par'] == faulty_row['row idx par']) &
                                   (merged_distance_df['Cost par'] < merged_distance_df['Cost prev'])].shape[0]

        # (merged_distance_df['row idx prev'] == faulty_row['row idx prev'])
        second_cond_number_of_occ_par = \
            merged_distance_df.loc[(merged_distance_df['row idx par'] == faulty_row['row idx par']) &
                                   (merged_distance_df['stat prev'] == 'multi')].shape[0]

        if number_of_occ_par > 1 or second_cond_number_of_occ_par > 1:
            df, img_array = remove_object_from_image(df, faulty_row, img_array, regions_coordinates)

        elif faulty_row['stat prev'] == 'natural':

            # stat prev : natural
            # Idea: There should be an object that maps to the natural region in both data frames.
            # Otherwise, if all the records mapped to the natural region in the `prev` df are mapped to particles
            # in the `particle` df, it can indicate an error. So we delete them.
            number_of_occ = \
                merged_distance_df[
                    (merged_distance_df['row idx prev'] == faulty_row['row idx prev']) &
                    (merged_distance_df['stat par'] == 'natural')
                    ].shape[0]

            if number_of_occ >= 1:
                if faulty_row['stat par'] == 'natural':

                    num_row_in_corrected_rows = \
                        correct_rows_df.loc[correct_rows_df['row idx par'] == faulty_row['row idx par']].shape[0]
                    if num_row_in_corrected_rows > 0:
                        # remove
                        df, img_array = \
                            remove_object_from_image(df, faulty_row, img_array, regions_coordinates)
                    else:
                        df, img_array, coordinate_array, color_array = \
                            modify_existing_object(df, regions_color, faulty_row, img_array, regions_coordinates,
                                                   coordinate_array, color_array)
                else:
                    df, img_array, coordinate_array, color_array = \
                        modify_existing_object(df, regions_color, faulty_row, img_array, regions_coordinates,
                                               coordinate_array, color_array)

            else:
                df, img_array = \
                    remove_object_from_image(df, faulty_row, img_array, regions_coordinates)

        elif faulty_row['Cost par'] < faulty_row['Cost prev']:

            if faulty_row['stat par'] == 'natural':

                num_row_in_corrected_rows = \
                    correct_rows_df.loc[correct_rows_df['row idx par'] == faulty_row['row idx par']].shape[0]
                if num_row_in_corrected_rows > 0:
                    # remove
                    df, img_array = \
                        remove_object_from_image(df, faulty_row, img_array, regions_coordinates)
                else:
                    # stat prev: multi, stat par: particle or natural!
                    df, img_array, coordinate_array, color_array = \
                        modify_existing_object(df, regions_color, faulty_row, img_array, regions_coordinates,
                                               coordinate_array, color_array)
            else:
                # stat prev: multi, stat par: particle or natural!
                df, img_array, coordinate_array, color_array = \
                    modify_existing_object(df, regions_color, faulty_row, img_array, regions_coordinates,
                                           coordinate_array, color_array)

        elif faulty_row['Cost prev'] < faulty_row['Cost par']:

            if faulty_row['stat par'] == 'natural':

                num_row_in_corrected_rows = \
                    correct_rows_df.loc[correct_rows_df['row idx par'] == faulty_row['row idx par']].shape[0]
                if num_row_in_corrected_rows > 0:
                    # remove
                    df, img_array = \
                        remove_object_from_image(df, faulty_row, img_array, regions_coordinates)
                else:
                    df, img_array, coordinate_array, color_array = \
                        update_object_records(df, faulty_row, img_array, regions_coordinates, regions_color,
                                              regions_features, all_rel_center_coord_cols,
                                              parent_image_number_col, parent_object_number_col, coordinate_array,
                                              color_array)
            else:

                df, img_array, coordinate_array, color_array = \
                    update_object_records(df, faulty_row, img_array, regions_coordinates, regions_color,
                                          regions_features, all_rel_center_coord_cols,
                                          parent_image_number_col, parent_object_number_col, coordinate_array,
                                          color_array)

        elif faulty_row['Cost par'] == faulty_row['Cost prev']:
            raise ValueError(
                f"Ambiguous cost match for faulty row at index {faulty_row_ndx}. "
                "Both current and previous costs are identical."
            )

    for par_ndx in par_not_in_min_df:
        # Update the img_array with the background color (0, 0, 0) for this region
        img_array[tuple(zip(*regions_coordinates[par_ndx]))] = (0, 0, 0)

    for correct_bac_idx, correct_bac_row in correct_rows_df.iterrows():
        bac_ndx = correct_bac_row['cp index']
        this_region_color = regions_color[correct_bac_row['row idx par']]

        color_array[bac_ndx] = this_region_color
        coordinates_array = regions_coordinates[correct_bac_row['row idx par']]
        x, y = coordinates_array[:, 0], coordinates_array[:, 1]
        # Cantor Pairing Function
        encoded_numbers = (x + y) * (x + y + 1) // 2 + y
        coordinate_array[bac_ndx, encoded_numbers] = True

    # now check correct_rows_df with duplicate
    correct_rows_dup_df = correct_rows_df[correct_rows_df.duplicated('row idx par', keep=False)]
    for row_idx, faulty_row in correct_rows_dup_df.iterrows():
        df, img_array = \
            remove_object_from_image(df, faulty_row, img_array, regions_coordinates)

    return df, coordinate_array, color_array
