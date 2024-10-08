import numpy as np
import pandas as pd


def modify_existing_object(df, regions_color, faulty_row, img_array, regions_coordinates):
    # It means that the Cellprofiler detected both multi-regions and particles
    this_region_color = regions_color[faulty_row['row idx par']]

    # Update the img_array with the new color for this region
    img_array[tuple(zip(*regions_coordinates[faulty_row['row idx par']]))] = this_region_color

    df.at[faulty_row['cp index'], 'color_mask'] = tuple(this_region_color)
    df.at[faulty_row['cp index'], 'coordinate'] = set(map(tuple, regions_coordinates[faulty_row['row idx par']]))

    return df, img_array


def remove_existing_object(df, faulty_row, img_array, regions_coordinates):
    # Update the img_array with the new color for this region
    img_array[tuple(zip(*regions_coordinates[faulty_row['row idx par']]))] = (0, 0, 0)
    df.at[faulty_row['cp index'], 'noise_bac'] = True

    return df, img_array


def updating_records(df, faulty_row, img_array, regions_coordinates, regions_color, regions_features,
                     all_center_coordinate_columns, parent_image_number_col, parent_object_number_col):
    this_region_color = regions_color[faulty_row['row idx par']]

    # Update the img_array with the new color for this region
    img_array[tuple(zip(*regions_coordinates[faulty_row['row idx par']]))] = \
        this_region_color

    # update dataframe
    updates = {
        'color_mask': tuple(this_region_color),
        **{x_center_col: regions_features[faulty_row['row idx par']]['center_x']
           for x_center_col in all_center_coordinate_columns['x']},
        **{y_center_col: regions_features[faulty_row['row idx par']]['center_y']
           for y_center_col in all_center_coordinate_columns['y']},
        "AreaShape_MajorAxisLength": regions_features[faulty_row['row idx par']]['major'],
        "AreaShape_MinorAxisLength": regions_features[faulty_row['row idx par']]['minor'],
        "AreaShape_Orientation":
            -(regions_features[faulty_row['row idx par']]['orientation'] + 90) * np.pi / 180,
        parent_image_number_col: 0,
        parent_object_number_col: 0,
        'coordinate': set(map(tuple, regions_coordinates[faulty_row['row idx par']]))
    }

    this_bac_ndx = faulty_row['cp index']

    # Update the DataFrame for the given index (bac_ndx) with all updates
    df.loc[this_bac_ndx, updates.keys()] = updates.values()

    return df, img_array


def multi_region_correction(df, img_array, img_npy_file, distance_df_particles, img_number,
                            regions_center_particles_stat, regions_center_particles, regions_color,
                            regions_coordinates, regions_features, all_center_coordinate_columns,
                            parent_image_number_col, parent_object_number_col, min_distance_prev_objects_df):
    rows_with_duplicates = distance_df_particles.T.apply(lambda row: (row == row.min()).sum() > 1, axis=1)

    # two same regions from multi regions
    if rows_with_duplicates.any():
        print("two same regions from multi regions")
        print('time step: ' + str(img_number))
        breakpoint()

    else:

        distance_df_particles_min_val_idx = distance_df_particles.idxmin()
        distance_df_particles_min_val = distance_df_particles.min()

        # Extracting corresponding values
        regions_center_min_particles_stat = [regions_center_particles_stat[i] for i in
                                             distance_df_particles_min_val_idx]
        regions_center_particles_x = [regions_center_particles[i][0] for i in distance_df_particles_min_val_idx]
        regions_center_particles_y = [regions_center_particles[i][1] for i in distance_df_particles_min_val_idx]

        min_distance_particles_df = pd.DataFrame({
            'cp index': distance_df_particles.columns,
            'row idx par': distance_df_particles_min_val_idx,
            'Cost par': distance_df_particles_min_val,
            'stat par': regions_center_min_particles_stat,
            'center_x_par': regions_center_particles_x,
            'center_y_par': regions_center_particles_y,
        })

        merged_distance_df = pd.merge(min_distance_prev_objects_df, min_distance_particles_df, on='cp index')

        # merged_distance_df['center_x_compare'] = \
        #    merged_distance_df['center_x_par'] - merged_distance_df['center_x_prev']
        # merged_distance_df['center_y_compare'] = \
        #    merged_distance_df['center_y_par'] - merged_distance_df['center_y_prev']
        # if both stats are `natural` --> compare stat is True
        merged_distance_df['compare_stat'] = merged_distance_df['stat prev'] == merged_distance_df['stat par']

        # faulty_rows_df = merged_distance_df.loc[(merged_distance_df['compare_stat'] == False) |
        #                                        (merged_distance_df['center_x_compare'] != 0) |
        #                                        (merged_distance_df['center_y_compare'] != 0)]

        # how it can be possible stat = False:
        # 1. prev: Multi - particle: N
        # 2. prev: Multi - particle: particle
        # 3. prev: natural - particle: particle
        faulty_rows_df = merged_distance_df.loc[merged_distance_df['compare_stat'] == False]

        # correct_rows_df = merged_distance_df.loc[~merged_distance_df.index.isin(faulty_rows_df.index)]
        correct_rows_df = merged_distance_df.loc[merged_distance_df['compare_stat'] == True]

        # noise regions
        par_not_in_min_df = [idx for idx in distance_df_particles.index if idx not in
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
                df, img_array = \
                    remove_existing_object(df, faulty_row, img_array, regions_coordinates)

            elif faulty_row['stat prev'] == 'natural':

                # stat prev : natural
                # idea: Idea: There should be an object that maps to the natural region in both data frames.
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
                                remove_existing_object(df, faulty_row, img_array, regions_coordinates)
                        else:
                            df, img_array = \
                                modify_existing_object(df, regions_color, faulty_row, img_array, regions_coordinates)
                    else:
                        df, img_array = \
                            modify_existing_object(df, regions_color, faulty_row, img_array, regions_coordinates)

                else:
                    df, img_array = \
                        remove_existing_object(df, faulty_row, img_array, regions_coordinates)

            elif faulty_row['Cost par'] < faulty_row['Cost prev']:

                if faulty_row['stat par'] == 'natural':

                    num_row_in_corrected_rows = \
                        correct_rows_df.loc[correct_rows_df['row idx par'] == faulty_row['row idx par']].shape[0]
                    if num_row_in_corrected_rows > 0:
                        # remove
                        df, img_array = \
                            remove_existing_object(df, faulty_row, img_array, regions_coordinates)
                    else:
                        # stat prev: multi, stat par: particle or natural!
                        df, img_array = \
                            modify_existing_object(df, regions_color, faulty_row, img_array, regions_coordinates)
                else:
                    # stat prev: multi, stat par: particle or natural!
                    df, img_array = \
                        modify_existing_object(df, regions_color, faulty_row, img_array, regions_coordinates)

            elif faulty_row['Cost prev'] < faulty_row['Cost par']:

                if faulty_row['stat par'] == 'natural':

                    num_row_in_corrected_rows = \
                        correct_rows_df.loc[correct_rows_df['row idx par'] == faulty_row['row idx par']].shape[0]
                    if num_row_in_corrected_rows > 0:
                        # remove
                        df, img_array = \
                            remove_existing_object(df, faulty_row, img_array, regions_coordinates)
                    else:
                        df, img_array, = \
                            updating_records(df, faulty_row, img_array, regions_coordinates, regions_color,
                                             regions_features, all_center_coordinate_columns,
                                             parent_image_number_col, parent_object_number_col)
                else:

                    df, img_array, = \
                        updating_records(df, faulty_row, img_array, regions_coordinates, regions_color,
                                         regions_features, all_center_coordinate_columns,
                                         parent_image_number_col, parent_object_number_col)

            elif faulty_row['Cost par'] == faulty_row['Cost prev']:
                breakpoint()

        for par_ndx in par_not_in_min_df:
            # Update the img_array with the background color (0, 0, 0) for this region
            img_array[tuple(zip(*regions_coordinates[par_ndx]))] = (0, 0, 0)

        for correct_bac_idx, correct_bac_row in correct_rows_df.iterrows():
            bac_ndx = correct_bac_row['cp index']
            this_region_color = regions_color[correct_bac_row['row idx par']]
            df.at[bac_ndx, 'color_mask'] = tuple(this_region_color)
            df.at[bac_ndx, 'coordinate'] = set(map(tuple, regions_coordinates[correct_bac_row['row idx par']]))

        # now check correct_rows_df with duplicate
        correct_rows_dup_df = correct_rows_df[correct_rows_df.duplicated('row idx par', keep=False)]
        for row_idx, faulty_row in correct_rows_dup_df.iterrows():
            df, img_array = \
                remove_existing_object(df, faulty_row, img_array, regions_coordinates)

        # Save the modified img_array to a new .npy file
        # new_file_name = os.path.splitext(img_npy_file)[0] + '_modified.npy'
        # np.save(new_file_name, img_array)

    return df
