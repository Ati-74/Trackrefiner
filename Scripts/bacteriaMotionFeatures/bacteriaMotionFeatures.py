import pandas as pd
import numpy as np
from helper import calculate_all_bac_endpoints, calculate_all_bac_slopes, calculate_angles_between_slopes, \
    calculate_trajectory_angles
from functools import reduce


if __name__ == '__main__':

    # interval time unit: minute
    datasets_dict = {'unconstrained_1': {'interval_time': 2}, 'unconstrained_2': {'interval_time': 2},
                     'unconstrained_3': {'interval_time': 2}, 'unconstrained_4': {'interval_time': 10},
                     'constrained_1': {'interval_time': 3}, 'constrained_3': {'interval_time': 2},
                     'constrained_4': {'interval_time': 2}, 'constrained_2': {'interval_time': 5}}

    center_coord_cols = {'x': 'Center_X', 'y': 'Center_Y'}

    gt_out_dir = '../../09.GroundTruth/'

    out_dir = '../../11.frame_wise_features'

    for dataset_name in datasets_dict.keys():

        gt_df = (
            pd.read_csv(f'{gt_out_dir}/ground_truth_{dataset_name}/post_processing/GT_bacteria_feature_analysis.csv'))

        interval_time = datasets_dict[dataset_name]['interval_time']

        # calculate endpoints
        gt_df = calculate_all_bac_endpoints(gt_df, center_coord_cols)
        gt_df = calculate_all_bac_slopes(gt_df)

        gt_df[['prev_Center_X', 'prev_Center_Y']] = gt_df.groupby('id')[['Center_X', 'Center_Y']].shift(1)
        gt_df[['prev_endpoint1_X', 'prev_endpoint1_Y']] = gt_df.groupby('id')[['endpoint1_X', 'endpoint1_Y']].shift(1)
        gt_df[['prev_endpoint2_X', 'prev_endpoint2_Y']] = gt_df.groupby('id')[['endpoint2_X', 'endpoint2_Y']].shift(1)
        gt_df['prev_bacteria_slope'] = gt_df.groupby('id')['bacteria_slope'].shift(1)
        unique_ids = gt_df.drop_duplicates(subset='id', keep='last')

        # daughters
        daughters_at_division_time = gt_df.loc[(gt_df['prev_Center_X'].isna()) & (gt_df['parent_id'] != 0)]

        daughters_with_parents = daughters_at_division_time.merge(unique_ids, left_on='parent_id', right_on='id',
                                                                  suffixes=('_daughter', '_parent'))

        gt_df.loc[daughters_at_division_time.index, ['prev_Center_X', 'prev_Center_Y']] = (
            daughters_with_parents)[['Center_X_parent', 'Center_Y_parent']].values

        gt_df.loc[daughters_at_division_time.index, ['prev_endpoint1_X', 'prev_endpoint1_Y']] = (
            daughters_with_parents)[['endpoint1_X_parent', 'endpoint1_Y_parent']].values

        gt_df.loc[daughters_at_division_time.index, ['prev_endpoint2_X', 'prev_endpoint2_Y']] = (
            daughters_with_parents)[['endpoint2_X_parent', 'endpoint2_Y_parent']].values

        gt_df.loc[daughters_at_division_time.index, ['prev_bacteria_slope']] = (
            daughters_with_parents)[['bacteria_slope_parent']].values

        gt_df = gt_df.loc[gt_df['TrackObjects_ParentImageNumber_50'] != 0]

        gt_df['speed movement_center'] = np.sqrt((gt_df['Center_X'] - gt_df['prev_Center_X']) ** 2 +
                                                 (gt_df['Center_Y'] - gt_df['prev_Center_Y']) ** 2) / interval_time

        gt_df['angular_motion'] = \
            calculate_angles_between_slopes(gt_df['prev_bacteria_slope'].values, gt_df['bacteria_slope'].values)

        average_angular_motion_df = \
            gt_df.groupby('TimeStep')['angular_motion'].mean().reset_index(name='avg_angular_motion')

        # "center_angle", "endpoint1_angle", "endpoint2_angle"
        avg_speed_movement_center_df = gt_df.groupby('TimeStep')['speed movement_center'].mean().reset_index(
                name='avg_speed movement_center')

        # List of all DataFrames to merge
        dfs = [
            average_angular_motion_df,
            avg_speed_movement_center_df
        ]

        # Merge all DataFrames on 'TimeStep' using reduce
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='TimeStep', how='inner'), dfs)

        merged_df.to_csv(f'{out_dir}/{dataset_name}/bacteriaMotionFeatures.csv', index=False)
