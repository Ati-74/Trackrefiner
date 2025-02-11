import pandas as pd
from scipy.stats import pearsonr


if __name__ == '__main__':

    """
    Script for computing correlations.

    This script processes multiple datasets by merging bacterial tracking data with error reports, 
    ground truth counts, and frame-wise motion features. It then computes the Pearson correlation 
    between specific movement features and error percentages.

    Workflow:
    1. Reads density, ground truth, frame-wise motion features, and error report CSVs for each dataset.
    2. Merges these data sources based on the `ImageNumber` column.
    3. Computes the Pearson correlation between selected features and the percentage of errors.
    4. Saves the correlation results to a CSV file.

    Parameters:
    - `um_per_pixel` (float): Conversion factor from pixels to micrometers.
    - `features` (list): List of features to analyze for correlation.
    - `frame_wise_out` (str): Path to frame-wise feature files.
    - `gt_out` (str): Path to ground truth data.
    - `performance_out` (str): Path to performance reports.

    Outputs:
    - `correlation.csv` containing the Pearson correlation coefficients and p-values.

    """

    dataset_list = ['unconstrained_1', 'unconstrained_2', 'unconstrained_3', 'unconstrained_4',
                    'constrained_1', 'constrained_2', 'constrained_3', 'constrained_4']

    um_per_pixel = 0.144

    features = ['avg_angular_motion', 'avg_speed movement_center','pixel density']

    frame_wise_out = '../../11.frame_wise_features'
    gt_out = '../../09.GroundTruth'
    performance_out = '../../10.performance_report'

    merge_dict = {}

    for dataset in dataset_list:

        density_path = f'{frame_wise_out}/{dataset}/density.csv'
        gt_path = f'{gt_out}/ground_truth_{dataset}/{dataset}.GT.csv'
        frame_based_analysis_path = f'{frame_wise_out}/{dataset}/bacteriaMotionFeatures.csv'
        errors_each_time_step = f'{performance_out}/performance_report_{dataset}/error_reported_based_on_each_time.csv'

        density_df = pd.read_csv(density_path)
        errors_df = pd.read_csv(errors_each_time_step)
        gr_df = pd.read_csv(gt_path)
        frame_based_analysis_df = pd.read_csv(frame_based_analysis_path)

        density_df = density_df.rename(columns={'Density': 'pixel density'})
        frame_based_analysis_df = frame_based_analysis_df.rename(columns={'TimeStep': 'ImageNumber'})

        merged_df = errors_df.merge(density_df, on='ImageNumber')

        gr_size = gr_df.groupby('ImageNumber').size().reset_index(name='Number of objects in GT')
        merged_df = merged_df.merge(gr_size, on='ImageNumber')
        merged_df = merged_df.merge(frame_based_analysis_df, on='ImageNumber')

        merge_dict[dataset] = merged_df

    rows = []

    for feature in features:
        for dataset in merge_dict.keys():
            errors_each_time_step_df = merge_dict[dataset]

            errors_each_time_step_df = errors_each_time_step_df.loc[errors_each_time_step_df['ImageNumber'] != 1]

            x_val = errors_each_time_step_df[feature].values
            y_val = errors_each_time_step_df['% errors'].values
            # Calculate Spearman's rank correlation
            correlation, p_value = pearsonr(x_val, y_val)
            rows.append([dataset, feature, correlation, p_value])

    df = pd.DataFrame(data=rows, columns=['Dataset', 'Feature', 'Pearson correlation', 'p_value'])
    df.to_csv('../../12.correlation/correlation.csv', index=False)
