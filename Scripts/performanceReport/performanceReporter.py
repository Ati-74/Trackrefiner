import pandas as pd
import numpy as np


def find_total_errors(errors_in_cp_df, only_segmentation_errors_df, unexpected_beginning_segmentation_errors,
                      unexpected_end_segmentation_errors):

    """
    Identify and separate errors due to segmentation and tracking issues.

    This function distinguishes between errors caused by segmentation issues and those
    that are genuine tracking errors. It filters out errors related to unexpected segmentation
    at the beginning or end of tracking and removes noise-based artifacts.

    :param pd.DataFrame errors_in_cp_df:
        DataFrame containing tracking errors in CellProfiler (CP).
    :param pd.DataFrame only_segmentation_errors_df:
        DataFrame containing segmentation errors that do not affect tracking.
    :param pd.DataFrame unexpected_beginning_segmentation_errors:
        DataFrame containing unexpected segmentation errors at the beginning of tracking.
    :param pd.DataFrame unexpected_end_segmentation_errors:
        DataFrame containing unexpected segmentation errors at the end of tracking.

    :return:
        - **errors_really_reason_tracking** (*pd.DataFrame*):
          DataFrame containing only the errors that are truly due to tracking issues.
    """

    errors_cause_only_segmentation = errors_in_cp_df.merge(only_segmentation_errors_df,
                                                           left_on=['ImageNumber', 'ObjectNumber'],
                                                           right_on=['stepNum', 'ObjectNumber'],
                                                           suffixes=('_error', '_seg'), how='inner')

    if errors_cause_only_segmentation.shape[0] != only_segmentation_errors_df.shape[0]:
        breakpoint()

    # first important result
    errors_really_reason_tracking = \
        errors_in_cp_df.loc[~ errors_in_cp_df['index_error'].isin(
            errors_cause_only_segmentation['index_error'].values)]

    # also we should think about UB noises
    # they are not tracking error because by removing then nothing happen in tracking
    ub_noises = errors_really_reason_tracking.loc[
        (errors_really_reason_tracking['TrackObjects_ParentImageNumber_50_gr'].isna()) &
        (errors_really_reason_tracking['TrackObjects_ParentImageNumber_50_cp'] == 0)]

    errors_really_reason_tracking = \
        errors_really_reason_tracking.loc[~ errors_really_reason_tracking['index_error'].isin(
            ub_noises['index_error'].values)]

    if unexpected_beginning_segmentation_errors.shape[0] > 0:

        errors_cause_only_ub_segmentation = errors_in_cp_df.merge(unexpected_beginning_segmentation_errors,
                                                                  left_on=['ImageNumber', 'ObjectNumber'],
                                                                  right_on=['stepNum', 'ObjectNumber'],
                                                                  suffixes=('_error', '_seg'), how='inner')

        if errors_cause_only_ub_segmentation.shape[0] != unexpected_beginning_segmentation_errors.shape[0]:
            breakpoint()

        # first important result
        errors_really_reason_tracking = \
            errors_really_reason_tracking.loc[~ errors_really_reason_tracking['index_error'].isin(
                errors_cause_only_ub_segmentation['index_error'].values)]

    if unexpected_end_segmentation_errors.shape[0] > 0:

        errors_cause_only_ue_segmentation = \
            errors_in_cp_df.merge(unexpected_end_segmentation_errors,
                                  left_on=['TrackObjects_ParentImageNumber_50_cp',
                                           'TrackObjects_ParentObjectNumber_50_cp'],
                                  right_on=['stepNum', 'ObjectNumber'], suffixes=('_error', '_seg'), how='inner')

        # first important result
        errors_really_reason_tracking = \
            errors_really_reason_tracking.loc[~ errors_really_reason_tracking['index_error'].isin(
                errors_cause_only_ue_segmentation['index_error'].values)]

    return errors_really_reason_tracking


def detected_errors(errors_really_reason_tracking_df, undetected_errors_df):

    """
    Identify errors that were detected versus those that were missed.

    This function determines which tracking errors were detected and which were missed
    by comparing the list of known tracking errors against a set of undetected errors.

    :param pd.DataFrame errors_really_reason_tracking_df:
        DataFrame containing tracking-related errors after filtering out segmentation-related errors.
    :param pd.DataFrame undetected_errors_df:
        DataFrame containing errors that were not detected.

    :return:
        - **errors_detected** (*pd.DataFrame*):
          DataFrame containing the detected tracking errors after filtering out undetected ones.
    """

    errors_not_detected = errors_really_reason_tracking_df.merge(undetected_errors_df,
                                                                 left_on=['ImageNumber', 'ObjectNumber'],
                                                                 right_on=['stepNum', 'ObjectNumber'],
                                                                 suffixes=('_error', '_undetected'), how='inner')

    if errors_not_detected.shape[0] != undetected_errors_df.shape[0]:
        print(dataset)

        difference1 = errors_not_detected[
            ~errors_not_detected.set_index(['ImageNumber', 'ObjectNumber']).index.isin(
                undetected_errors_df.set_index(['stepNum', 'ObjectNumber']).index
            )
        ]

        difference2 = undetected_errors_df[
            ~undetected_errors_df.set_index(['stepNum', 'ObjectNumber']).index.isin(
                errors_not_detected.set_index(['ImageNumber', 'ObjectNumber']).index
            )
        ]

        print(difference1)
        print(difference2)

        breakpoint()

    # second important result
    errors_detected = \
        errors_really_reason_tracking_df.loc[
            ~ errors_really_reason_tracking_df['index_error'].isin(errors_not_detected['index_error'].values)]

    return errors_detected


def fixed_errors(errors_detected_df, failed_fix_errors_df):

    """
    Identify errors that were successfully fixed versus those that remained unresolved.

    This function compares detected tracking errors with a list of errors that failed to be corrected
    to determine which tracking errors were successfully resolved.

    :param pd.DataFrame errors_detected_df:
        DataFrame containing detected errors.
    :param pd.DataFrame failed_fix_errors_df:
        DataFrame containing errors that were not successfully fixed.

    :return:
        - **errors_fixed** (*pd.DataFrame*):
          DataFrame containing errors that were successfully corrected.
    """

    errors_failed_in_correction = errors_detected_df.merge(failed_fix_errors_df,
                                                           left_on=['ImageNumber', 'ObjectNumber'],
                                                           right_on=['stepNum', 'ObjectNumber'],
                                                           suffixes=('_error', '_undetected'), how='inner')

    if errors_failed_in_correction.shape[0] != failed_fix_errors_df.shape[0]:
        print(dataset)

        difference1 = errors_failed_in_correction[
            ~errors_failed_in_correction.set_index(['ImageNumber', 'ObjectNumber']).index.isin(
                failed_fix_errors_df.set_index(['stepNum', 'ObjectNumber']).index
            )
        ]

        difference2 = failed_fix_errors_df[
            ~failed_fix_errors_df.set_index(['stepNum', 'ObjectNumber']).index.isin(
                errors_failed_in_correction.set_index(['ImageNumber', 'ObjectNumber']).index
            )
        ]

        print(difference1)
        print(difference2)
        breakpoint()

    # second important result
    errors_fixed = \
        errors_detected_df.loc[
            ~ errors_detected_df['index_error'].isin(errors_failed_in_correction['index_error'].values)]

    return errors_fixed


def calc_performance(ground_truth_df, errors_really_reason_tracking_df, detected_errors_dataframe,
                     errors_fixed_dataframe, trackrefiner_errors_df, noise_objects_dataframe, dict_pref):

    """
    Calculate performance metrics for tracking error detection and correction.

    This function computes key performance indicators, including:
        - Total errors
        - Number of detected errors
        - Number of fixed errors
        - Error detection rate
        - Error fixation rate
        - False discovery rate (FDR)

    :param pd.DataFrame ground_truth_df:
        Ground truth DataFrame containing tracking information.
    :param pd.DataFrame errors_really_reason_tracking_df:
        DataFrame containing only tracking-related errors.
    :param pd.DataFrame detected_errors_dataframe:
        DataFrame containing detected tracking errors.
    :param pd.DataFrame errors_fixed_dataframe:
        DataFrame containing errors that were successfully corrected.
    :param pd.DataFrame trackrefiner_errors_df:
        DataFrame containing incorrectly removed correct links.
    :param pd.DataFrame noise_objects_dataframe:
        DataFrame containing objects identified as noise.
    :param dict dict_pref:
        Dictionary storing dataset-specific performance metrics.

    :return:
        - **dict_pref** (*dict*): Updated dictionary containing calculated performance metrics.
    """

    total_num_errors = errors_really_reason_tracking_df.shape[0]
    num_detected_errors = detected_errors_dataframe.shape[0]
    errors_fixed = errors_fixed_dataframe.shape[0]

    # it shows number of links
    gr_objects_with_links = ground_truth_df.loc[ground_truth_df['TrackObjects_ParentImageNumber_50'] != 0]

    dict_pref[dataset]['Total number of links in GR'] = gr_objects_with_links.shape[0]
    dict_pref[dataset]['number of errors'] = total_num_errors
    dict_pref[dataset]['number of detected errors'] = num_detected_errors
    dict_pref[dataset]['number of fixed errors'] = errors_fixed
    dict_pref[dataset]['number of incorrectly removed correct links'] = trackrefiner_errors_df.shape[0]
    dict_pref[dataset]['Number of noise objects detected'] = noise_objects_dataframe.shape[0]

    gt_num_bac = ground_truth_df.groupby('ImageNumber').size().reset_index(name='number of objects in GR')

    # filtration
    # ignore noise objects
    errors_really_reason_tracking_df = \
        errors_really_reason_tracking_df.loc[
            ~ errors_really_reason_tracking_df['TrackObjects_ParentImageNumber_50_gr'].isna()]
    detected_errors_dataframe = \
        detected_errors_dataframe.loc[~ detected_errors_dataframe['TrackObjects_ParentImageNumber_50_gr'].isna()]
    errors_fixed_dataframe = \
        errors_fixed_dataframe.loc[~ errors_fixed_dataframe['TrackObjects_ParentImageNumber_50_gr'].isna()]

    total_num_errors = \
        errors_really_reason_tracking_df.groupby('ImageNumber').size().reset_index(name='number of errors')
    total_num_detected_errors = \
        detected_errors_dataframe.groupby('ImageNumber').size().reset_index(name='number of detected errors')
    total_num_fixed_errors = \
        errors_fixed_dataframe.groupby('ImageNumber').size().reset_index(name='number of fixed errors')

    merged_df = gt_num_bac.merge(
        total_num_errors, on='ImageNumber', how='left'
    ).merge(
        total_num_detected_errors, on='ImageNumber', how='left'
    ).merge(
        total_num_fixed_errors, on='ImageNumber', how='left'
    )

    merged_df = merged_df.fillna(0)

    merged_df = merged_df.loc[merged_df['number of errors'] > 0]

    # print(total_num_errors)
    # breakpoint()

    merged_df['% errors'] = np.round((merged_df['number of errors'] / merged_df['number of objects in GR']) * 100,
                                     4)

    merged_df['% detected errors'] = \
        np.round((merged_df['number of detected errors'] / merged_df['number of errors']) * 100, 4)

    merged_df['% fixed errors'] = \
        np.round((merged_df['number of fixed errors'] / merged_df['number of errors']) * 100, 4)

    merged_df.to_csv(f'{out_dir}/performance_report_{dataset}/error_reported_based_on_each_time.csv', index=False)

    return dict_pref


if __name__ == '__main__':

    datasets_dict = {'unconstrained_1': {'Interval time (min)': 2, 'doubling time (min)': 20},
                     'unconstrained_2': {'Interval time (min)': 2, 'doubling time (min)': 20},
                     'unconstrained_3': {'Interval time (min)': 2, 'doubling time (min)': 20},
                     'unconstrained_4': {'Interval time (min)': 10, 'doubling time (min)': 20},
                     'constrained_1': {'Interval time (min)': 3, 'doubling time (min)': 20},
                     'constrained_2': {'Interval time (min)': 5, 'doubling time (min)': 20},
                     'constrained_3': {'Interval time (min)': 2, 'doubling time (min)': 45},
                     'constrained_4': {'Interval time (min)': 2, 'doubling time (min)': 45},
                     }

    cp_out_path = '../../04.run_CellProfiler_Omnipose/'
    gt_out_path = '../../09.GroundTruth/'
    tr_out_dir = '../../07.runTrackrefiner/'
    tr_validation_dir = '../../08.Trackrefiner_validation/'
    out_dir = '../../10.performance_report/'

    for dataset in datasets_dict.keys():
        cp_out_df = pd.read_csv(f'{cp_out_path}/cellProfiler_omnipose_{dataset}/FilterObjects.csv')
        gt_df = pd.read_csv(f'{gt_out_path}/ground_truth_{dataset}/{dataset}.GT.csv')
        tr_result = pd.read_csv(f'{tr_out_dir}/trackrefiner_{dataset}/'
                                f'Trackrefiner.FilterObjects_Average_analysis.csv')

        tr_log_df = pd.read_csv(f'{tr_out_dir}/trackrefiner_{dataset}/Trackrefiner.FilterObjects_Average_logs.csv')

        bac_links_change_errors_df = pd.read_csv(f'{tr_validation_dir}/trackrefiner_validation_{dataset}/'
                                                 f'Bacteria_Link_Change_Errors.csv')
        tR_ue_ub_links_df = pd.read_csv(f'{tr_validation_dir}/trackrefiner_validation_{dataset}/'
                                        f'TR.UE_UB_links.csv')
        tr_errors = pd.read_csv(f'{tr_validation_dir}/trackrefiner_validation_{dataset}/'
                                f'Incorrectly_Removed_Correct_Links.csv')

        noise_objects_df = tr_log_df.loc[tr_log_df['NoiseObject'] == True]

        undetected_errors = bac_links_change_errors_df.loc[bac_links_change_errors_df['Undetected Errors'] == "Yes"]

        only_segmentation_errors = bac_links_change_errors_df.loc[
            (bac_links_change_errors_df['Segmentation Errors'] == "Yes") &
            (bac_links_change_errors_df['Successfully Fixed Errors'].isna())]

        ub_segmentation_errors = \
            tR_ue_ub_links_df.loc[(tR_ue_ub_links_df['Segmentation Errors'] == "Yes") &
                                  (~ tR_ue_ub_links_df['Correct Removal Creating Temporary UB'].isna())]

        ue_segmentation_errors = \
            tR_ue_ub_links_df.loc[(tR_ue_ub_links_df['Segmentation Errors'] == "Yes") &
                                  (~ tR_ue_ub_links_df['Correct Removal Creating Temporary UE'].isna())]

        failed_fix_errors = bac_links_change_errors_df.loc[
            bac_links_change_errors_df['Successfully Fixed Errors'] == "No"]

        gt_df['index_gr'] = gt_df.index.values
        tr_result['index_tr'] = tr_result.index.values

        merge_cp_gr = gt_df.merge(cp_out_df, on=['ImageNumber', 'ObjectNumber'], suffixes=('_gr', '_cp'), how='outer')

        errors_in_cp = merge_cp_gr.loc[
            (merge_cp_gr['TrackObjects_ParentImageNumber_50_gr'] !=
             merge_cp_gr['TrackObjects_ParentImageNumber_50_cp']) |
            (merge_cp_gr['TrackObjects_ParentObjectNumber_50_gr'] !=
             merge_cp_gr['TrackObjects_ParentObjectNumber_50_cp'])].copy()

        errors_in_cp['index_error'] = errors_in_cp.index

        # first important result
        errors_really_reason_tracking = find_total_errors(errors_in_cp, only_segmentation_errors,
                                                          ub_segmentation_errors, ue_segmentation_errors)

        # second important result
        detected_errors_df = detected_errors(errors_really_reason_tracking, undetected_errors)

        errors_fixed_df = fixed_errors(detected_errors_df, failed_fix_errors)

        datasets_dict = calc_performance(gt_df, errors_really_reason_tracking, detected_errors_df, errors_fixed_df,
                                         tr_errors, noise_objects_df, datasets_dict)

    df_total_pref = pd.DataFrame.from_dict(datasets_dict, orient='index')

    df_total_pref['Error detection rate'] = np.round((df_total_pref['number of detected errors'] /
                                                      df_total_pref['number of errors']) * 100, 3)

    df_total_pref['Error fixation rate'] = np.round((df_total_pref['number of fixed errors'] /
                                                     df_total_pref['number of errors']) * 100, 3)

    df_total_pref['FDR'] = np.round((df_total_pref['number of incorrectly removed correct links'] /
                                     (df_total_pref['number of fixed errors'] +
                                      df_total_pref['number of incorrectly removed correct links'])) * 100, 3)

    df_total_pref.to_csv(f'{out_dir}/performance report.csv')
