from Script.processDatasets import process_datasets

if __name__ == '__main__':

    dataset_list = ['unconstrained_1', 'unconstrained_2', 'unconstrained_3', 'unconstrained_4',
                    'constrained_1', 'constrained_3', 'constrained_4', 'constrained_2']

    um_per_pixel = 0.144

    cp_omnipose_output_dir = '../../04.run_CellProfiler_Omnipose/'
    gt_output_dir = '../../09.GroundTruth/'

    out_dir = '../../11.frame_wise_features/'

    for i, dataset_name in enumerate(dataset_list):
        time_step_list_this_dataset = 'all'

        print(dataset_name)
        gt_out = f'{gt_output_dir}/ground_truth_{dataset_name}/{dataset_name}.GT.csv'
        cp_omnipose_out = f'{cp_omnipose_output_dir}/cellProfiler_omnipose_{dataset_name}/FilterObjects.csv'
        segmentation_dir = f'{cp_omnipose_output_dir}/cellProfiler_omnipose_{dataset_name}/objects/'
        output_directory = f'{out_dir}/{dataset_name}/'

        # run post-processing
        process_datasets(gt_out_path=gt_out, cp_out_path=cp_omnipose_out, npy_files_dir=segmentation_dir,
                         output_directory=output_directory, um_per_pixel=um_per_pixel,
                         time_step_list=time_step_list_this_dataset)
