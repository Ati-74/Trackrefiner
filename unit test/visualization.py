from CellProfilerAnalysis.strain.ProcessCellProfilerData import process_data
from CellProfilerAnalysis.strain.correction.action.processing import find_vertex, bacteria_features, convert_to_pixel, angle_convert_to_radian
from CellProfilerAnalysis.strain.correction.Find_Fix_Errors import data_modification
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cv2


def text_position(ends, pos):

    # adding text inside the plot
    pos1x = np.abs(ends[0][0] + pos[0]) / 2
    pos2x = np.abs(ends[1][0] + pos[0]) / 2

    pos1y = np.abs(ends[0][1] + pos[1]) / 2
    pos2y = np.abs(ends[1][1] + pos[1]) / 2

    final_pos1x = np.abs(pos1x + pos[0]) / 2
    final_pos2x = np.abs(pos2x + pos[0]) / 2

    final_pos1y = np.abs(pos1y + pos[1]) / 2
    final_pos2y = np.abs(pos2y + pos[1]) / 2

    return [[final_pos1x, final_pos1y], [final_pos2x, final_pos2y]]


def bacteria_plot(pickle_data, current_time_step_data, raw_img, output_dir):

    # draw Objects
    fig, ax = plt.subplots(1, 2, num="step-" + '0' * (6-len(str(pickle_data['stepNum']))) + str(pickle_data['stepNum']))
    # setting font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.default'] = 'regular'

    # read image file
    img = cv2.imread(raw_img)
    ax[0].imshow(img)
    ax[1].imshow(img)

    # raw data
    for bac_index, bac in current_time_step_data.iterrows():

        features = bacteria_features(bac)
        major = features['major']
        radius = features['radius']
        orientation = features['orientation']
        center_x = features['center_x']
        center_y = features['center_y']
        ends = find_vertex([center_x, center_y], major, orientation)

        # plot bacterium
        ax[0].plot([ends[0][0], ends[1][0]], [ends[0][1], ends[1][1]], lw=radius, solid_capstyle="round")
        # add text
        [pos1, pos2] = text_position(ends, [center_x, center_y])
        ax[0].text(pos1[0], pos1[1], int(bac['id']), fontsize=6, color="yellow")
        ax[0].text(pos2[0], pos2[1], int(bac['parent_id']), fontsize=6, color="purple")

    for bac_id in pickle_data['cellStates'].keys():
        bacterium = pickle_data['cellStates'][bac_id]
        major, radius, ends, pos = convert_to_pixel(bacterium.length, bacterium.radius, bacterium.ends, bacterium.pos,
                                                     um_per_pixel=0.144)
        # plot bacterium
        ax[1].plot([ends[0][0], ends[1][0]], [ends[0][1], ends[1][1]], lw=radius, solid_capstyle="round")
        # add text
        [pos1, pos2] = text_position(ends, pos)
        ax[1].text(pos1[0], pos1[1], int(bac_id), fontsize=6, color="yellow")
        if pickle_data['lineage']:
            ax[1].text(pos2[0], pos2[1], int(pickle_data['lineage'][bac_id]), fontsize=6, color="purple")
        else:
            ax[1].text(pos2[0], pos2[1], 0, fontsize=6, color="purple")

    # title
    ax[0].title.set_text('CellProfiler Output')
    ax[1].title.set_text('Post processing output')

    # legend
    id_patch = mpatches.Patch(color='yellow', label='identity id')
    parent_id_patch = mpatches.Patch(color='purple', label='parent id')

    fig.legend(handles=[id_patch, parent_id_patch], loc='upper right', ncol=6, bbox_to_anchor=(.7, 1),
               prop={'size': 12})

    fig.tight_layout()
    plt.show()
    fig.savefig(output_dir + "step-" + '0' * (6-len(str(pickle_data['stepNum']))) + str(pickle_data['stepNum']) +
                ".png", dpi=600)
    # close fig
    # fig.clf()
    plt.close()


def visualization(img_dir, pickle_files_dir, cp_output_csv_file, output_dir):
    """
    @param img_dir str raw images directory
    @param pickle_files_dir str pickle files directory
    @param cp_output_csv_file .csv CellProfiler output csv file
    @param output_dir str output directory
    """

    raw_img_list = [filename for filename in sorted(glob.glob(img_dir + '*.tif'))]
    pickle_file_list = [filename for filename in sorted(glob.glob(pickle_files_dir + '*.pickle'))]

    raw_df = pd.read_csv(cp_output_csv_file)
    # convert to radian
    raw_df = data_modification(raw_df, intensity_threshold=0.1)
    raw_df = angle_convert_to_radian(raw_df)
    time_steps = raw_df['ImageNumber'].unique()

    for index, img in enumerate(raw_img_list):
        selected_pickle_file = pickle_file_list[index]
        # read pickle file
        pickle_data = pickle.load(open(selected_pickle_file, 'rb'))
        current_time_step_data = raw_df.loc[raw_df['ImageNumber'] == time_steps[index]]

        bacteria_plot(pickle_data, current_time_step_data, img, output_dir)


if __name__ == '__main__':

    # raw images directory
    img_dir = '../examples/K12/raw img/'
    # pickle files directory
    pickle_files_dir = '../examples/K12/outputs/'
    # CellProfiler output csv file
    cp_output_csv_file = '../examples/K12/K12_CP_output.csv'
    # img output directory
    output_dir = '../examples/K12/compare/'

    visualization(img_dir, pickle_files_dir, cp_output_csv_file, output_dir)
