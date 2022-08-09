import os
import networkx as nx
import pandas as pd
import numpy as np
import pickle
from ConnectedComponents import edge_list_to_adjacency_matrix, plot_graph
from DataCorrection import finding_incorrect_bacteria
from summary_statistics import fit_ellipse, aspect_ratio_calc, anisotropy_calc, summary_statistic_plot


class Dict2Class(object):
    """
        goal:  Change Dictionary Into Class

    """

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def create_pickle_files(micro_colonies_in_current_time_step, local_aspect_ratio_list, local_anisotropy_list, time_step,
                        num_micro_colonies, path):

    if not local_anisotropy_list:
        local_anisotropy_list = [''] * len(local_aspect_ratio_list)
    if not local_aspect_ratio_list:
        local_aspect_ratio_list = [''] * len(local_anisotropy_list)

    df = pd.DataFrame(list(zip(micro_colonies_in_current_time_step, local_aspect_ratio_list, local_anisotropy_list)),
                      columns=['micro_colony_ids', 'aspect_ratio', 'anisotropy'])
    # convert to dictionary
    df_to_dict = df.to_dict('index')
    # change dictionary to class
    for dictionary_key in df_to_dict.keys():
        df_to_dict[dictionary_key] = Dict2Class(df_to_dict[dictionary_key])
    data = {"stepNum": time_step, "num_micro_colonies": num_micro_colonies, "micro_colonies_States": df_to_dict}
    write_to_pickle_file(data, path, time_step)


def write_to_pickle_file(data, path, time_step):
    output_file = path + "/pickle/step-" + str(time_step) + ".pickle"

    with open(output_file, "wb") as export:
        pickle.dump(data, export, protocol=-1)


def remove_incorrect_bac(adjacency_matrix, incorrect_bac_in_current_time_step):
    """
    Goal: some bacteria are detected wrongly by CP, so that we stored them (by the function
    called "finding_incorrect_bacteria"). What "remove_incorrect_bac" does is removing
    incorrect bacteria from adjacency matrix.
    
    @param adjacency_matrix                            data frame   adjacency matrix of the current time-step
    @param incorrect_bac_in_current_time_step   data frame   "ImageNumber", "ObjectNumber" of incorrect bac
    in current time-step

    """

    incorrect_index = incorrect_bac_in_current_time_step["ObjectNumber"].values
    # remove rows
    adjacency_matrix.drop(incorrect_index, axis=0, inplace=True)
    # remove columns
    adjacency_matrix.drop(incorrect_index, axis=1, inplace=True)

    return adjacency_matrix


def finding_sub_graphs(graph):
    """
    Goal: Finding the largest disconnected sub graphs

    @param graph   graph  neighboring bacteria graph in current time-step
    """
    sub_graphs = sorted(nx.connected_components(graph), key=len, reverse=True)

    return sub_graphs


def micro_colony_analysis(cells_info_file, neighbor_file, output_directory, summary_statistic_method):
    """
    Goal: this is the main function; the function calls other function that are described above and
    calculates aspect ratio for each micro colony in each time-step and store the outputs in png format and pickle.

    @param cells_info_file            dataframe  all the required information of bac. like id, orientation,etc.
    @param neighbor_file              dataframe  object relationship file that contains info. about each cell's neighbor
    @param summary_statistic_method   string     the method that we apply on the micro-colonies
    
    """

    # minimum size of micro colony
    min_size_of_micro_colony = 2

    # store aspect ratio of micro colonies
    aspect_ratio_list = []
    anisotropy_list = []

    # read cells information
    cell_information_df = pd.read_csv(cells_info_file)
    # finding incorrect bacteria
    # I define a gap number to find parent in other previous time steps
    number_of_gap = 1
    df_correct, df_incorrect = finding_incorrect_bacteria(cell_information_df, number_of_gap)

    # read neighbor file
    neighbor_df = pd.read_csv(neighbor_file)
    # filter `MeasureObjectNeighbors` module rows
    neighbor_df = neighbor_df.loc[neighbor_df["Module"] == "MeasureObjectNeighbors"]

    # In order to identify merged micro colonies, micro colony labels are stored.
    micro_colonies = []
    time_steps = list(set(df_correct["ImageNumber"].values))

    for time_step in time_steps:

        # store ellipses
        ellipses = []
        # store anisotropies of micro colonies of this rime step
        local_anisotropy_list = []
        # store anisotropies of micro colonies of this rime step
        local_aspect_ratio_list = []
        micro_colonies_in_current_time_step = []

        print('Time step: ' + str(time_step))

        # cells information in current time step
        df_current_time_step = df_correct.loc[df_correct["ImageNumber"] == time_step]
        # second image number is not checked because the neighborhood relationship is
        # a time-step to time-step relationship.
        edge_df = neighbor_df.loc[neighbor_df["First Image Number"] == time_step]
        adjacency_matrix = edge_list_to_adjacency_matrix(edge_df["First Object Number"],
                                                         edge_df["Second Object Number"])

        # find and remove incorrect bacteria in current time step from adjacency matrix
        incorrect_bac_in_current_time_step = df_incorrect.loc[df_incorrect["ImageNumber"] == time_step]

        if not incorrect_bac_in_current_time_step.empty:
            adjacency_matrix = remove_incorrect_bac(adjacency_matrix, incorrect_bac_in_current_time_step)
        # making graph
        G = nx.from_pandas_adjacency(adjacency_matrix)
        # finding sub graphs
        sub_graphs = finding_sub_graphs(G)

        for subgraph in sub_graphs:
            if len(subgraph) >= min_size_of_micro_colony:
                nodes = list(subgraph)
                # find unique bacteria labels
                bacteria_label = list(set(df_current_time_step[df_current_time_step["ObjectNumber"].isin(nodes)]
                                          ["TrackObjects_Label_50"].values.tolist()))
                # check micro colony
                common_micro_colony_index = []

                for mico_colony_index, micro_colony in enumerate(micro_colonies):
                    common_elements = list(set(micro_colony).intersection(bacteria_label))
                    if common_elements:
                        common_micro_colony_index.append(mico_colony_index)

                if len(common_micro_colony_index) <= 1:  # As a result, the micro_colonies were not merged
                    # Add new labels to its micro_colony list
                    if common_micro_colony_index:
                        micro_colonies[common_micro_colony_index[0]].extend(bacteria_label)
                        micro_colonies[common_micro_colony_index[0]] = \
                            list(set(micro_colonies[common_micro_colony_index[0]]))
                    else:
                        micro_colonies.append(bacteria_label)
                    # bacteria in this micro_colony
                    bac_in_micro_colony = df_current_time_step[df_current_time_step["ObjectNumber"].isin(nodes)]
                    micro_colonies_in_current_time_step.append(nodes)

                    # fit ellipse
                    ellipse_params = fit_ellipse(bac_in_micro_colony)
                    # append ellipse
                    ellipses.append(ellipse_params)

                    """
                        calculation of aspect ratio
                    """
                    if "Aspect Ratio" in summary_statistic_method:
                        aspect_ratio = aspect_ratio_calc(ellipse_params)
                        # store aspect ratio
                        aspect_ratio_list.append(aspect_ratio)
                        local_aspect_ratio_list.append(aspect_ratio)
                    """
                        Finish
                    """

                    """
                        calculation of Anisotropy
                    """
                    if "Anisotropy" in summary_statistic_method:
                        mean_anisotropy = anisotropy_calc(bac_in_micro_colony)
                        # store anisotropy
                        anisotropy_list.append(mean_anisotropy)
                        local_anisotropy_list.append(mean_anisotropy)
                    """
                        Finish
                    """

        '''                
            show summary statistics of micro colonies in plot
        '''

        # aspect ratio
        if "Aspect Ratio" in summary_statistic_method:
            summary_statistic_plot(df_current_time_step, ellipses, time_step, output_directory,
                                   local_aspect_ratio_list, "Aspect Ratio")

        # anisotropy
        if "Anisotropy" in summary_statistic_method:
            summary_statistic_plot(df_current_time_step, ellipses, time_step, output_directory,
                                   local_anisotropy_list, "Anisotropy")

        # plot graph
        plot_graph(G, time_step, output_directory)

        # save micro_colonies information
        create_pickle_files(micro_colonies_in_current_time_step, local_aspect_ratio_list, local_anisotropy_list,
                            time_step, len(ellipses), output_directory)


def process_data(input_file, relationship_file, output_directory_path, summary_statistic_method_list):
    # create directories
    for summary_statistic_method in summary_statistic_method_list:
        summary_statistic_images_path = (output_directory_path + "/img/" + summary_statistic_method + "/")
        if not os.path.exists(summary_statistic_images_path):
            os.makedirs(summary_statistic_images_path)

    graph_images_path = output_directory_path + "/img/graph/"
    pickle_path = output_directory_path + "/pickle/"

    if not os.path.exists(graph_images_path):
        os.makedirs(graph_images_path)
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)

    # start analysis
    micro_colony_analysis(input_file, relationship_file, output_directory_path, summary_statistic_method_list)
