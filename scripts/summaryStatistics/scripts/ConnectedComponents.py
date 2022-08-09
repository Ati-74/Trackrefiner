# -*- coding: utf-8 -*-
"""
@author:  Atiyeh Ahmadi - Aaron Yip - Siddh
"""
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


def edge_list_to_adjacency_matrix(col1, col2):

    """
    Goal: neighbor cells' ID (of Object_relationship.csv) will be used to create adjacency data frame
    
    @param col1   data frame   First Object Number from Object_relationship.csv
    @param col2   data frame   Second Object Number from Object_relationship.csv
    """    
    df = pd.crosstab(col1, col2)
    idx = df.columns.union(df.index)
    df = df.reindex(index=idx, columns=idx, fill_value=0)
    return df


def plot_graph(G, time_step, output_directory):
    """
    Goal: plotting neighbor's graph and saving that (in png format) in output_directory
    
    @param G      graph       neighbor's graph
  
    """       
    fig = plt.figure()
    nx.draw_networkx(G, with_labels=True, node_size=200, font_size=9)
    plt.title("Graph of neighboring bacteria at timestep " + str(time_step))
    # plt.show()
    fig.savefig(output_directory + "/img/graph/graph_t" + str(time_step) + ".png", dpi=300)
    # close fig
    fig.clf()
    plt.close()
