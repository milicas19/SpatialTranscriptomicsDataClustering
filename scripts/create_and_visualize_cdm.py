import pandas as pd
import copy
import itertools
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import re
import os

import scipy.spatial.distance

from graph_util import create_normalized_adjacency_arrays, create_reduced_adjacency_dicts, create_coord_dict, create_gene_dict

CONST = 1000

def plot_cdm_hist(cdm_arr, cell_data_name, num_of_pca_components, ylabel, diff, color, metric=None, bins=1000):
    figure, ax = plt.subplots(nrows=1, ncols=1)
    figure.dpi = 100
    figure.set_figheight(11)
    figure.set_figwidth(17)

    _ = ax.hist(cdm_arr, bins=bins, color=color)
    if metric is None:
        ax.set_title("Histogram of dissimilarity", fontsize=30)
    else:
        ax.set_title(f"Histogram of dissimilarity using {metric} metric")
    ax.set_xlabel("Normalized dissimilarity", fontsize=30)
    ax.set_ylabel(ylabel, fontsize=28)
    
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)

    print(f"Saving CDM({diff}) distribution for {cell_data_name} with {num_of_pca_components} gene components . . .")
    figure.savefig(f"CDM_{diff}_for_{cell_data_name}_pca{num_of_pca_components}_{metric}.png")
    
def calc_dissimilarity(vector_1, vector_2, metric):
    if metric == "euclidean":
        return scipy.spatial.distance.euclidean(vector_1, vector_2)
    elif metric == "manhattan":
        return scipy.spatial.distance.cityblock(vector_1, vector_2)   
    elif metric == "minkowski":
        return scipy.spatial.distance.minkowski(vector_1, vector_2, p=3)
    elif metric == "cosine":
        return scipy.spatial.distance.cosine(vector_1, vector_2)
    
def cdm_value_diff(norm_adj_matrix_1, norm_adj_matrix_2, cell_data_name, num_of_pca_components, is_reduced_dict, cdm_matrix_name, color):
    if is_reduced_dict:
        upper_values = []
        for cell_pair, coord_dist in norm_adj_matrix_1.items():
            if cell_pair not in norm_adj_matrix_2:
                c1, c2 = cell_pair[0], cell_pair[1]
                gene_dist = norm_adj_matrix_2[(c2,c1)]
            else:
                gene_dist = norm_adj_matrix_2[cell_pair]
            upper_values.append(coord_dist - gene_dist)
    else:
        CDM = norm_adj_matrix_1 - norm_adj_matrix_2
        print(f"CDM matrix {cdm_matrix_name} created. Saving it . . .")
        if not os.path.exists('cdm_matrices'):
            os.makedirs('cdm_matrices')
        np.save(f"cdm_matrices/{cdm_matrix_name}", CDM)
        upper_values = CDM
        
    plot_cdm_hist(upper_values, f"{cell_data_name}_dict_{is_reduced_dict}", num_of_pca_components, "Number of cdm values with specified dissimilarity", "value difference", color)

# need to add functionality to cdm_cell_diff & cdm_same_cell_diff
def create_and_visualize_cdm(cell_data, cell_data_name, num_of_pca_components, cdm_type, metric, is_h5ad_data, is_reduced_dict, closest_neighbors, is_max_norm, color, modification):
    start_time = datetime.now()
    print(f"====== Creating and visualizing CDM for {cell_data_name}.tsv")

    cdm_name = f"cdm_{cell_data_name}_pca_{num_of_pca_components}_{cdm_type}_{metric}"
    coord_adj_matrix_name = f"coord_{cell_data_name}_adj_matrix"
    gene_adj_matrix_name = f"gene_{cell_data_name}_adj_matrix_pca_{num_of_pca_components}"
    coord_adj_dict_name = f"coord_{cell_data_name}_adj_dict"
    gene_adj_dict_name = f"gene_{cell_data_name}_adj_dict_pca_{num_of_pca_components}"
    
    cells, coord_dict = create_coord_dict(cell_data, is_h5ad_data)
    reduced_gene_dict = create_gene_dict(cell_data, num_of_pca_components, cells, is_h5ad_data)
    
    if is_reduced_dict:
        norm_adj_1, norm_adj_2 = create_reduced_adjacency_dicts(cell_data, num_of_pca_components, is_h5ad_data, closest_neighbors, coord_adj_dict_name, gene_adj_dict_name)
    else:
        norm_adj_1, norm_adj_2 = create_normalized_adjacency_arrays(coord_adj_matrix_name, gene_adj_matrix_name, is_max_norm, coord_dict, cells, reduced_gene_dict, modification)
    
    if cdm_type == "value":
        cdm_value_diff(norm_adj_1, norm_adj_2, cell_data_name, num_of_pca_components, is_reduced_dict, cdm_name, color)

    print(f"###### job completed in: {datetime.now() - start_time}")
    