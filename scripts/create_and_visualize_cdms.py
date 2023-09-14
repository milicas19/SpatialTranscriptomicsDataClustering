"""
Creating and visualizing CDM matrices
"""
import argparse
from datetime import datetime

from data_util import read_data
from create_and_visualize_cdm import create_and_visualize_cdm

def create_and_visualize_cdms(data_names, num_of_pca_components, cdm_type, metric, is_h5ad_data, is_reduced_dict, closest_neighbors, is_max_norm, dorsal_data, colors, modification):
    for i, data_name in enumerate(data_names):
        start_time = datetime.now()
        print(f"=== {i+1}. Creating and visualizing CDM for {data_name}")
        
        if dorsal_data:
            for j in range(5):
                st = f"_{j+1}"
                cell_data = read_data(data_name, is_h5ad_data, st)
                create_and_visualize_cdm(cell_data, f"{data_name}{st}", num_of_pca_components[j], cdm_type, metric, is_h5ad_data, is_reduced_dict, closest_neighbors, is_max_norm, colors[j], modification)
        else:
            cell_data = read_data(data_name, is_h5ad_data, "")
            create_and_visualize_cdm(cell_data, data_name, num_of_pca_components[i], cdm_type, metric, is_h5ad_data, is_reduced_dict, closest_neighbors, is_max_norm, colors[i], modification)
        
        print(f"=== job completed in: {datetime.now() - start_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparing CDM values for different sets of data')
    parser.add_argument('-dn', '--data_names', type=str, nargs='+', required=True,
                        help='cell data name on which CDM will be computed and visualized')
    parser.add_argument('-pca', '--num_of_pca_components', nargs='+', type=int, 
                        help='number of PCA gene components')
    parser.add_argument('-cdm', '--cdm_type', default='value', type=str, choices=['value', 'cell', 'same_cell'], 
                        help='metric used to create CDM matrix')
    parser.add_argument('-m', '--metric', default='euclidean', type=str, choices=['euclidean', 'manhattan', 'minkowski', 'cosine'],
                        help='metric used to create CDM matrix')
    parser.add_argument('-h5ad', '--is_h5ad_data', action='store_true',
                        help='is data Ann data')
    
    parser.add_argument('-dict', '--is_reduced_dict', action='store_true',
                        help='are we using reduced adjacency dict instead of adjacency matrix')
    parser.add_argument('-closest', '--closest_neighbors', default=30, type=int, choices=range(50),
                        help='closest neighbors for nodes in reduced adjacency dict')
    parser.add_argument('-norm', '--is_max_normalization', action='store_true',
                        help='using max normalization, otherwise cdf')
    
    parser.add_argument('-dorsal', '--dorsal_data', action='store_true',
                        help='for dorsal midbrain')
    
    parser.add_argument('-col', '--color', default=['blue'], type=str, nargs='+', 
                        help='colors for cdm hist')
    
    parser.add_argument('-mod', '--modification', default='median', type=str,
                        help='modification for gene adjacency matrix')
    args = parser.parse_args()

    create_and_visualize_cdms(args.data_names, args.num_of_pca_components, args.cdm_type, args.metric, args.is_h5ad_data, args.is_reduced_dict, args.closest_neighbors, args.is_max_normalization, args.dorsal_data, args.color, args.modification)
