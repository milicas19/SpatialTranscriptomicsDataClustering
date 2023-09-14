"""
Clustering data using Leiden algorithm
"""
import argparse
from datetime import datetime
import numpy as np
from data_util import read_data
from graph_util import cluster_leiden, create_normalized_adjacency_matrices, read_or_save_reduced_graph, create_coord_dict, create_gene_dict

const = 10000
node_count = 1000
edge_count = 10000
        
def cluster(data_names, num_of_pca_components, is_h5ad_data, dorsal_data, alg, reduction):
    start_time = datetime.now()
    for i, data_name in enumerate(data_names):
        if dorsal_data:
            for j in range(5):
                print(f"=== {j+1}. Cluster data for {data_name}{j+1}.tsv")
                st = f"_{j+1}"
                cell_data = read_data(data_name, is_h5ad_data, st)
                cluster_data(data_names[0], cell_data, num_of_pca_components[j], is_h5ad_data, st, alg, reduction)
        else:
            print(f"=== {i+1}. Cluster data for {data_name}.tsv")
            cell_data = read_data(data_name, is_h5ad_data, "")
            cluster_data(data_name, cell_data, num_of_pca_components[i], is_h5ad_data, "", alg, reduction)        
    print(f"=== job completed in: {datetime.now() - start_time}")

def cluster_data(data_name, cell_data, num_of_pca_components, is_h5ad_data, st, alg, reduction):
    start_time = datetime.now()
    
    closest_neighbors_options = [5, 10, 15, 20, 25, 30]
    closest_neighbors_options_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        
    data_name = f"{data_name}{st}"
    cells, coord_dict = create_coord_dict(cell_data, is_h5ad_data)
    reduced_gene_dict = create_gene_dict(cell_data, num_of_pca_components, cells, is_h5ad_data)
    for num_of_closest_coord_neighbors in closest_neighbors_options_2:
        for num_of_closest_gene_neighbors in closest_neighbors_options:
            if num_of_closest_coord_neighbors == 0 and num_of_closest_gene_neighbors == 0:
                continue
            print("******************************************************************")
            print(f"====== {num_of_closest_coord_neighbors} coord closest neighbors AND {num_of_closest_gene_neighbors} gene closest neighbors")
                
            coord_adj_matrix_name = f"coord_{data_name}_adj_matrix"
            gene_adj_matrix_name = f"gene_{data_name}_adj_matrix_pca_{num_of_pca_components}"
                
            norm_adj_matrix_1, norm_adj_matrix_2 = create_normalized_adjacency_matrices(coord_adj_matrix_name, gene_adj_matrix_name, reduction, coord_dict, cells, reduced_gene_dict)
            print(f"MAX reduced = {np.max(norm_adj_matrix_1)}")
            print(f"MIN reduced = {np.min(norm_adj_matrix_1)}")
            print(f"MAX reduced = {np.max(norm_adj_matrix_2)}")
            print(f"MIN reduced = {np.min(norm_adj_matrix_2)}")
            gname = f"{data_name}_pca_{num_of_pca_components}_neighbors"
                
            reduced_graph_name_1 = f"graphs/coord_reduced_graph_{gname}_{num_of_closest_coord_neighbors}.graphml"
            reduced_graph_1 = read_or_save_reduced_graph(reduced_graph_name_1, norm_adj_matrix_1, num_of_closest_coord_neighbors)
                
            reduced_graph_name_2 = f"graphs/gene_reduced_graph_{gname}_{num_of_closest_gene_neighbors}.graphml"
            reduced_graph_2 = read_or_save_reduced_graph(reduced_graph_name_2, norm_adj_matrix_2, num_of_closest_gene_neighbors)

            if alg == 'leiden':
                cluster_leiden(reduced_graph_1, reduced_graph_2, cell_data, data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors,is_h5ad_data)
            
            print(f"=== job completed in: {datetime.now() - start_time}")
            print("******************************************************************")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparing CDM values for different sets of data')
    parser.add_argument('-dn', '--data_names', type=str, nargs='+', required=True,
                        help='data sets on which CDM will be computed and visualized')
    parser.add_argument('-pca', '--num_of_pca_components', nargs='+', type=int, 
                        help='number of PCA gene components')
    parser.add_argument('-h5ad', '--is_h5ad_data', action='store_true', 
                        help='is data Ann data')
    parser.add_argument('-dorsal', '--dorsal_data', action='store_true',
                        help='for dorsal midbrain')
    parser.add_argument('-alg', '--clustering_alg', default='leiden', type=str,
                        help='clustering algorithm')
    parser.add_argument('-red', '--reduction', action='store_true', 
                        help='is there reduction of adj matrices')
    args = parser.parse_args()
    
    cluster(args.data_names, args.num_of_pca_components, args.is_h5ad_data, args.dorsal_data, args.clustering_alg, args.reduction)
