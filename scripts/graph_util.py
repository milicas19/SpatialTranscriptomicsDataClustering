from ast import List
from matplotlib.axes import Axes
import pandas as pd
import copy
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as Patch
from matplotlib import cm
import re
import heapq
import os
import pickle
import scanpy
import heapq

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

import scipy.spatial.distance as distance_

import igraph as ig
import leidenalg

import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
# constants
CONST = 10000
MIN_CONST = 1000
NODE_COUNT = 1000
EDGE_COUNT = 10000
EPS = 0.001

PR = 5

######### HELPER FUNCTIONS FOR CREATING ADJACENCY MATRICES #########

#### create and save ADJACENCY MATRIX
def adjacency_matrix(data_dict, cells, adj_matrix_name):
    if os.path.exists(f"matrices/{adj_matrix_name}.npy"):
        print(f"====== Adjacency matrix {adj_matrix_name} already exists. Fetching it.")
        return np.load(f"matrices/{adj_matrix_name}.npy")
    else:
        start_time = datetime.now()
        print(f"====== Creating adjacency matrix {adj_matrix_name} . . .")
        max_element = 0
        n = len(cells)
        adjacency_matrix = np.zeros(shape=(n, n))

        for i in range(0, n - 1):
            if i != 0 and i % CONST == 0:
                print(f"Progress for calculating adjacency matrix: {i}/{n}")
            for j in range(i + 1, n):
                dist = distance_.euclidean(data_dict[cells[i]], data_dict[cells[j]])
                adjacency_matrix[i][j] = dist
                adjacency_matrix[j][i] = dist
                if dist > max_element:
                    max_element = dist
        
        print("Adjacency matrix created. Saving it . . .")
        if not os.path.exists('matrices'):
            os.makedirs('matrices')
        np.save(f"matrices/{adj_matrix_name}.npy", adjacency_matrix)\
        
        print(f"###### job completed in: {datetime.now() - start_time}")

        return adjacency_matrix
    
#### NORMALIZATION of matrix and array

def max_reduce_and_normalize_matrices(matrix1, matrix2):
    start_time = datetime.now()
    print(f"====== Reducing and normalizing adjacency matrix . . .")
    
    upper_indices = np.triu_indices(matrix2.shape[0], k=1)
    data_array = matrix2[upper_indices]
    std = np.std(data_array)
    mean = np.mean(data_array)
    up = mean + 3 * std
    down = mean - 3 * std
    indicesD = np.where(matrix2 < down)
    indicesU = np.where(matrix2 > up)
    
        
    matrix1[indicesD] = -1
    matrix1[indicesU] = -1
    max1= np.max(matrix1)
    matrix1[indicesD] = 10 * max1
    matrix1[indicesU] = 10 * max1
    min1 = np.min(matrix1)
    matrix1 = (matrix1 - min1)/(max1 - min1)
        
    nodes1=0
    for arr in matrix1:
        if np.sum(arr <= 1) < 30:
            nodes1 += 1
    print(f"Nodes num gene -> {nodes1}")
    
    matrix2[indicesD] = -1
    matrix2[indicesU] = -1
    max2= np.max(matrix2)
    matrix2[indicesD] = 10 * max2
    matrix2[indicesU] = 10 * max2
    min2 = np.min(matrix2)
    matrix2 = (matrix2 - min2)/(max2 - min2)
    
    nodes2=0
    for arr in matrix2:
        if np.sum(arr <= 1) < 30:
            nodes2 += 1
    print(f"Nodes num coord -> {nodes2}")
        
    print(f"###### job completed in: {datetime.now() - start_time}")
    return matrix1, matrix2

def max_normalize_matrix(matrix, matrix_name):
    if os.path.exists(f"matrices/max_normalized_{matrix_name}.npy"):
        print(f"====== Normalized adjacency matrix max_normalized_{matrix_name} already exists. Fetching it.")
        return np.load(f"matrices/max_normalized_{matrix_name}.npy")
    else:
        start_time = datetime.now()
        print(f"====== Normalizing adjacency matrix {matrix_name} . . .")
        
        max_element = np.max(matrix)
        min_element = np.min(matrix)
        matrix = (matrix - min_element)/(max_element - min_element)
        
        print("Max normalized adjacency matrix created. Saving it . . .")
        if not os.path.exists('matrices'):
            os.makedirs('matrices')
        np.save(f"matrices/max_normalized_{matrix_name}.npy", matrix)

        print(f"###### job completed in: {datetime.now() - start_time}")
        return matrix

def cdf_normalize_matrix(matrix, matrix_name):
    if os.path.exists(f"matrices/normalized_{matrix_name}.npy"):
        print(f"====== Normalized adjacency matrix normalized_{matrix_name} already exists. Fetching it.")
        return np.load(f"matrices/normalized_{matrix_name}.npy")
    else:
        start_time = datetime.now()
        print(f"====== Normalizing adjacency matrix {matrix_name} . . .")

        flattened_matrix = matrix.flatten()
        sorted_values = np.sort(flattened_matrix)
        cdf = np.cumsum(sorted_values) / np.sum(sorted_values)
        normalized_cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
        matrix_interp = np.interp(flattened_matrix, sorted_values, normalized_cdf)
        normalized_matrix = matrix_interp.reshape(matrix.shape)
        
        print("Normalized adjacency matrix created. Saving it . . .")
        if not os.path.exists('matrices'):
            os.makedirs('matrices')
        np.save(f"matrices/normalized_{matrix_name}.npy", normalized_matrix)
        
        print(f"###### job completed in: {datetime.now() - start_time}")
        return normalized_matrix

def max_normalize_array(adjacency_array, adj_matrix_name):
    if os.path.exists(f"arrays/max_normalized_{adj_matrix_name}.npy"):
        print(f"====== Normalized adjacency matrix max_normalized_{adj_matrix_name} already exists. Fetching it.")
        return np.load(f"arrays/max_normalized_{adj_matrix_name}.npy")
    else:
        start_time = datetime.now()
        print(f"====== Normalizing adjacency matrix {adj_matrix_name} . . .")
        max_element = np.max(adjacency_array)
        adjacency_array /= max_element
        
        print("Max normalized adjacency matrix created. Saving it . . .")
        if not os.path.exists('arrays'):
            os.makedirs('arrays')
        np.save(f"arrays/max_normalized_{adj_matrix_name}.npy", adjacency_array)

        print(f"###### job completed in: {datetime.now() - start_time}")
        return adjacency_array

def cdf_normalize_array(adjacency_array, adj_matrix_name):
    if os.path.exists(f"arrays/normalized_{adj_matrix_name}.npy"):
        print(f"====== Normalized adjacency matrix normalized_{adj_matrix_name} already exists. Fetching it.")
        return np.load(f"arrays/normalized_{adj_matrix_name}.npy")
    else:
        start_time = datetime.now()
        print(f"====== Normalizing adjacency matrix {adj_matrix_name} . . .")

        sorted_values = np.sort(adjacency_array)
        cdf = np.cumsum(sorted_values) / np.sum(sorted_values)
        normalized_cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
        normalized_array = np.interp(adjacency_array, sorted_values, normalized_cdf)
        
        print("Normalized adjacency matrix created. Saving it . . .")
        if not os.path.exists('arrays'):
            os.makedirs('arrays')
        np.save(f"arrays/normalized_{adj_matrix_name}.npy", normalized_array)
        
        print(f"###### job completed in: {datetime.now() - start_time}")
        return normalized_array

def eq_normalize_array(adjacency_array, adj_matrix_name):
    if os.path.exists(f"arrays/eq_normalized_{adj_matrix_name}.npy"):
        print(f"====== Normalized adjacency matrix normalized_{adj_matrix_name} already exists. Fetching it.")
        return np.load(f"arrays/eq_normalized_{adj_matrix_name}.npy")
    else:
        start_time = datetime.now()
        print(f"====== Normalizing adjacency matrix {adj_matrix_name} . . .")

        occurrences = []
        occ = 1
        sorted_values = np.sort(adjacency_array)
        n = len(sorted_values)
        print(f"sorted_values_len={n}")
        for i in range(n - 1):
            if abs(sorted_values[i + 1] - sorted_values[i]) < EPS:
                occ += 1
            else:
                occurrences.append(occ)
                occ = 1
        occurrences.append(occ)
        
        cdf = np.cumsum(occurrences) / np.sum(occurrences)
        print(f"adj_arr={len(adjacency_array)}, sum_occ={np.sum(occurrences)}")
        normalized_cdf = cdf
        #(cdf - cdf.min()) / (n - cdf.min())
        array = []
        for i, occ in enumerate(occurrences):
            arr = [normalized_cdf[i] for _ in range(occ)]
            if len(arr) != occ:
                print(f"no {i}")
            array += arr
        print(f"adj_arr={len(adjacency_array)}, arr={len(array)}")
        normalized_array = np.interp(adjacency_array, sorted_values, array)

        print("Normalized adjacency matrix created. Saving it . . .")
        if not os.path.exists('arrays'):
            os.makedirs('arrays')
        np.save(f"arrays/eq_normalized_{adj_matrix_name}.npy", normalized_array)

        print(f"###### job completed in: {datetime.now() - start_time}")
        return normalized_array

#### creating coord and gene DICTS
def reduce_gene_expr_components(cell_data, num_of_pca_components):
    gene_data = cell_data.copy()
    if 'cellID' in gene_data.columns:
        gene_data.drop('cellID', axis=1, inplace=True)
    if 'x' in gene_data.columns and 'y' in gene_data.columns:
        gene_data.drop('x', axis=1, inplace=True)
        gene_data.drop('y', axis=1, inplace=True)

    scaling = StandardScaler()
    scaled_gene_data = scaling.fit_transform(gene_data)

    pca = PCA(n_components=num_of_pca_components)
    reduced_gene_data = pca.fit_transform(scaled_gene_data)
    explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
    print(f"explained_variance_ratio={explained_variance_ratio[-1]}")

    return reduced_gene_data

def create_coord_dict(cell_data, is_h5ad_data):
    start_time = datetime.now()
    print("====== Creating coordinate dictionary and list of cell ids. . .")
    print(f"h5ad={is_h5ad_data}")
    coord_dict = {}
    cells = []
    if not is_h5ad_data:
        for _, row in cell_data.iterrows():
            coord_dict[row['cellID']] = [row['x'], row['y']]
            cells.append(row['cellID'])
    else:
        coord_data, _ , _= cell_data
        for i, coord_list in enumerate(coord_data):
            coord_dict[i] = [coord_list[0], coord_list[1]]
            cells.append(i)

    print(f"###### job completed in: {datetime.now() - start_time}")

    return cells, coord_dict        

def create_gene_dict(cell_data, num_of_pca_components, cells, is_h5ad_data):
    start_time = datetime.now()    
    print("====== Creating gene expression dictionary . . .")
    reduced_gene_expr_dict = {}
    if not is_h5ad_data:
        reduced_gene_data = reduce_gene_expr_components(cell_data, num_of_pca_components)    
    else:
        _, gene_df, _ = cell_data
        reduced_gene_data = reduce_gene_expr_components(gene_df, num_of_pca_components) 
    
    for i in range(0, len(reduced_gene_data)):
        reduced_gene_expr_dict[cells[i]] = reduced_gene_data[i]
                
    print(f"###### job completed in: {datetime.now() - start_time}")
    return reduced_gene_expr_dict

#### turn adjacency matrix into array (move out all values higher then x * median_value)
def process_adjacency_matrices_median(adjacency_matrix_1, adjacency_matrix_2):
    num_higher_then_3m_coord = 0
    num_higher_then_3m_gene = 0
    start_time = datetime.now()    
    print("====== Processing adjacency matrices -> removing values higher than 3 * median_value . . .")
    adjacency_array_1 = []
    adjacency_array_2 = []
    n = len(adjacency_matrix_1)
    median_1 = np.median(adjacency_matrix_1.flatten())
    median_2 = np.median(adjacency_matrix_2.flatten())
    for i in range(n - 1):
        if i % MIN_CONST == 0 and i != 0:
                print(f"Processing adjacency matrices: {i}/{n}")
        for j in range(i + 1, n):
            m1 = 100000 * median_1
            m2 = 100000 * median_2
            if adjacency_matrix_1[i][j] > m1:
                adjacency_array_1.append(m1)
                num_higher_then_3m_coord += 1
            else:
                adjacency_array_1.append(adjacency_matrix_1[i][j]) 
            if adjacency_matrix_2[i][j] > m2:    
                adjacency_array_2.append(m2)
                num_higher_then_3m_gene += 1
            else:
                adjacency_array_2.append(adjacency_matrix_2[i][j])
    
    print(f"Number of values higher then 3M in coord dict -> {num_higher_then_3m_coord}")
    print(f"Number of values higher then 3M in gene dict -> {num_higher_then_3m_gene}")
    
    print(f"###### job completed in: {datetime.now() - start_time}")
    return np.array(adjacency_array_1), np.array(adjacency_array_2)

def process_adjacency_matrix_stddev(adjacency_matrix_1, adjacency_matrix_2):
    num_lower_then_3s = 0
    start_time = datetime.now()    
    print("====== Processing adjacency matrices -> removing cell pairs that have values higher than mean + 3 * std_dev in gene matrix . . .")
    adjacency_array_1 = []
    adjacency_array_2 = []
    n = len(adjacency_matrix_1)
    upper_indices = np.triu_indices(adjacency_matrix_2.shape[0], k=1)
    data_array = adjacency_matrix_2[upper_indices]
    m = len(data_array)
    std = np.std(data_array)
    mean = np.mean(data_array)
    # Adjust k for k*std - this should be added as parameter in script
    up = mean + 3 * std
    down = mean - 3 * std
    down_num = 0
    print(f"mean={mean} & std={std} -> val={up}")
    
    for i in range(n - 1):
        if i % MIN_CONST == 0 and i != 0:
                print(f"Processing adjacency matrices: {i}/{n}")
        for j in range(i + 1, n):
            v1 = adjacency_matrix_1[i][j]
            v2 = adjacency_matrix_2[i][j]
            if v2 <= up and v2 >= down:
                adjacency_array_1.append(v1)
                adjacency_array_2.append(v2)
                num_lower_then_3s += 1
            if v2 < down:
                down_num += 1
                
    print(f"Number of values lower then 3S -> {num_lower_then_3s}/{m} ({int(num_lower_then_3s/m * 100) })")
    print(f"V < MEAN - 3STD -> {down_num}")
    
    print(f"###### job completed in: {datetime.now() - start_time}")
    return np.array(adjacency_array_1), np.array(adjacency_array_2)

def process_adjacency_matrix_percentage(adjacency_matrix_1, adjacency_matrix_2):
    start_time = datetime.now()    
    print(f"====== Processing adjacency matrices -> removing {100-PR}% of highest values . . .")
    adjacency_array_1 = []
    adjacency_array_2 = []
    
    upper_indices = np.triu_indices(adjacency_matrix_2.shape[0], k=1)
    arr = adjacency_matrix_2[upper_indices]
    arr.sort()
    n = len(adjacency_matrix_1)
    print(f"n = {n}")
    m = len(arr)
    # Adjust PR at the beginning - this should be added as parameter in script
    last_index = int(m * PR / 100)
    last_element = arr[last_index]
    print(f"num_of_elements={m} -> last_index={last_index} & last_element={last_element} -> ({last_index/m}%)")
    
    for i in range(n - 1):
        if i % MIN_CONST == 0 and i != 0:
                print(f"Processing adjacency matrices: {i}/{n}")
        for j in range(i + 1, n):
            el1 = adjacency_matrix_1[i][j]
            el2 = adjacency_matrix_2[i][j]
            if el2 <= last_element:
                adjacency_array_1.append(el1)
                adjacency_array_2.append(el2)
            
    print(f"###### job completed in: {datetime.now() - start_time}")
    print(f"arr_size = {len(adjacency_array_2)}")
    return adjacency_array_1, adjacency_array_2

######### END HELPER FUNCTIONS FOR CREATING ADJACENCY MATRICES #########

#--------------------- CREATING (NORMALIZED) ADJACENCY MATRICES ---------------------

def create_adjacency_matrices(coord_adj_matrix_name, gene_adj_matrix_name, coord_dict, cells, reduced_gene_dict):

    adjacency_matrix_1 = adjacency_matrix(coord_dict, cells, coord_adj_matrix_name)
    adjacency_matrix_2 = adjacency_matrix(reduced_gene_dict, cells, gene_adj_matrix_name)

    return adjacency_matrix_1, adjacency_matrix_2

def create_normalized_adjacency_arrays(coord_adj_matrix_name, gene_adj_matrix_name, is_max_normalization, coord_dict, cells, reduced_gene_dict, modification):    
    adjacency_matrix_1, adjacency_matrix_2 = create_adjacency_matrices(coord_adj_matrix_name, gene_adj_matrix_name, coord_dict, cells, reduced_gene_dict)
    
    if modification == 'median':
        adjacency_array_1, adjacency_array_2 = process_adjacency_matrices_median(adjacency_matrix_1, adjacency_matrix_2)
    elif modification == 'percentage':
        adjacency_array_1, adjacency_array_2 = process_adjacency_matrix_percentage(adjacency_matrix_1, adjacency_matrix_2)
    elif modification == 'stddev':
        adjacency_array_1, adjacency_array_2 = process_adjacency_matrix_stddev(adjacency_matrix_1, adjacency_matrix_2)
    else:
        upper_indices_1 = np.triu_indices(adjacency_matrix_1.shape[0], k=1)
        adjacency_array_1 = adjacency_matrix_1[upper_indices_1]
        upper_indices_2 = np.triu_indices(adjacency_matrix_2.shape[0], k=1)
        adjacency_array_2 = adjacency_matrix_2[upper_indices_2]
    
    if is_max_normalization:
        adj1 = max_normalize_array(adjacency_array_1, coord_adj_matrix_name)
        adj2 = max_normalize_array(adjacency_array_2, gene_adj_matrix_name)
    else:
        adj1 = eq_normalize_array(adjacency_array_1, coord_adj_matrix_name)
        adj2 = eq_normalize_array(adjacency_array_2, gene_adj_matrix_name)
        
    return adj1, adj2

def create_normalized_adjacency_matrices(coord_adj_matrix_name, gene_adj_matrix_name, reduction, coord_dict, cells, reduced_gene_dict):    
    adjacency_matrix_1, adjacency_matrix_2 = create_adjacency_matrices(coord_adj_matrix_name, gene_adj_matrix_name, coord_dict, cells, reduced_gene_dict)

    if reduction:
        norm_1, norm_2 = max_reduce_and_normalize_matrices(adjacency_matrix_1, adjacency_matrix_2)
    else:
        norm_1 = max_normalize_matrix(adjacency_matrix_1, coord_adj_matrix_name)
        norm_2 = max_normalize_matrix(adjacency_matrix_2, gene_adj_matrix_name)
    
    return norm_1, norm_2
    
######### HELPER FUNCTIONS FOR CREATING ADJACENCY DICTS #########

def add_elements_from_dict_to(from_dict, to_dict, data_dict):
    for cell_pair, _ in from_dict.items():
        print(cell_pair)
        c1, c2 = cell_pair[0], cell_pair[1]
        if cell_pair not in to_dict and (c2, c1) not in to_dict:
            to_dict[cell_pair] = distance_.euclidean(data_dict[c1], data_dict[c2])
    return to_dict

def find_smallest_dist(pairs, num_elements):
    return heapq.nsmallest(num_elements, pairs, key=lambda x: x[1])

def reduced_adjacency_dict(data_dict, cells, closest_neighbors, adj_dict_name):
    adj_dict= f"dicts/{adj_dict_name}.npy"
    if not os.path.exists(adj_dict):
        return create_and_save_adjacency_dict(cells, data_dict, closest_neighbors, adj_dict_name)
    print(f"====== Adjacency dict {adj_dict_name} already exists. Fetching it.")
    with open(adj_dict, 'rb') as f:
        return pickle.load(f)

def create_and_save_adjacency_dict(cells, data_dict, closest_neighbors, adj_dict_name):
    start_time = datetime.now()
    print("====== Creating adjacency dict . . .")

    n = len(cells)
    reduced_adjacency_dict = {}
    pairs = []
    for i in range(n):
        if i != 0 and i % MIN_CONST == 0:
                print(f"Progress for reducing dict: {i}/{n}")
        for j in range(n):
            if i == j:
                continue
            dist = distance_.euclidean(data_dict[cells[i]], data_dict[cells[j]])
            pairs.append((cells[j], dist))
        smallest_pairs = find_smallest_dist(pairs, closest_neighbors)
        for pair in smallest_pairs:
            cellId, distance = pair[0], pair[1]
            cell_pair = (cells[i], cellId)
            cell_pair_reverse = (cellId, cells[i])
            if cell_pair_reverse not in reduced_adjacency_dict:
                reduced_adjacency_dict[cell_pair] = distance

    print("Adjacency dict created. Saving it . . .")
    with open(f"dicts/{adj_dict_name}.npy", 'wb') as f:
        pickle.dump(reduced_adjacency_dict, f)

    print(f"###### job completed in: {datetime.now() - start_time}")

    return reduced_adjacency_dict

######### END HELPER FUNCTIONS FOR CREATING ADJACENCY DICTS #########

#--------------------- CREATING (NORMALIZED )ADJACENCY DICTS ---------------------

def create_reduced_adjacency_dicts(cell_data, num_of_pca_components, is_h5ad_data, closest_neighbors, coord_adj_dict_name, gene_adj_dict_name):
    start_time = datetime.now()
    print("====== Creating adjacency dicts . . .")
    
    cells, coord_dict = create_coord_dict(cell_data, is_h5ad_data)
    reduced_gene_dict = create_gene_dict(cell_data, num_of_pca_components, cells, is_h5ad_data)
    
    adjacency_dict_1 = reduced_adjacency_dict(coord_dict, cells, closest_neighbors, coord_adj_dict_name)
    adjacency_dict_2 = reduced_adjacency_dict(reduced_gene_dict, cells, closest_neighbors, gene_adj_dict_name)
    
    add_elements_from_dict_to(adjacency_dict_2, adjacency_dict_1, coord_dict, cells)
    add_elements_from_dict_to(adjacency_dict_1, adjacency_dict_2, reduced_gene_dict, cells)

    print(f"###### job completed in: {datetime.now() - start_time}")
    
    return adjacency_dict_1, adjacency_dict_2


#--------------------- CLUSTERING FUNCTIONS ---------------------

# REDUCING GRAPH 
def reduce_graph(adjacency_matrix, N):
    start_time = datetime.now()
    print(F"====== Reducing graph to {N} closest neighbors. . .")

    if N == 0:
        reduced_G = None
    else:
        G = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist(), mode='UNDIRECTED')
        reduced_G = ig.Graph()
        nodes_num = G.vcount()
        reduced_G.add_vertices(nodes_num)

        for i, node in enumerate(G.vs):
            if i != 0 and i % NODE_COUNT == 0:
                print(f"Progress for reducing graph: {i}/{nodes_num}")
            neighbors = sorted(G.neighbors(node, mode="out"), key=lambda n: G.es[G.get_eid(node.index, n)]['weight'])[:N]
            for neighbor in neighbors:
                weight = G.es[G.get_eid(node.index, neighbor)]['weight']
                if weight <= 1:
                    reduced_G.add_edge(node.index, neighbor, weight=weight)

    print(f"###### job completed in: {datetime.now() - start_time}")

    return reduced_G

def reduce_matrix(A, k):
    print(f"MAX norm = {np.max(A)}")
    print(f"MIN norm = {np.min(A)}")
    print(F"====== Reducing matrix to {k} closest neighbors. . .")
    if k == 0:
        return None
    
    for i in range(A.shape[0]):
        kth_smallest_value = np.partition(A[i], k-1)[k-1]
        A[i, A[i] > kth_smallest_value] = 2
    print(f"MAX reduced = {np.max(A)}")
    print(f"MIN reduced = {np.min(A)}")  
    return A

def make_matrix_union(A1, A2):
    n = len(A1)
    A = np.full((n, n), 2)
    for i in range(n):
        for j in range(n):
            if A1[i][j] == 2 and A2[i][j] != 2:
                A[i][j] = A2[i][j]
            elif A1[i][j] != 2 and A2[i][j] == 2:
                A[i][j] = A1[i][j]
            elif A1[i][j] != 2 and A2[i][j] != 2:
                A[i][j] = (A1[i][j] + A2[i][j])/2
    return A

def read_or_save_reduced_graph(reduced_graph_name, norm_adj_matrix, num_of_closest_neighbors):
    if os.path.exists(reduced_graph_name):
            print(f"Reduced graph for {reduced_graph_name} exits. Fetching it . . .")
            reduced_graph = ig.Graph.Read_GraphML(reduced_graph_name)
    else:
        reduced_graph = reduce_graph(norm_adj_matrix, num_of_closest_neighbors)
        if reduced_graph is not None:
            print(f"Saving reduced graph for {reduced_graph_name} . . .")
            if not os.path.exists('graphs'):
                os.makedirs('graphs')
            reduced_graph.write_graphml(reduced_graph_name)
    return reduced_graph

# PLOTTING CLUSTERS AND GRAPH

def plot_clusters(G, node_colors, cell_data, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, note_text, is_h5ad_data):
    cells, cell_dict = create_coord_dict(cell_data, is_h5ad_data)
    figure, ax = plt.subplots()

    for node in G.vs:
        (x, y) = cell_dict[cells[node.index]]
        color = node_colors[node.index]
        ax.scatter(x, y, color=color, s=10, alpha=0.7)

    ax.set_title(f"clusters -> {note_text}")
    ax.set_xlabel("cell x coordinate")
    ax.set_ylabel("cell y coordinate")

    if "midbrain" not in cell_data_name:
        ax.invert_yaxis()
    
    cell_data_name = re.sub(r'\.', '_', cell_data_name)
    
    print("Saving histogram of cell clustering with x and y coordinates of cells . . .")
    figure.savefig(f"coordinate_clusters_for_{cell_data_name}_pca_{num_of_pca_components}_coord_{num_of_closest_coord_neighbors}_gene{num_of_closest_gene_neighbors}")
    
def plot_clustered_graph(G, node_colors, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, note_text):
    figure, ax = plt.subplots()
    layout = G.layout_fruchterman_reingold(weights="weight")
    coords = np.array(layout)
    
    ax.scatter(coords[:, 0], coords[:, 1], s=10, color=node_colors, alpha=1)

    ax.set_title(f"graph clusters -> {note_text}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    if "midbrain" not in cell_data_name:
        ax.invert_yaxis()
    
    cell_data_name = re.sub(r'\.', '_', cell_data_name)
    
    print("Saving histogram of cell clustering for graph . . .")
    figure.savefig(f"graph_clusters_for_{cell_data_name}_pca_{num_of_pca_components}_coord_{num_of_closest_coord_neighbors}_gene{num_of_closest_gene_neighbors}")

# UNION OF TWO GRAPHS
def make_union(graph_1, graph_2):
    start_time = datetime.now()
    print("====== Making union . . .")
    
    G_union = ig.Graph(len(graph_1.vs), directed=False)

    num_edges = graph_1.ecount()
    for i, edge in enumerate(graph_1.es):
        if i != 0 and i % EDGE_COUNT == 0:
            print(f"Progress for first graph: {i}/{num_edges}")
        source, target = edge.tuple
        G_union.add_edge(source, target, weight=edge["weight"])

    num_edges = graph_2.ecount()
    for i, edge in enumerate(graph_2.es):
        if i != 0 and i % EDGE_COUNT == 0:
            print(f"Progress for second graph: {i}/{num_edges}")
        source, target = edge.tuple
        if G_union.are_connected(source, target):
            existing_edge = G_union.es[G_union.get_eid(source, target)]
            existing_edge["weight"] = (existing_edge["weight"] + edge["weight"]) / 2
        else:
            G_union.add_edge(source, target, weight=edge["weight"])
    
    print(f"###### job completed in: {datetime.now() - start_time}")
    return G_union

# LEIDEN CLUSTERING

def cluster_leiden(graph_1, graph_2, cell_data, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors,is_h5ad_data):
    start_time = datetime.now()
    print("====== Clustering data . . .")
    
    if graph_1 is None:
        G_union = graph_2
    elif graph_2 is None:
        G_union = graph_1
    else:
        G_union = make_union(graph_1, graph_2)

    partition = leidenalg.find_partition(graph=G_union, partition_type=leidenalg.ModularityVertexPartition, n_iterations=-1, seed=0)
    
    annotations = []
    if is_h5ad_data:
        _, _, annotations = cell_data
    
    print(f"Modularity: {partition.modularity:.3f}")
    print(f"Number of clusters: {len(set(partition.membership))}")

    color_map = cm.get_cmap('tab20c', len(set(partition.membership)))
    colors = [color_map(i) for i in range(len(set(partition.membership)))]
    
    
    node_colors = [colors[cluster] for cluster in  partition.membership]
    labels = [f"cluster{i}" for i in range(len(set(partition.membership)))]
    print(set(node_colors))

    note_text = f"modularity: {partition.modularity:.3f} number of clusters: {len(set(partition.membership))}"
    if len(annotations) != 0:
        ari_score = adjusted_rand_score(partition.membership, annotations)
        print(f"ARI score: {ari_score}")
        note_text = f"modularity: {partition.modularity:.3f} ari score: {ari_score:.3f} number of clusters: {len(set(partition.membership))}"
    
    plot_clusters(G_union, node_colors, cell_data, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, note_text, is_h5ad_data)
    plot_clustered_graph(G_union, node_colors, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, note_text)

    print(f"###### job completed in: {datetime.now() - start_time}")    
    return partition, G_union

def optimize_partition(partition, graph_1, graph_2, cell_data, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, is_h5ad_data):
    start_time = datetime.now()
    print("====== Optimizing partition . . .")

    optimiser = leidenalg.Optimiser()
    steps = -1
    diff = 1

    while diff > 0:
        diff = optimiser.optimise_partition(partition)
        steps += 1

    if steps == 0:
        print("This is already optimal partition")
    else:
        print(f"Number of steps: {str(steps)}")

        print(f"Optimal modularity: {partition.modularity:.3f}")
        print(f"Number of clusters: {len(set(partition.membership))}")

        color_map = cm.get_cmap('tab20c', len(set(partition.membership)))
        colors = [color_map(i) for i in range(len(set(partition.membership)))]

        node_colors = [colors[cluster] for cluster in  partition.membership]

        note_text = f"modularity: {partition.modularity:.3f} number of clusters: {len(set(partition.membership))}"
        G_union = make_union(graph_1, graph_2)
        plot_clusters(G_union, node_colors, cell_data, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, note_text, is_h5ad_data)
        plot_clustered_graph(G_union, node_colors, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, note_text) 
    
    print(f"###### job completed in: {datetime.now() - start_time}")
    
def cluster_report(partition, G, cell_data, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, is_h5ad_data):
    start_time = datetime.now()
    print("====== Clustering report . . .")
    
    print(f"Optimal modularity: {partition.modularity:.3f}")
    print(f"Number of clusters: {len(set(partition.membership))}")
    print(f"Optimal coord closest neighbors: {num_of_closest_coord_neighbors}")
    print(f"Optimal gene closest neighbors: {num_of_closest_gene_neighbors}")

    color_map = cm.get_cmap('tab20c', len(set(partition.membership)))
    colors = [color_map(i) for i in range(len(set(partition.membership)))]

    node_colors = [colors[cluster] for cluster in  partition.membership]

    note_text = f"modularity: {partition.modularity:.3f} number of clusters: {len(set(partition.membership))}"
    plot_clusters(G, node_colors, cell_data, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, note_text, is_h5ad_data)
    plot_clustered_graph(G, node_colors, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, note_text)

    print(f"###### job completed in: {datetime.now() - start_time}")
    
######################

def cluster_h(graph_1, graph_2, matrix1, matrix2, cell_data, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors,is_h5ad_data):
    start_time = datetime.now()
    print("====== Clustering data . . .")
    
    if matrix1 is None:
        distances_matrix = matrix2
    elif matrix2 is None:
        distances_matrix = matrix1
    else:
        distances_matrix = make_matrix_union(matrix1, matrix2)

    print(f"MAX distance matrix = {np.max(distances_matrix)}")
    print(f"MIN distance matrix = {np.min(distances_matrix)}")

    annotations = []
    if is_h5ad_data:
        _, _, annotations = cell_data
    
    linkage_matrix = linkage(distances_matrix, method='ward')
    # Assign clusters based on a distance threshold
    distance_threshold = 1  # Adjust this threshold as needed
    print(f"lower than {distance_threshold} -> {np.sum(distances_matrix < distance_threshold)}")
    clusters = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

    print(clusters)
    print(f"Number of clusters: {len(set(clusters))}")
    
    color_map = cm.get_cmap('tab20c', len(set(clusters)))
    colors = [color_map(i) for i in range(len(set(clusters)))]

    node_colors = [colors[cluster] for cluster in  clusters]

    if len(annotations) != 0:
        ari_score = adjusted_rand_score(clusters, annotations)
        print(f"ARI score: {ari_score}")
        note_text = f"ari score: {ari_score:.3f} number of clusters: {len(set(clusters))}"
    
    if graph_1 is None:
        G_union = graph_2
    elif graph_2 is None:
        G_union = graph_1
    else:
        G_union = make_union(graph_1, graph_2)
        
    plot_clusters(G_union, node_colors, cell_data, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, note_text, is_h5ad_data)
    plot_clustered_graph(G_union, node_colors, cell_data_name, num_of_pca_components, num_of_closest_coord_neighbors, num_of_closest_gene_neighbors, note_text)

    print(f"###### job completed in: {datetime.now() - start_time}")    
    return clusters, G_union
