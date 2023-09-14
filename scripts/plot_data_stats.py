"""
Plotting distributions for adjacency matrices for graph 1 (coordinate graph) and graph 2 (gene graphs) for data sets
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime
import argparse
from scipy.stats import kurtosis, skew
import scipy.stats as stats
import logging
import os

def plot_data(array, name, color, bins=1000):
    figure, ax = plt.subplots(nrows=1, ncols=1)
    figure.dpi = 100
    figure.set_figheight(11)
    figure.set_figwidth(17)

    _ = ax.hist(array, bins=bins, color=color)
    ax.set_title("Distribution", fontsize=30)
    ax.set_xlabel("Distance", fontsize=30)
    ax.set_ylabel("Number of cell pairs with specified distance", fontsize=28)
    
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)

    print(f"Saving distribution for {name} . . .")
    name = re.sub(r'\.', '', name)
    figure.savefig(f"distribution__for_{name}.png")
    plt.close()

def calc_stats(matrices_name, arrays_name):
    start_time = datetime.now()
    logging.info("====== Calculating means, medians, variances and standard deviations . . .")
    
    means = []
    medians = []
    variances = []
    stand_devs = []
    
    labels = []
    
    kurtosis_val = [] 
    skewness_val = []
    
    colors = ['#BEBADA', '#AAA3DE', '#9F95E8', '#9183F0', '#7F6FEC', '#EA96A7', '#F57C94', '#fb607f', '#9EE3D7', '#77DFCD', '#59DCC5']

    n = len(colors)
    for i, matrix_name in enumerate(matrices_name):
        data_matrix = np.load(f"matrices/{matrix_name}.npy")
        upper_indices = np.triu_indices(data_matrix.shape[0], k=1)
        data_array = data_matrix[upper_indices]
        i = i % n
        plot_data(data_array, matrix_name, colors[i])
        
        '''
        labels.append(matrix_name)
            
        mean = np.round(np.mean(data_array), 2)
        logging.info(f"Mean: {mean}")
        means.append(mean)
            
        median = round(np.median(data_array), 2)
        logging.info(f"Median: {median}")
        medians.append(median)
            
        variance = round(np.var(data_array), 2)
        logging.info(f"Variance: {variance}")
        variances.append(variance)
            
        stand_dev = round(np.std(data_array), 2)
        logging.info(f"Standard deviation: {stand_dev}")
        stand_devs.append(stand_dev)
            
        print(kurtosis(data_array))
        k = round(kurtosis(data_array), 2)
        logging.info(f"Kurtosis: {k}")
        kurtosis_val.append(k)
            
        s = round(skew(data_array), 2)
        logging.info(f"Skewness: {s}")
        skewness_val.append(s)
        '''
    for i, array_name in enumerate(arrays_name):
        data_array= np.load(f"arrays/{array_name}.npy")
        i = i % n
        plot_data(data_array, array_name, colors[i])
        
        '''
        labels.append(matrix_name)
            
        mean = np.round(np.mean(data_array), 2)
        logging.info(f"Mean: {mean}")
        means.append(mean)
            
        median = round(np.median(data_array), 2)
        logging.info(f"Median: {median}")
        medians.append(median)
            
        variance = round(np.var(data_array), 2)
        logging.info(f"Variance: {variance}")
        variances.append(variance)
            
        stand_dev = round(np.std(data_array), 2)
        logging.info(f"Standard deviation: {stand_dev}")
        stand_devs.append(stand_dev)
            
        print(kurtosis(data_array))
        k = round(kurtosis(data_array), 2)
        logging.info(f"Kurtosis: {k}")
        kurtosis_val.append(k)
            
        s = round(skew(data_array), 2)
        logging.info(f"Skewness: {s}")
        skewness_val.append(s)     
        '''  
    logging.info(f"###### job completed in: {datetime.now() - start_time}")
    
    return labels, means, medians, variances, stand_devs, kurtosis_val, skewness_val

def plot_stat(array, labels, name, data_name):
    start_time = datetime.now()
    logging.info(f"====== Plotting {name} from data sets . . .")
    
    colors = ['#8dd3c7', '#bebada', '#8dd3c7', '#bebada']

    color_map = dict(zip(labels, colors))
    data_name = re.sub(r'\.', '', data_name)
    name = re.sub(r'\.', '', name)
    '''
        if not os.path.exists(f"legend_{data_name}_compare_gene_coord_adj_{name}"):
        legend_fig = plt.figure()

        legend_ax = legend_fig.add_subplot(111)

        legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color_map[label]) for label in labels]
        legend_ax.legend(legend_patches, labels)

        legend_fig.savefig(f"legend_{data_name}_compare_gene_coord_adj_{name}")
        plt.close()
    '''
    main_fig = plt.figure()

    main_ax = main_fig.add_subplot(111)
    main_ax.bar(range(len(array)), array, color=[color_map[label] for label in labels])

    logging.info("Saving it ...")
    main_fig.suptitle(name, fontsize=12, fontweight='bold')
    main_fig.savefig(f"{data_name}_dataset_compare_coord_gene_adj_{name}")
    
    plt.close()
    
    logging.info(f"###### job completed in: {datetime.now() - start_time}")


def plot_stats(matrices_names, arrays_name):
    start_time = datetime.now()
    logging.info("====== Comparing gene and coord adjacency matrix . . .")
    logging.info(f"====== Comparing gene and coord adjacency matrix for {matrices_names}. . .")
    calc_stats(matrices_names, arrays_name)
    #labels, means, medians, variances, stand_devs, kurtosis_val, skewness_val = calc_stats(matrices_names, arrays_name)
    #stats = [means, medians, variances, stand_devs, kurtosis_val, skewness_val]
    #names = ['mean', 'median', 'variance', 'stand_dev', 'kurtosis', 'skewness']
        #for i, stat in enumerate(stats):
            #plot_stat(stat, labels, names[i], data_name)
        
    logging.info(f"###### job completed in: {datetime.now() - start_time}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparing gene and coordinates adjacency matrices')
    parser.add_argument('-mn', '--matrices_name', type=str, nargs='+', required=False,
                        help='Matrices names')
    parser.add_argument('-an', '--arrays_name', type=str, nargs='+', required=True,
                        help='Arrays name')
    args = parser.parse_args()
    
    logging.basicConfig(filename=f"plot_data_stats_log_{datetime.now()}.txt", level=logging.INFO)

    print("Execution of script started ...")
    plot_stats(args.matrices_name, args.arrays_name)
    print("Execution finished ...")
    