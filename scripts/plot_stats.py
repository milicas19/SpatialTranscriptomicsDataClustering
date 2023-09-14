"""
Plotting mode, mean, median, variance, standard deviation, kurtosis, skewness, 
percentage of data in [-0.05, 0.05], percentage of data in [-0.1, 0.1], percentage of data in [-0.2, 0.2],
percentage of data in [mean - stand_dev, mean + stand_dev], percentage of data in [mean - 2 * stand_dev, mean + 2 * stand_dev], 
percentage of data in [mean - 3 * stand_dev, mean + 3 * stand_dev] for data sets (Dorsal Midbrain 12.5, 14.5 & 16.5, 
Human Cortex & Medula, Mouse Kidney, Mouse Embrio 9.5 (E1S1 & E2S4) & 10.5 (E2S1))
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
import statistics as st

EPS = 0.05
EPS1 = 0.1
EPS2 = 0.2
ALFA = 0.7

def stats_distr(data_array, data_name, plot_name):
    start_time = datetime.now()
    logging.info(f"====== Checking if CDM data for {data_name} has Gaussian distribution . . .")
    
    result = stats.anderson(data_array)
    logging.info('Statistic: %.3f' % result.statistic)
    
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            logging.info('Probably Gaussian at the %.1f%% level' % sl)
        else:
            logging.info('Probably not Gaussian at the %.1f%% level' % sl)

    plt.figure()
    stats.probplot(data_array, dist="norm", plot=plt)
    
    logging.info("Saving plot ...")
    
    data_name = re.sub(r'\.', '', data_name)
    
    plt.savefig(f"probplot_{data_name}_{plot_name}")
    plt.close() 
    
    logging.info(f"###### job completed in: {datetime.now() - start_time}")

def calc_stats(data_names, plot_name, is_adj, is_norm):
    start_time = datetime.now()
    logging.info("====== Calculating means, medians, variances and standard deviations . . .")
    
    means = []
    medians = []
    variances = []
    stand_devs = []
    modes = []
    
    labels = []
    
    kurtosis_val = [] 
    skewness_val = []
    onedev = []
    twodev = []
    threedev = []
    p_val = []
    p1_val = []
    p2_val = []
    al_val = []
    
    for data_name in data_names:
        logging.info(f"************************ {data_name} ************************")

        if is_norm:
            data_array = np.load(f"arrays/{data_name}.npy")
        elif is_adj:
            data_matrix = np.load(f"matrices/{data_name}.npy")
            upper_indices = np.triu_indices(data_matrix.shape[0], k=1)
            data_array = data_matrix[upper_indices]
        else:
            data_array = np.load(f"cdm_matrices/{data_name}.npy")

        labels.append(data_name)
        
        arr = data_array
        arr = np.round(arr, 2)
        l = list(arr)
        mode = max(set(l), key=l.count)
        print(mode)
        logging.info(f"Mode: {mode}")
        modes.append(mode)
        
        mean = np.mean(data_array)
        logging.info(f"Mean: {mean}")
        means.append(mean)
        
        median = np.median(data_array)
        logging.info(f"Median: {median}")
        medians.append(median)
        
        variance = round(np.var(data_array), 2)
        logging.info(f"Variance: {variance}")
        variances.append(variance)
        
        stand_dev = round(np.std(data_array), 2)
        logging.info(f"Standard deviation: {stand_dev}")
        stand_devs.append(stand_dev)
        
        k = round(kurtosis(data_array), 2)
        logging.info(f"Kurtosis: {k}")
        kurtosis_val.append(k)
        
        s = skew(data_array)
        logging.info(f"Skewness: {s}")
        skewness_val.append(s)
        
        counter = 0
        counter1 = 0
        counter2 = 0
        counter3 = 0
        counterst1 = 0
        counterst2 = 0
        counterst3 = 0
        for value in data_array:
            if value >= -EPS and value <= EPS:
                counter += 1
            if value >= -EPS1 and value <= EPS1:
                counter1 += 1
            if value >= -EPS2 and value <= EPS2:
                counter2 += 1
            if value >= ALFA and value <= -ALFA:
                counter3 += 1
            if value >= mean - stand_dev and value <= mean + stand_dev:
                counterst1 += 1
            if value >= mean - 2 * stand_dev and value <= mean + 2 * stand_dev:
                counterst2 += 1
            if value >= mean - 3 * stand_dev and value <= mean + 3* stand_dev:
                counterst3 += 1
        
        p = round(100 * (counter/len(data_array)), 1)
        logging.info(f"Percentage of data in [{-EPS}, {EPS}]: {counter}/{len(data_array)} -> {p}%")
        p_val.append(p)
        p1 = round(100 * (counter1/len(data_array)), 1)
        logging.info(f"Percentage of data in [{-EPS1}, {EPS1}]: {counter1}/{len(data_array)} -> {p1}%")
        p1_val.append(p1)
        p2 = round(100 * (counter2/len(data_array)), 1)
        logging.info(f"Percentage of data in [{-EPS2}, {EPS2}]: {counter2}/{len(data_array)} -> {p2}%")
        p2_val.append(p2)
        al = round(100 * (counter3/len(data_array)), 1)
        logging.info(f"Percentage of data in ends ({ALFA}): {counter3}/{len(data_array)} -> {al}%")
        al_val.append(al)
        dev1 = round(100 * (counterst1/len(data_array)), 1)
        logging.info(f"Percentage of data in 1 dev: {counterst1}/{len(data_array)} -> {dev1}%")
        onedev.append(dev1)
        dev2 = round(100 * (counterst2/len(data_array)), 1)
        logging.info(f"Percentage of data in 2 dev: {counterst2}/{len(data_array)} -> {dev2}%")
        twodev.append(dev2)
        dev3 = round(100 * (counterst3/len(data_array)), 1)
        logging.info(f"Percentage of data in 3 dev: {counterst3}/{len(data_array)} -> {dev3}%")
        threedev.append(dev3)
        logging.info("************************************************")
        
    logging.info(f"###### job completed in: {datetime.now() - start_time}")
    
    return labels, modes, means, medians, variances, stand_devs, kurtosis_val, skewness_val, p_val, p1_val, p2_val, al_val, onedev, twodev, threedev

def plot_stat(array, labels, name, metric, is_adj, is_norm):
    start_time = datetime.now()
    logging.info(f"====== Plotting {name} from data sets . . .")
    
    #colors = ['#EA96A7', '#F57C94', '#fb607f'] # for Embrio data
    #colors = ['#9EE3D7', '#77DFCD', '#59DCC5'] # for Kidney data
    #colors = ['#BEBADA', '#AAA3DE', '#9F95E8', '#9183F0', '#7F6FEC'] # for Dorsal Midbrain data
    colors = ['#EA96A7', '#F57C94', '#fb607f', '#BEBADA', '#AAA3DE', '#9F95E8', '#9183F0', '#7F6FEC', '#9EE3D7', '#77DFCD', '#59DCC5'] # for all
    
    color_map = dict(zip(labels, colors))
    if not os.path.exists(f"legend.png"):
        legend_fig = plt.figure()

        legend_ax = legend_fig.add_subplot(111)

        legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color_map[label]) for label in labels]
        legend_ax.legend(legend_patches, labels)

        legend_fig.savefig(f"legend.png")
        plt.close()

    main_fig = plt.figure()
    
    main_ax = main_fig.add_subplot(111)
    main_ax.set_title(name, fontsize=20)
    main_ax.bar(range(len(array)), array, color=[color_map[label] for label in labels])
    
    main_ax.xaxis.set_tick_params(labelsize=18)
    main_ax.yaxis.set_tick_params(labelsize=18)

    logging.info("Saving it ...")
    name = re.sub(r'\.', '', name)
    if is_adj:
        main_fig.savefig(f"dataset_adj_{name}_{metric}_isNorm_{is_norm}")
    else:
        main_fig.savefig(f"dataset_{name}_{metric}")
    
    plt.close()
    
    logging.info(f"###### job completed in: {datetime.now() - start_time}")
    
def plot_stats(data_names, plot_name, metric, is_adj, is_norm):
    start_time = datetime.now()
    logging.info(f"====== Comparing CDM for metric {metric} . . .")
    
    labels, modes, means, medians, variances, stand_devs, kurtosis_val, skewness_val, p_val, p1_val, p2_val, al_val, onedev, twodev, threedev= calc_stats(data_names, plot_name, is_adj, is_norm)
    stats = [modes, means, medians, variances, stand_devs, kurtosis_val, skewness_val, p_val, p1_val, p2_val, al_val, onedev, twodev, threedev]
    names = ['mode', 'mean', 'median', 'variance', 'standard deviation', 'kurtosis', 'skewness', 'percentage of data in [-0.05, 0.05]', 'percentage of data in [-0.1, 0.1]', 'percentage of data in [-0.2, 0.2]', 'al_val', 'percentage of data in [mean - stand_dev, mean + stand_dev]', 'percentage of data in [mean - 2 * stand_dev, mean + 2 * stand_dev]', 'percentage of data in [mean - 3 * stand_dev, mean + 3 * stand_dev]']
    for i, stat in enumerate(stats):
        plot_stat(stat, labels, names[i], metric, is_adj, is_norm)
    
    logging.info(f"###### job completed in: {datetime.now() - start_time}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparing different CDMs')
    parser.add_argument('-cdms', '--cdm_names', type=str, nargs='+', required=True,
                        help='CMD names')
    parser.add_argument('-plt', '--plot_name', type=str, required=True,
                        help='plot name for similarity with Gaussian distribution')
    parser.add_argument('-m', '--metric', default='euclidean', type=str, choices=['value', 'euclidean', 'manhattan', 'minkowski', 'cosine'],
                        help='metric used to create CDM matrix')
    parser.add_argument('-adj', '--is_adj_matrices', action='store_true',
                        help='are we using adjacency matrices instead of cdms')
    parser.add_argument('-norm', '--is_max_normalization', action='store_true',
                        help='using max normalization, otherwise cdf')
    args = parser.parse_args()
    
    logging.basicConfig(filename=f"plot_stats_log_{datetime.now()}_{args.metric}.txt", level=logging.INFO)

    print("Execution of script started ...")
    plot_stats(args.cdm_names, args.plot_name, args.metric, args.is_adj_matrices, args.is_max_normalization)
    print("Execution finished ...")
