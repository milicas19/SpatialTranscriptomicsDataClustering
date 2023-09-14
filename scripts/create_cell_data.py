"""
Creating cell data for different data sets
"""
import argparse
from datetime import datetime

from data_util import create_cell_data

def create_data(data_sets, cell_data_names):
    for i, data_set in enumerate(data_sets):
        start_time = datetime.now()
        print(f"=== {i+1}. Creating data for {data_set}")
        
        create_cell_data(data_set, cell_data_names[i])
        
        print(f"### job completed in: {datetime.now() - start_time}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparing CDM values for different sets of data')
    parser.add_argument('-ds', '--data_sets', type=str, nargs='+', required=True,
                        help='data sets for which cell data will be computed')
    parser.add_argument('-cdn', '--cell_data_names', type=str, nargs='+', required=True,
                        help='data names for cell data')
    
    args = parser.parse_args()

    create_data(args.data_sets, args.cell_data_names)

