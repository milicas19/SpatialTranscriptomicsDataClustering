import pandas as pd
from datetime import datetime
import os
import anndata

const = 500000
DIV = 1000

######### HELPER FUNCTIONS FOR READING DATA #########
#### Reading ANN data
def read_anndata(data_name, st):
    print(f"====== Reading anndata {data_name}")
    
    if "midbrain" in data_name:
        return read_midbrain_data(data_name, st)
    
    adata = anndata.read_h5ad(f"anndata/{data_name}.h5ad")
    if 'X_spatial' in adata.obsm:
        spatial = adata.obsm['X_spatial']
    else:
        spatial = adata.obsm['spatial']
    print (f"Number of cells/spots: {len(spatial)}")
    sparse_matrix = adata.X
    
    # filter = int(len(spatial)/DIV)
    print("Removing genes with all zero values")
    dense_matrix = sparse_matrix.toarray()
    gene_df = pd.DataFrame(dense_matrix)
    filtered_columns = [col for col in gene_df.columns if (gene_df[col] != 0).sum() > 0]
    gene_df_filtered = gene_df[filtered_columns]
    
    ann = {}
    arr = []
    i = 0
    if "MOSTA" in data_name:
        for obj in adata.obs['annotation']:
            if obj not in ann:
                ann[obj] = i
                i += 1
        for obj in adata.obs['annotation']:
            arr.append(ann[obj])
    else:
        for obj in adata.obs['cell_type']:
            if obj not in ann:
                ann[obj] = i
                i += 1
        for obj in adata.obs['cell_type']:
            arr.append(ann[obj])
    
    return spatial, gene_df_filtered, arr

def read_midbrain_data(data_name, num_string):
    start =0
    end = 0
    adata = anndata.read_h5ad(f"anndata/{data_name}.h5ad")
    num = 0
    for i, cellId in enumerate(adata.obs['annotation'].index):
        if num_string in cellId:
            if num == 0:
                start = i
            num += 1
    end = start + num
    print(f"Midbrain -> spots/cells = {end-start}")
    spatial = adata.obsm['spatial'][start:end]
    print (f"Number of cells/spots: {len(spatial)}")
    sparse_matrix = adata.X[start:end]
    arr = []
    ann = {}
    i = 0
    index = adata.obs['annotation'].index[start:end]
    for id in index:
        obj = adata.obs['annotation'][id]
        if obj not in ann:
            ann[obj] = i
            i += 1
    for id in index:
        obj = adata.obs['annotation'][id]
        arr.append(ann[obj])
        
    # filter = int(len(spatial)/DIV)
    print("Removing genes with all zero values")
    dense_matrix = sparse_matrix.toarray()
    gene_df = pd.DataFrame(dense_matrix)
    filtered_columns = [col for col in gene_df.columns if (gene_df[col] != 0).sum() > 0]
    gene_df_filtered = gene_df[filtered_columns]
        
    return spatial, gene_df_filtered, arr
    
    
#### Reading CSV data
def read_csv(data_name):
    print(f"====== Reading csv data {data_name}")
    
    cell_data = pd.read_csv(f"output_data/{data_name}.tsv", sep='\t')
    cell_data.rename(columns={cell_data.columns[0]: 'cellID'}, inplace=True)
    
    print(f"Number of cells/spots {len(cell_data)}")
    
    return cell_data

######### END HELPER FUNCTIONS FOR READING DATA #########
    
def read_data(data_name, is_h5ad_data, st):
    return read_anndata(data_name, st) if is_h5ad_data else read_csv(data_name)

######### HELPER FUNCTIONS FOR CREATING CELL DATA #########
def calculate_mean(list):
    return int(sum(list)/len(list))

def add_cell_column(data):
    start_time = datetime.now()
    print("====== Adding cell column")
    row_num = len(data)
    cells = {}

    data['cell'] = list(range(0, row_num))
    
    for index, row in data.iterrows():
        if index != 0 and index % const == 0:
            print(f"Progress for adding cell column: {index}/{len(data)}")
            
        x = row['x']
        y = row['y']
        
        if (x,y) not in cells:
            cells[(x, y)] = index
        
        data.at[index, 'cell'] = cells[(x, y)]
    print("###### Cell column added")
    print(f"###### job completed in: {datetime.now() - start_time}")

def calculate_cell_coordinates(data, data_name):
    start_time = datetime.now()
    print(f"====== Calculating cell coordinates for {data_name}")
    n = len(data)
    cells_x = {}
    cells_y = {}
    cells = set()
    cells_coord = {}
        
    for index, row in data.iterrows():
        if index != 0 and index % const == 0:
            print(f"Progress for calculating cell coordinates: {index}/{n}")
        
        cellId = row['cell']
        cells.add(cellId)
        
        if cellId not in cells_x: 
            cells_x[cellId] = [row['x']]
        else:
            cells_x[cellId].append(row['x'])
            
        if cellId not in cells_y:
            cells_y[cellId] = [row['y']]
        else:
            cells_y[cellId].append(row['y'])

    for cellId in cells:
        x = int(calculate_mean(cells_x[cellId]))
        y = int(calculate_mean(cells_y[cellId]))
        cells_coord[cellId] = (x,y)
    
    print(f"###### job completed in: {datetime.now() - start_time}")
    return cells_coord

######### END HELPER FUNCTIONS FOR CREATING CELL DATA #########

def create_cell_data(input_data_name, output_data_name):
    if os.path.exists(f"output_data/{output_data_name}.tsv"):
        print(f"###### Cell data {output_data_name} already exists")
    else:
        start_time = datetime.now()
        print(f"====== Creating cell data for {input_data_name}")

        gene_exp_data = pd.read_csv(f"input_data/{input_data_name}.tsv", sep = '\t')
        num_rows = len(gene_exp_data)
        print(f"Data rows: {num_rows}")

        if 'cell' not in gene_exp_data.columns:
            add_cell_column(gene_exp_data)

        columns = set(gene_exp_data['geneID'].values)
        columns.add('x')
        columns.add('y')

        rows = set(gene_exp_data['cell'].values)

        cell_data = pd.DataFrame(0, index=list(rows), columns=list(columns))
        cells_coord = calculate_cell_coordinates(gene_exp_data, input_data_name)

        for index, row in gene_exp_data.iterrows():
            if index != 0 and index % const == 0:
                print(f"Progress: {index}/{num_rows}")
            if cell_data.loc[row['cell'], ['x']].values[0] == 0:
                (x, y) = cells_coord[row['cell']]
                cell_data.loc[row['cell'], ['x']] = x
                cell_data.loc[row['cell'], ['y']] = y
            if 'MIDCounts' in gene_exp_data.columns:
                cell_data.loc[row['cell'], [row['geneID']]] = row['MIDCounts']
            else:
                cell_data.loc[row['cell'], [row['geneID']]] = row['MIDCount']

        cell_data.to_csv(f"output_data/{output_data_name}.tsv", sep='\t')

    print(f"###### job completed in: {datetime.now() - start_time}")

