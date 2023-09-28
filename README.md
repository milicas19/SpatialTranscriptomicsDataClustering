# Spatial Transcriptomics Data Clustering

Data sets used in this research:
- Mouse Embryo from https://db.cngb.org/stomics/mosta/download/ (E9.5_E1S1.MOSTA, E9.5_E2S4.MOSTA & E10.5_E2S1.MOSTA)
- Mouse Dorsal Midbrain from https://db.cngb.org/stomics/mosta/download/ (Dorsal_midbrain_cell_bin)
- Human Kideny and Mouse Kidney from https://cellxgene.cziscience.com/collections/8e880741-bf9a-4c8e-9227-934204631d2a (human_renal_medula_Puck_200205_13, human_cortex_kidney_Puck_200205_17 & mouse_kidney_Puck_191112_05)

## Task 1
Investigating the correlation between the spatial coordinates and gene expressions of the cells

- using 40 PCA components
<pre>
<code>
python3 create_and_visualize_cdms.py -dn  Dorsal_midbrain_cell_bin -pca 40 40 40 40 40 40 -norm -h5ad -dorsal -col '#BEBADA' '#AAA3DE' '#9F95E8' '#9183F0' '#7F6FEC' -mod "stddev"
</code>
</pre>

<pre>
<code>
python3 create_and_visualize_cdms.py -dn E9.5_E1S1.MOSTA  E9.5_E2S4.MOSTA E10.5_E2S1.MOSTA human_renal_medula_Puck_200205_13 human_cortex_kidney_Puck_200205_17 mouse_kidney_Puck_191112_05 -pca 40 40 40 40 40 40 -norm -h5ad -col '#EA96A7' '#F57C94' '#fb607f' '#9EE3D7' '#77DFCD' '#59DCC5' -mod "stddev"
</code>
</pre>

<pre>
<code>
python3 plot_stats.py -cdms cdm_E9.5_E1S1.MOSTA_pca_40_value_euclidean cdm_E9.5_E2S4.MOSTA_pca_40_value_euclidean cdm_E10.5_E2S1.MOSTA_pca_40_value_euclidean cdm_Dorsal_midbrain_cell_bin_1_pca_40_value_euclidean cdm_Dorsal_midbrain_cell_bin_2_pca_40_value_euclidean cdm_Dorsal_midbrain_cell_bin_3_pca_40_value_euclidean cdm_Dorsal_midbrain_cell_bin_4_pca_40_value_euclidean cdm_Dorsal_midbrain_cell_bin_5_pca_40_value_euclidean cdm_human_renal_medula_Puck_200205_13_pca_40_value_euclidean cdm_human_cortex_kidney_Puck_200205_17_pca_40_value_euclidean cdm_mouse_kidney_Puck_191112_05_pca_40_value_euclidean -plt pltName -m value
</code>
</pre>

<pre>
<code>
python3 plot_data_stats.py -mn coord_Dorsal_midbrain_cell_bin_1_adj_matrix coord_Dorsal_midbrain_cell_bin_2_adj_matrix coord_Dorsal_midbrain_cell_bin_3_adj_matrix coord_Dorsal_midbrain_cell_bin_4_adj_matrix coord_Dorsal_midbrain_cell_bin_5_adj_matrix coord_E9.5_E1S1.MOSTA_adj_matrix coord_E9.5_E2S4.MOSTA_adj_matrix coord_E10.5_E2S1.MOSTA_adj_matrix  coord_human_renal_medula_Puck_200205_13_adj_matrix coord_human_cortex_kidney_Puck_200205_17_adj_matrix coord_mouse_kidney_Puck_191112_05_adj_matrix gene_Dorsal_midbrain_cell_bin_1_adj_matrix_pca_40 gene_Dorsal_midbrain_cell_bin_2_adj_matrix_pca_40 gene_Dorsal_midbrain_cell_bin_3_adj_matrix_pca_40 gene_Dorsal_midbrain_cell_bin_4_adj_matrix_pca_40 gene_Dorsal_midbrain_cell_bin_5_adj_matrix_pca_40 gene_E9.5_E1S1.MOSTA_adj_matrix_pca_40 gene_E9.5_E2S4.MOSTA_adj_matrix_pca_40 gene_E10.5_E2S1.MOSTA_adj_matrix_pca_40 gene_human_renal_medula_Puck_200205_13_adj_matrix_pca_40  gene_human_cortex_kidney_Puck_200205_17_adj_matrix_pca_40  gene_mouse_kidney_Puck_191112_05_adj_matrix_pca_40  -an max_normalized_coord_Dorsal_midbrain_cell_bin_1_adj_matrix max_normalized_coord_Dorsal_midbrain_cell_bin_2_adj_matrix max_normalized_coord_Dorsal_midbrain_cell_bin_3_adj_matrix max_normalized_coord_Dorsal_midbrain_cell_bin_4_adj_matrix max_normalized_coord_Dorsal_midbrain_cell_bin_5_adj_matrix  max_normalized_coord_E9.5_E1S1.MOSTA_adj_matrix max_normalized_coord_E9.5_E2S4.MOSTA_adj_matrix max_normalized_coord_E10.5_E2S1.MOSTA_adj_matrix max_normalized_coord_human_renal_medula_Puck_200205_13_adj_matrix max_normalized_coord_human_cortex_kidney_Puck_200205_17_adj_matrix max_normalized_coord_mouse_kidney_Puck_191112_05_adj_matrix max_normalized_gene_Dorsal_midbrain_cell_bin_1_adj_matrix_pca_40 max_normalized_gene_Dorsal_midbrain_cell_bin_2_adj_matrix_pca_40 max_normalized_gene_Dorsal_midbrain_cell_bin_3_adj_matrix_pca_40 max_normalized_gene_Dorsal_midbrain_cell_bin_4_adj_matrix_pca_40 max_normalized_gene_Dorsal_midbrain_cell_bin_5_adj_matrix_pca_40  max_normalized_gene_E9.5_E1S1.MOSTA_adj_matrix_pca_40 max_normalized_gene_E9.5_E2S4.MOSTA_adj_matrix_pca_40 max_normalized_gene_E10.5_E2S1.MOSTA_adj_matrix_pca_40 max_normalized_gene_human_renal_medula_Puck_200205_13_adj_matrix_pca_40 max_normalized_gene_human_cortex_kidney_Puck_200205_17_adj_matrix_pca_40 max_normalized_gene_mouse_kidney_Puck_191112_05_adj_matrix_pca_40
</code>
</pre>

- using a number of PCA components to cover 80% of the data variance
<pre>
<code>
python3 create_and_visualize_cdms.py -dn  Dorsal_midbrain_cell_bin -pca 2300 3800 2300 3200 4000 -norm -h5ad -dorsal -col '#BEBADA' '#AAA3DE' '#9F95E8' '#9183F0' '#7F6FEC' -mod "stddev"
</code>
</pre>

<pre>
<code>
python3 create_and_visualize_cdms.py -dn E9.5_E1S1.MOSTA  E9.5_E2S4.MOSTA E10.5_E2S1.MOSTA human_renal_medula_Puck_200205_13 human_cortex_kidney_Puck_200205_17 mouse_kidney_Puck_191112_05 -pca 3400 3400 4500 4500 3000 4000 -norm -h5ad -col '#EA96A7' '#F57C94' '#fb607f' '#9EE3D7' '#77DFCD' '#59DCC5' -mod "stddev"
</code>
</pre>

<pre>
<code>
python3 plot_stats.py -cdms cdm_E9.5_E1S1.MOSTA_pca_3400_value_euclidean cdm_E9.5_E2S4.MOSTA_pca_3400_value_euclidean cdm_E10.5_E2S1.MOSTA_pca_4500_value_euclidean cdm_Dorsal_midbrain_cell_bin_1_pca_2300_value_euclidean cdm_Dorsal_midbrain_cell_bin_2_pca_3800_value_euclidean cdm_Dorsal_midbrain_cell_bin_3_pca_2300_value_euclidean cdm_Dorsal_midbrain_cell_bin_4_pca_3200_value_euclidean cdm_Dorsal_midbrain_cell_bin_5_pca_4000_value_euclidean cdm_human_renal_medula_Puck_200205_13_pca_4500_value_euclidean cdm_human_cortex_kidney_Puck_200205_17_pca_3000_value_euclidean cdm_mouse_kidney_Puck_191112_05_pca_4000_value_euclidean -plt pltName -m value
</code>
</pre>

<pre>
<code>
python3 plot_data_stats.py -mn coord_Dorsal_midbrain_cell_bin_1_adj_matrix coord_Dorsal_midbrain_cell_bin_2_adj_matrix coord_Dorsal_midbrain_cell_bin_3_adj_matrix coord_Dorsal_midbrain_cell_bin_4_adj_matrix coord_Dorsal_midbrain_cell_bin_5_adj_matrix coord_E9.5_E1S1.MOSTA_adj_matrix coord_E9.5_E2S4.MOSTA_adj_matrix coord_E10.5_E2S1.MOSTA_adj_matrix  coord_human_renal_medula_Puck_200205_13_adj_matrix coord_human_cortex_kidney_Puck_200205_17_adj_matrix coord_mouse_kidney_Puck_191112_05_adj_matrix gene_Dorsal_midbrain_cell_bin_1_adj_matrix_pca_2300 gene_Dorsal_midbrain_cell_bin_2_adj_matrix_pca_3800 gene_Dorsal_midbrain_cell_bin_3_adj_matrix_pca_2300 gene_Dorsal_midbrain_cell_bin_4_adj_matrix_pca_3200 gene_Dorsal_midbrain_cell_bin_5_adj_matrix_pca_4000 gene_E9.5_E1S1.MOSTA_adj_matrix_pca_3400 gene_E9.5_E2S4.MOSTA_adj_matrix_pca_3400 gene_E10.5_E2S1.MOSTA_adj_matrix_pca_4500 gene_human_renal_medula_Puck_200205_13_adj_matrix_pca_4500  gene_human_cortex_kidney_Puck_200205_17_adj_matrix_pca_3000  gene_mouse_kidney_Puck_191112_05_adj_matrix_pca_4000  -an max_normalized_coord_Dorsal_midbrain_cell_bin_1_adj_matrix max_normalized_coord_Dorsal_midbrain_cell_bin_2_adj_matrix max_normalized_coord_Dorsal_midbrain_cell_bin_3_adj_matrix max_normalized_coord_Dorsal_midbrain_cell_bin_4_adj_matrix max_normalized_coord_Dorsal_midbrain_cell_bin_5_adj_matrix  max_normalized_coord_E9.5_E1S1.MOSTA_adj_matrix max_normalized_coord_E9.5_E2S4.MOSTA_adj_matrix max_normalized_coord_E10.5_E2S1.MOSTA_adj_matrix max_normalized_coord_human_renal_medula_Puck_200205_13_adj_matrix max_normalized_coord_human_cortex_kidney_Puck_200205_17_adj_matrix max_normalized_coord_mouse_kidney_Puck_191112_05_adj_matrix max_normalized_gene_Dorsal_midbrain_cell_bin_1_adj_matrix_pca_2300 max_normalized_gene_Dorsal_midbrain_cell_bin_2_adj_matrix_pca_3800 max_normalized_gene_Dorsal_midbrain_cell_bin_3_adj_matrix_pca_2300 max_normalized_gene_Dorsal_midbrain_cell_bin_4_adj_matrix_pca_3200 max_normalized_gene_Dorsal_midbrain_cell_bin_5_adj_matrix_pca_4000  max_normalized_gene_E9.5_E1S1.MOSTA_adj_matrix_pca_3400 max_normalized_gene_E9.5_E2S4.MOSTA_adj_matrix_pca_3400 max_normalized_gene_E10.5_E2S1.MOSTA_adj_matrix_pca_4500 max_normalized_gene_human_renal_medula_Puck_200205_13_adj_matrix_pca_4500 max_normalized_gene_human_cortex_kidney_Puck_200205_17_adj_matrix_pca_3000 max_normalized_gene_mouse_kidney_Puck_191112_05_adj_matrix_pca_4000
</code>
</pre>

## Taks 2
Analyzing the impact of spatial coordinates and gene expressions of cell on cell type

- using 40 PCA components WITHOUT reduction:
<pre>
<code>
python3 cluster_data.py -dn Dorsal_midbrain_cell_bin -dorsal -pca 40 40 40 40 40 -h5ad
</code>
</pre>

<pre>
<code>
python3 cluster_data.py -dn E9.5_E1S1.MOSTA E9.5_E2S4.MOSTA E10.5_E2S1.MOSTA human_renal_medula_Puck_200205_13 human_cortex_kidney_Puck_200205_17 mouse_kidney_Puck_191112_05 -pca 40 40 40 40 40 40 -h5ad
</code>
</pre>

- using 40 PCA components WITH reduction:
<pre>
<code>
python3 cluster_data.py -dn Dorsal_midbrain_cell_bin -dorsal -pca 40 40 40 40 40 -h5ad -red
</code>
</pre>

<pre>
<code>
python3 cluster_data.py -dn E9.5_E1S1.MOSTA E9.5_E2S4.MOSTA E10.5_E2S1.MOSTA human_renal_medula_Puck_200205_13 human_cortex_kidney_Puck_200205_17 mouse_kidney_Puck_191112_05 -pca 40 40 40 40 40 40 -h5ad -red
</code>
</pre>

- using a number of PCA components to cover 80% of the data variance WITHOUT reduction:
<pre>
<code>
python3 cluster_data.py -dn Dorsal_midbrain_cell_bin -dorsal -pca 2300 3800 2300 3200 4000 -h5ad
</code>
</pre>

<pre>
<code>
python3 cluster_data.py -dn E9.5_E1S1.MOSTA E9.5_E2S4.MOSTA E10.5_E2S1.MOSTA human_renal_medula_Puck_200205_13 human_cortex_kidney_Puck_200205_17 mouse_kidney_Puck_191112_05 -pca 3400 3400 4500 4500 3000 4000 -h5ad
</code>
</pre>

- using a number of PCA components to cover 80% of the data variance WITH reduction:
<pre>
<code>
python3 cluster_data.py -dn Dorsal_midbrain_cell_bin -dorsal -pca 2300 3800 2300 3200 4000 -h5ad -red
</code>
</pre>

<pre>
<code>
python3 cluster_data.py -dn E9.5_E1S1.MOSTA E9.5_E2S4.MOSTA E10.5_E2S1.MOSTA human_renal_medula_Puck_200205_13 human_cortex_kidney_Puck_200205_17 mouse_kidney_Puck_191112_05 -pca 3400 3400 4500 4500 3000 4000 -h5ad -red
</code>
</pre>
