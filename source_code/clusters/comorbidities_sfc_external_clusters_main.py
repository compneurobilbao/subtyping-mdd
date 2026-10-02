""" Determining comorbidity compositions for SFC external clusters

This script performs comorbidity composition analyses for SFC external clusters derived from the "module_connectivity_features_with_covariates.csv" file.
It includes the following steps:
1. Load and preprocess the cohort data containing SFC external clusters.
2. Load and preprocess the comorbidity data containing ICD-10 codes for each individual.
3. Extract relevant columns from both datasets and merge them on the unique identifier "eid".
4. For each SFC external cluster (Cluster 0 and Cluster 1), determine the distribution of ICD-10 codes and categories (letters) among individuals in that cluster.
5. Test for significant differences in comorbidity distributions between the two clusters using appropriate statistical tests.
6. Generate plots to visualize the comorbidity distributions and category distributions for each cluster.

Key inputs expected:
- Cohort data with SFC external clusters:
    This is the csv file "module_connectivity_features_with_covariates.csv" produced by "module_clustering_main.py". It should contain the following columns (among many others):
    - "eid": Unique identifier for each individual
    - "sfc_external_cluster": SFC external cluster (Control, Cluster 0, Cluster 1)
- Comorbidity data:
    This is the csv file "depression_cohort_F32.csv" containing the ICD-10 codes for each individual in the depression cohort. It should contain the following columns (among many others):
    - "eid": Unique identifier for each individual
    - "codes": A string of ICD-10 codes associated with the individual, separated by '|'.
- Coding file:
    This is the tsv file "coding19.tsv" containing the mapping of ICD-10 codes to their respective categories (letters). 
    It originates from the UK Biobank data showcase.
    
"""
import os
import logging
import pandas as pd
from comorbidities_sfc_external_clusters_utils import(
    setup_logging,
    load_and_preprocess_cohort_data,
    load_and_preprocess_comorbidity_data,
    extract_relevant_columns_and_merge,
    count_codes_in_cohort,
    letter_stats_overall,
    build_comorbidity_indicator_matrix,
    compare_comorbidities,
    compare_category_proportions,
    plot_comorbidity_distribution,
    plot_comorbidity_categories
)

def main():
    # Configuration
    GENERAL_DATA_PATH = ".../data/UKB"
    SFC_EXT_CLUSTERS_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "cohorts", "module_connectivity_features_with_covariates.csv")
    COMORBIDITIES_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "cohorts", "depression_cohort_F32.csv")
    PLOTS_DIR = ".../reports/plots/comorbidities_sfc_external_clusters"
    CODING_FILE_PATH = os.path.join(GENERAL_DATA_PATH, "coding19.tsv")
    LOG_PATH = os.path.join(GENERAL_DATA_PATH, "comorbidity_sfc_ext", "comorbidities_sfc_external_clusters_log.txt")

    # Set up logging to a file  
    setup_logging(LOG_PATH)

    # Load and preprocess cohort data
    cohort_df = load_and_preprocess_cohort_data(SFC_EXT_CLUSTERS_DATA_PATH)

    # Load and preprocess comorbidity data
    comorbidity_df = load_and_preprocess_comorbidity_data(COMORBIDITIES_DATA_PATH)
    
    # Extract relevant columns from SFC_EXT_DF and COMORBIDITIES_DF, and merge them on 'eid'
    merged_df = extract_relevant_columns_and_merge(cohort_df, comorbidity_df)
    
    # Comorbidities analysis
    print("Starting comorbidities analysis for SFC external clusters...")
    # 1: Comorbidity distribution for each SFC external cluster
    print("Calculating comorbidity distribution for each SFC external cluster...")
    # Get the ICD-10 code distribution for each SFC external cluster
    code_distribution_cluster_0 = count_codes_in_cohort(
        merged_df[merged_df['sfc_external_cluster'] == 'Cluster 0'], coding_filepath=CODING_FILE_PATH
    )
    logging.info(f"Code distribution for Cluster 0:\n{code_distribution_cluster_0.head()}")

    code_distribution_cluster_1 = count_codes_in_cohort(
        merged_df[merged_df['sfc_external_cluster'] == 'Cluster 1'], coding_filepath=CODING_FILE_PATH
    )
    logging.info(f"Code distribution for Cluster 1:\n{code_distribution_cluster_1.head()}")
    # Test significant differences in code distributions between clusters
    # First add comorbidity indicator columns to the combined_df for each cluster
    combined_df_with_indicators = build_comorbidity_indicator_matrix(
        merged_df,
        coding_filepath=CODING_FILE_PATH,
        eid_column="eid",
        codes_column_cohort="codes",
        proportion_threshold=0.10,
        exclude_codes="F32"
    )
    # Now test
    comorbidity_cols = [col for col in combined_df_with_indicators.columns if col != "sfc_external_cluster" and col != "eid" and col != "codes"]
    results = compare_comorbidities(
        combined_df_with_indicators,
        group_col="sfc_external_cluster",
        comorbidity_cols=comorbidity_cols,
    )
    logging.info(f"Comorbidity comparison results:\n{results}")  # all insignificant (p_adj > 0.05) after FDR correction, so no significant differences between clusters (consistent with previous tests on the top 3 in our supplementary material)
    # Plot the comorbidity distribution for each cluster
    plot_comorbidity_distribution(
        code_distribution_cluster_0,
        proportion_threshold=0.10,
        title="ICD-10 Code Distribution for SFC External Cluster 0",
        output_path=os.path.join(PLOTS_DIR, "sfc_external_cluster_0_comorbidity_distribution.svg"),
    )

    plot_comorbidity_distribution(
        code_distribution_cluster_1,
        proportion_threshold=0.10,
        title="ICD-10 Code Distribution for SFC External Cluster 1",
        output_path=os.path.join(PLOTS_DIR, "sfc_external_cluster_1_comorbidity_distribution.svg"),
    )
    
    # 2: Category distribution for each SFC external cluster
    print("Calculating ICD-10 category distribution for each SFC external cluster...")
    # Determine stats of distinct ICD-10 categories (letter counts) for each cluster
    f_letter_stats_overall_cluster_0 = letter_stats_overall(
        merged_df[merged_df['sfc_external_cluster'] == 'Cluster 0'],
        codes_column_cohort='codes'
    )
    logging.info(f"Letter stats for Cluster 0: {f_letter_stats_overall_cluster_0}")

    f_letter_stats_overall_cluster_1 = letter_stats_overall(
        merged_df[merged_df['sfc_external_cluster'] == 'Cluster 1'],
        codes_column_cohort='codes'
    )
    logging.info(f"Letter stats for Cluster 1: {f_letter_stats_overall_cluster_1}")
    # Test significant differences in letter category distributions between clusters
    # First, create a DataFrame with counts of each letter category for each cluster
    letter_counts_df = pd.DataFrame({
        'category': ['E', 'F', 'G', 'I', 'K', 'R', 'Z'],
        'count_cluster_0': [f_letter_stats_overall_cluster_0.get(f"{cat.lower()}_count", 0) for cat in ['E', 'F', 'G', 'I', 'K', 'R', 'Z']],
        'count_cluster_1': [f_letter_stats_overall_cluster_1.get(f"{cat.lower()}_count", 0) for cat in ['E', 'F', 'G', 'I', 'K', 'R', 'Z']],
    })
    n_cluster_0 = f_letter_stats_overall_cluster_0.get('total_letters', 1)
    n_cluster_1 = f_letter_stats_overall_cluster_1.get('total_letters', 1)
    # Now test
    category_results = compare_category_proportions(
        letter_counts_df,
        category_col='category',
        count_col_g1='count_cluster_0',
        count_col_g2='count_cluster_1',
        n_g1=n_cluster_0,
        n_g2=n_cluster_1,
        group_labels=('Cluster 0', 'Cluster 1'),
    )
    logging.info(f"Category comparison results:\n{category_results}")  # all insignificant (p_adj > 0.05) after FDR correction, so no significant differences between clusters 
    # Plot the letter category distribution for each cluster
    plot_comorbidity_categories(
        f_letter_stats_overall_cluster_0,
        title="ICD-10 Category Distribution for SFC External Cluster 0",
        output_path=os.path.join(PLOTS_DIR, "sfc_external_cluster_0_category_distribution.svg")
    )

    plot_comorbidity_categories(
        f_letter_stats_overall_cluster_1,
        title="ICD-10 Category Distribution for SFC External Cluster 1",
        output_path=os.path.join(PLOTS_DIR, "sfc_external_cluster_1_category_distribution.svg")
    )

    # Print completion message
    print("Comorbidity composition analyses completed successfully.")
    print(f"Results logged in: {LOG_PATH}")
    print(f"Plots saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()