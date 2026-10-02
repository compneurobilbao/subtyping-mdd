""" MDD subclinical symptom associations with SFC-EXT derived MDD subtypes

This script performs association analyses between MDD subclinical symptom variables and the SFC-EXT derived MDD subtypes (Cluster 0 and Cluster 1),
as well as the total depression cohort (Cluster 0 + Cluster 1 combined) versus controls. It also generates plots to visualize these associations. 
Association analyses include Mann-Whitney U tests for continuous variables and Chi-squared tests for categorical variables, with 
multiple testing correction applied to the p-values. Information is provided on the number of individuals included in each analysis, 
as well as the number of individuals excluded due to null or unsuitable values.

Key symptom variables analyzed include:
- Guilty feelings
- Tiredness/lethargy
- Depressed mood
- Unenthusiasm/disinterest
- Restlessness

Key inputs expected:
- Cohort data with SFC-EXT derived subtypes:
    This is the csv file "module_connectivity_features_with_covariates.csv" produced by "module_clustering_main.py". It should contain the 
    following columns (among many others):
    - "eid": Unique identifier for each individual
    - "sfc_external_cluster": SFC-EXT derived subtype (Control, Cluster 0, Cluster 1)
- Symptom variable dataset:
    This is the csv file "mental_health.csv" containing the subclinical symptom variables for each individual.

"""
import os
from symptom_associations_utils import(
    load_and_preprocess_cohort_data,
    load_and_preprocess_symptom_data,
    print_percentiles_and_categorical_percentages,
    perform_statistical_tests_and_corrections,
    plot_symptom_distributions
)

def main():
    # Configuration
    GENERAL_DATA_PATH = ".../data/UKB"
    COHORT_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "cohorts")
    SYMPTOMS_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "mental_health")
    LOG_PATH = os.path.join(SYMPTOMS_DATA_PATH, "symptom_associations_log.txt")
    PLOTS_DIR = ".../reports/plots/symptom_associations"
    
    # Load and preprocess cohort data
    cohort_df = load_and_preprocess_cohort_data(COHORT_DATA_PATH)
    
    # Load and preprocess symptom datasets
    symptom_dfs, symptom_dfs_plots = load_and_preprocess_symptom_data(SYMPTOMS_DATA_PATH, cohort_df)
    
    # Print percentiles and categorical percentages for each dataset
    print_percentiles_and_categorical_percentages(symptom_dfs)
    
    # Perform statistical tests and corrections for each dataset
    corrected_pvals = perform_statistical_tests_and_corrections(symptom_dfs, LOG_PATH)
    
    # Plot distributions for each dataset
    plot_symptom_distributions(symptom_dfs_plots, corrected_pvals, PLOTS_DIR)

    # Print completion message
    print("Symptom association analyses completed successfully.")
    print(f"Results logged in: {LOG_PATH}")
    print(f"Plots saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
