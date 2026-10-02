""" Lifestyle and environment associations with SFC-EXT derived MDD subtypes.

This script performs association analyses between lifestyle/environment variables and the SFC-EXT derived MDD subtypes (Cluster 0 and Cluster 1), 
as well as the total depression cohort (Cluster 0 + Cluster 1 combined) versus controls. It also generates plots to visualize 
these associations. Association analyses include Mann-Whitney U tests for continuous variables and Chi-squared tests for categorical variables, 
with multiple testing correction applied to the p-values. Information is provided on the number of individuals included in each analysis, 
as well as the number of individuals excluded due to null or unsuitable values.

Key lifestyle/environment variables analyzed include:
- Alcohol consumption (drinker status and intake frequency)
- NMR metabolomics
- Physical activity
- Sleep patterns
- Social support
- Sun exposure

Key inputs expected:
- Cohort data with SFC-EXT derived subtypes:
    This is the csv file "module_connectivity_features_with_covariates.csv" produced by "module_clustering_main.py". It should contain the 
    following columns (among many others):
    - "eid": Unique identifier for each individual
    - "sfc_external_cluster": SFC-EXT derived subtype (Control, Cluster 0, Cluster 1)
- Lifestyle and environment variable datasets:
    These are the csv files "alcohol.csv", "NMR_metabolomics.csv", "physical_activity.csv", "sleep.csv", "social_support.csv", and "sun_exposure.csv"
"""
import os
from lifestyle_environment_associations_utils import(
    load_and_preprocess_cohort_data,
    load_and_preprocess_lifestyle_environment_data,
    print_percentiles_and_categorical_percentages,
    perform_statistical_tests_and_corrections,
    plot_lifestyle_environment_distributions
)

def main():
    # Configuration
    GENERAL_DATA_PATH = ".../data/UKB"
    COHORT_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "cohorts")
    LIFESTYLE_ENVIRONMENT_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "lifestyle_environment")
    LOG_PATH = os.path.join(LIFESTYLE_ENVIRONMENT_DATA_PATH, "lifestyle_environment_associations_log.txt")
    PLOTS_DIR = ".../reports/plots/lifestyle_environment_associations"
    
    # Load and preprocess cohort data
    cohort_df = load_and_preprocess_cohort_data(COHORT_DATA_PATH)
    
    # Load and preprocess lifestyle/environment datasets
    lifestyle_environment_dfs, lifestyle_environment_dfs_plots = load_and_preprocess_lifestyle_environment_data(LIFESTYLE_ENVIRONMENT_DATA_PATH, cohort_df)
    
    # Print percentiles and categorical percentages for each dataset
    print_percentiles_and_categorical_percentages(lifestyle_environment_dfs)
    
    # Perform statistical tests and corrections for each dataset
    corrected_pvals = perform_statistical_tests_and_corrections(lifestyle_environment_dfs, LOG_PATH)
    
    # Plot distributions for each dataset
    plot_lifestyle_environment_distributions(lifestyle_environment_dfs_plots, corrected_pvals, PLOTS_DIR)

    # Print completion message
    print("Lifestyle and environment association analyses completed successfully.")
    print(f"Results logged in: {LOG_PATH}")
    print(f"Plots saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()

