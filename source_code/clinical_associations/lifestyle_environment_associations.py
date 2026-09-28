""" Lifestyle and environment associations with SFC-EXT derived MDD subtypes
"""
import os
import pandas as pd
import seaborn as sns
import ptitprince as pt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import mannwhitneyu, chi2_contingency
import numpy as np
from typing import List, Tuple, Optional, Literal
from statsmodels.stats import multitest

# ==============================================================================
# Configuration
# ==============================================================================
GENERAL_DATA_PATH = ".../data/UKB"
COHORT_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "cohorts")
LIFESTYLE_ENVIRONMENT_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "lifestyle_environment")
LOG_PATH = os.path.join(LIFESTYLE_ENVIRONMENT_DATA_PATH, "lifestyle_environment_associations_log.txt")
PLOTS_DIR = ".../reports/plots/lifestyle_environment_associations"
# ==============================================================================
# Load datasets
# ==============================================================================
# Combined MDD and control cohort, with SFC-EXT derived subtypes
COHORT_DF = pd.read_csv(os.path.join(COHORT_DATA_PATH, "module_connectivity_features_with_covariates.csv"))
print(f"Combined MDD and control cohort loaded with shape: {COHORT_DF.shape}")
# Check if subtype column exists
if "sfc_external_cluster" not in COHORT_DF.columns:
    raise ValueError("Column 'sfc_external_cluster' not found in COHORT_DF")
# Replace cluster labels with 'Cluster 0' and 'Cluster 1' for clarity
COHORT_DF["sfc_external_cluster"] = COHORT_DF["sfc_external_cluster"].map({"Control": "Control", "0": "Cluster 0", "1": "Cluster 1"})

# ALCOHOL DATASET
print("\n[1/6] Processing alcohol dataset...")
ALCOHOL_DF = pd.read_csv(os.path.join(LIFESTYLE_ENVIRONMENT_DATA_PATH, "alcohol.csv"))
print(f"Alcohol dataset loaded with shape: {ALCOHOL_DF.shape}")
# Get just the variables we need for the analysis
ALCOHOL_VARS = ["p20117_i2", "p1558_i2"] # alcohol drinker status, alcohol intake frequency
ALCOHOL_DF = ALCOHOL_DF[["eid"] + ALCOHOL_VARS]
print(f"Alcohol dataset filtered to relevant variables with shape: {ALCOHOL_DF.shape}")
# Now check if MDD and control people have these variables and that values are non-null
# First ensure both eid columns are of the same type
COHORT_DF["eid"] = COHORT_DF["eid"].astype(str)
ALCOHOL_DF["eid"] = ALCOHOL_DF["eid"].astype(str)
# Now check if the eids in ALCOHOL_DF are in COHORT_DF
ALCOHOL_DF = ALCOHOL_DF[ALCOHOL_DF["eid"].isin(COHORT_DF["eid"])]
print(f"Alcohol dataset filtered to MDD and control cohort with shape: {ALCOHOL_DF.shape}")
# Check for null-values, if any individual dropped due to null values, show which eid they had and which variable was null
null_eids = ALCOHOL_DF[ALCOHOL_DF[ALCOHOL_VARS].isnull().any(axis=1)]["eid"].tolist()
if null_eids:
    print(f"Warning: The following eids had null values for relevant alcohol variables and were dropped: {null_eids}")
    ALCOHOL_DF = ALCOHOL_DF.dropna(subset=ALCOHOL_VARS)
    print(f"Alcohol dataset filtered to remove null values with shape: {ALCOHOL_DF.shape}")
# Now check unique values for remaining individuals
for var in ALCOHOL_VARS:
    print(f"Variable {var} unique values: {ALCOHOL_DF[var].unique()}")
# Both variables show the value -3, which indicates "Prefer not to answer" in UKB coding. We will drop these individuals from the analysis.
unsuitable_eids = ALCOHOL_DF[ALCOHOL_DF[ALCOHOL_VARS].isin([-3]).any(axis=1)]["eid"].tolist()
ALCOHOL_DF = ALCOHOL_DF[~ALCOHOL_DF[ALCOHOL_VARS].isin([-3]).any(axis=1)]
print(f"Alcohol dataset filtered to remove 'Prefer not to answer' responses with shape: {ALCOHOL_DF.shape}")
# Check again the range of values for remaining individuals
for var in ALCOHOL_VARS:
    print(f"Variable {var} unique values: {ALCOHOL_DF[var].unique()}")
# Check to which cluster the excluded individuals belonged
excluded_eids = null_eids + unsuitable_eids
excluded_cluster_info = COHORT_DF[COHORT_DF["eid"].isin(excluded_eids)][["eid", "sfc_external_cluster"]]
print(f"Excluded individuals from alcohol dataset due to null or unsuitable values:\n{excluded_cluster_info['sfc_external_cluster'].value_counts()}")
# Now set the correct data type for the variable p20117_i2 (alcohol drinker status) to categorical, as it is a categorical variable
ALCOHOL_DF["p20117_i2"] = ALCOHOL_DF["p20117_i2"].astype("str")
# Now add the sfc_external_cluster column to the alcohol dataset for association analysis
ALCOHOL_DF = ALCOHOL_DF.merge(COHORT_DF[["eid", "sfc_external_cluster"]], on="eid", how="left")
print(f"Alcohol dataset after merging with SFC-EXT clusters: {ALCOHOL_DF.info()}")
# Reverse the order of the alcohol intake frequency variable (p1558_i2) so that higher values indicate more frequent drinking, for easier interpretation in plots
ALCOHOL_DF["p1558_i2"] = ALCOHOL_DF["p1558_i2"].map({
    1: 6,  # "Daily or almost daily" -> 6
    2: 5,  # "Three or four times a week" -> 5
    3: 4,  # "Once or twice a week" -> 4
    4: 3,  # "One to three times a month" -> 3
    5: 2,  # "Special occasions only" -> 2
    6: 1,  # "Never" -> 1
})
# Append total depression cohort (Cluster 0 + Cluster 1 combined)
dep_df = ALCOHOL_DF[ALCOHOL_DF["sfc_external_cluster"].isin(["Cluster 0", "Cluster 1"])].copy()
dep_df["sfc_external_cluster"] = "Depression"
ALCOHOL_DF = pd.concat([ALCOHOL_DF, dep_df], ignore_index=True)

# Now ensure the sfc_external_cluster column is categorical with the correct order (correct for histplots)
ALCOHOL_plots_df = ALCOHOL_DF.copy() # We will use this copy for plotting, while keeping the original ALCOHOL_DF for analysis
ALCOHOL_plots_df["sfc_external_cluster"] = pd.Categorical(
    ALCOHOL_plots_df["sfc_external_cluster"], categories=["Control", "Depression", "Cluster 0", "Cluster 1"], ordered=True
)
# Now we have a clean alcohol dataset with relevant variables for the MDD and control cohort, ready for association analysis with SFC-EXT derived subtypes.

# NMR METABOLOMICS DATASET
print("\n[2/6] Processing NMR metabolomics dataset...")
NMR_METABOLOMICS_DF = pd.read_csv(os.path.join(LIFESTYLE_ENVIRONMENT_DATA_PATH, "NMR_metabolomics.csv"))
print(f"NMR metabolomics dataset loaded with shape: {NMR_METABOLOMICS_DF.shape}")
# Get just the variables we need for the analysis
NMR_METABOLOMICS_VARS = ["p23407_i0", "p23400_i0", "p23406_i0", "p23405_i0"] # total triglycerides, total cholesterol, HDL cholesterol, LDL cholesterol
NMR_METABOLOMICS_DF = NMR_METABOLOMICS_DF[["eid"] + NMR_METABOLOMICS_VARS]
print(f"NMR metabolomics dataset filtered to relevant variables with shape: {NMR_METABOLOMICS_DF.shape}")
# Now check if MDD and control people have these variables and that values are non-null
# First ensure both eid columns are of the same type
COHORT_DF["eid"] = COHORT_DF["eid"].astype(str)
NMR_METABOLOMICS_DF["eid"] = NMR_METABOLOMICS_DF["eid"].astype(str)
# Now check if the eids in NMR_METABOLOMICS_DF are in COHORT_DF
NMR_METABOLOMICS_DF = NMR_METABOLOMICS_DF[NMR_METABOLOMICS_DF["eid"].isin(COHORT_DF["eid"])]
print(f"NMR metabolomics dataset filtered to MDD and control cohort with shape: {NMR_METABOLOMICS_DF.shape}")
# Check for null-values, if any individual dropped due to null values, show which eid they had and which variable was null
null_eids = NMR_METABOLOMICS_DF[NMR_METABOLOMICS_DF[NMR_METABOLOMICS_VARS].isnull().any(axis=1)]["eid"].tolist()
if null_eids:
    print(f"Warning: The following eids had null values for relevant nmr variables and were dropped: {null_eids}")
    NMR_METABOLOMICS_DF = NMR_METABOLOMICS_DF.dropna(subset=NMR_METABOLOMICS_VARS)
    print(f"NMR metabolomics dataset filtered to remove null values with shape: {NMR_METABOLOMICS_DF.shape}")
    # Check to which cluster these individuals belonged
    dropped_cluster_info = COHORT_DF[COHORT_DF["eid"].isin(null_eids)][["eid", "sfc_external_cluster"]]
    print(f"Dropped individuals from NMR metabolomics dataset due to null values:\n{dropped_cluster_info['sfc_external_cluster'].value_counts()}")
# Now check range of values for remaining individuals
for var in NMR_METABOLOMICS_VARS:
    print(f"Variable {var} description: {NMR_METABOLOMICS_DF[var].describe()}")
# Now create the insulin resistance proxy as the ratio of triglycerides to HDL cholesterol (as done in Oliveri et al., 2024)
NMR_METABOLOMICS_DF["insulin_resistance_proxy"] = NMR_METABOLOMICS_DF["p23407_i0"] / NMR_METABOLOMICS_DF["p23406_i0"]
# Check the description of the new variable
print(f"Insulin resistance proxy variable description: {NMR_METABOLOMICS_DF['insulin_resistance_proxy'].describe()}")
# Now add the sfc_external_cluster column to the nmr dataset for association analysis
NMR_METABOLOMICS_DF = NMR_METABOLOMICS_DF.merge(COHORT_DF[["eid", "sfc_external_cluster"]], on="eid", how="left")
print(f"NMR metabolomics dataset after merging with SFC-EXT clusters: {NMR_METABOLOMICS_DF.info()}")
# Append total depression cohort (Cluster 0 + Cluster 1 combined)
dep_df = NMR_METABOLOMICS_DF[NMR_METABOLOMICS_DF["sfc_external_cluster"].isin(["Cluster 0", "Cluster 1"])].copy()
dep_df["sfc_external_cluster"] = "Depression"
NMR_METABOLOMICS_DF = pd.concat([NMR_METABOLOMICS_DF, dep_df], ignore_index=True)

# Now ensure the sfc_external_cluster column is categorical with the correct order (correct for histplots)
NMR_METABOLOMICS_plots_df = NMR_METABOLOMICS_DF.copy() # We will use this copy for plotting, while keeping the original NMR_METABOLOMICS_DF for analysis
NMR_METABOLOMICS_plots_df["sfc_external_cluster"] = pd.Categorical(
    NMR_METABOLOMICS_plots_df["sfc_external_cluster"], categories=["Control", "Depression", "Cluster 0", "Cluster 1"], ordered=True
)
# Now we have a clean NMR metabolomics dataset with relevant variables for the MDD and control cohort, ready for association analysis with SFC-EXT derived subtypes.

# PHYSICAL ACTIVITY DATASET
print("\n[3/6] Processing physical activity dataset...")
PHYSICAL_ACTIVITY_DF = pd.read_csv(os.path.join(LIFESTYLE_ENVIRONMENT_DATA_PATH, "physical_activity.csv"))
print(f"Physical activity dataset loaded with shape: {PHYSICAL_ACTIVITY_DF.shape}")
# Get just the variables we need for the analysis
PHYSICAL_ACTIVITY_VARS = ["p22040_i2", "p22037_i2", "p22038_i2", "p22039_i2"] # total MET minutes per week for all activity, walking MET minutes per week, moderate MET minutes per week, vigorous MET minutes per week
PHYSICAL_ACTIVITY_DF = PHYSICAL_ACTIVITY_DF[["eid"] + PHYSICAL_ACTIVITY_VARS]
print(f"Physical activity dataset filtered to relevant variables with shape: {PHYSICAL_ACTIVITY_DF.shape}")
# Now check if MDD and control people have these variables and that values are non-null
# First ensure both eid columns are of the same type
COHORT_DF["eid"] = COHORT_DF["eid"].astype(str)
PHYSICAL_ACTIVITY_DF["eid"] = PHYSICAL_ACTIVITY_DF["eid"].astype(str)
# Now check if the eids in PHYSICAL_ACTIVITY_DF are in COHORT_DF
PHYSICAL_ACTIVITY_DF = PHYSICAL_ACTIVITY_DF[PHYSICAL_ACTIVITY_DF["eid"].isin(COHORT_DF["eid"])]
print(f"Physical activity dataset filtered to MDD and control cohort with shape: {PHYSICAL_ACTIVITY_DF.shape}")
# Check for null-values, if any individual dropped due to null values, show which eid they had and which variable was null
null_eids = PHYSICAL_ACTIVITY_DF[PHYSICAL_ACTIVITY_DF[PHYSICAL_ACTIVITY_VARS].isnull().any(axis=1)]["eid"].tolist()
if null_eids:
    print(f"Warning: The following eids had null values for relevant physical activity variables and were dropped: {null_eids}")
    PHYSICAL_ACTIVITY_DF = PHYSICAL_ACTIVITY_DF.dropna(subset=PHYSICAL_ACTIVITY_VARS)
    print(f"Physical activity dataset filtered to remove null values with shape: {PHYSICAL_ACTIVITY_DF.shape}")
    # Check to which cluster these individuals belonged
    dropped_cluster_info = COHORT_DF[COHORT_DF["eid"].isin(null_eids)][["eid", "sfc_external_cluster"]]
    print(f"Dropped individuals from physical activity dataset due to null values:\n{dropped_cluster_info['sfc_external_cluster'].value_counts()}")
# Now check range of values for remaining individuals
for var in PHYSICAL_ACTIVITY_VARS:
    print(f"Variable {var} description: {PHYSICAL_ACTIVITY_DF[var].describe()}")
# Now add the sfc_external_cluster column to the physical activity dataset for association analysis
PHYSICAL_ACTIVITY_DF = PHYSICAL_ACTIVITY_DF.merge(COHORT_DF[["eid", "sfc_external_cluster"]], on="eid", how="left")
print(f"Physical activity dataset after merging with SFC-EXT clusters: {PHYSICAL_ACTIVITY_DF.info()}")
# Append total depression cohort (Cluster 0 + Cluster 1 combined)
dep_df = PHYSICAL_ACTIVITY_DF[PHYSICAL_ACTIVITY_DF["sfc_external_cluster"].isin(["Cluster 0", "Cluster 1"])].copy()
dep_df["sfc_external_cluster"] = "Depression"
PHYSICAL_ACTIVITY_DF = pd.concat([PHYSICAL_ACTIVITY_DF, dep_df], ignore_index=True)

# Now ensure the sfc_external_cluster column is categorical with the correct order (correct for histplots)
PHYSICAL_ACTIVITY_plots_df = PHYSICAL_ACTIVITY_DF.copy() # We will use this copy for plotting, while keeping the original PHYSICAL_ACTIVITY_DF for analysis
PHYSICAL_ACTIVITY_plots_df["sfc_external_cluster"] = pd.Categorical(
    PHYSICAL_ACTIVITY_plots_df["sfc_external_cluster"], categories=["Control", "Depression", "Cluster 0", "Cluster 1"], ordered=True
)
# Now we have a clean physical activity dataset with relevant variables for the MDD and control cohort, ready for association analysis with SFC-EXT derived subtypes.

# SLEEP DATASET
print("\n[4/6] Processing sleep dataset...")
SLEEP_DF = pd.read_csv(os.path.join(LIFESTYLE_ENVIRONMENT_DATA_PATH, "sleep.csv"))
print(f"Sleep dataset loaded with shape: {SLEEP_DF.shape}")
# Get just the variables we need for the analysis
SLEEP_VARS = ["p1160_i2", "p1200_i2", "p1180_i2"] # sleep duration in every 24h (including naps), sleeplessness/insomnia, morning/evening chronotype
SLEEP_DF = SLEEP_DF[["eid"] + SLEEP_VARS]
print(f"Sleep dataset filtered to relevant variables with shape: {SLEEP_DF.shape}")
# Now check if MDD and control people have these variables and that values are non-null
# First ensure both eid columns are of the same type
COHORT_DF["eid"] = COHORT_DF["eid"].astype(str)
SLEEP_DF["eid"] = SLEEP_DF["eid"].astype(str)
# Now check if the eids in SLEEP_DF are in COHORT_DF
SLEEP_DF = SLEEP_DF[SLEEP_DF["eid"].isin(COHORT_DF["eid"])]
print(f"Sleep dataset filtered to MDD and control cohort with shape: {SLEEP_DF.shape}")
# Check for null-values, if any individual dropped due to null values, show which eid they had and which variable was null
null_eids = SLEEP_DF[SLEEP_DF[SLEEP_VARS].isnull().any(axis=1)]["eid"].tolist()
if null_eids:
    print(f"Warning: The following eids had null values for relevant sleep variables and were dropped: {null_eids}")
    SLEEP_DF = SLEEP_DF.dropna(subset=SLEEP_VARS)
    print(f"Sleep dataset filtered to remove null values with shape: {SLEEP_DF.shape}")
# Now check range of values for remaining individuals
for var in SLEEP_VARS:
    print(f"Variable {var} unique values: {SLEEP_DF[var].unique()}")
# Some individuals have the value -1 for p1160_i2 and p1180_i2, which indicates "Do not know" in UKB coding. We will drop these individuals from the analysis.
unsuitable_eids = SLEEP_DF[SLEEP_DF[SLEEP_VARS].isin([-1]).any(axis=1)]["eid"].tolist()
SLEEP_DF = SLEEP_DF[~SLEEP_DF[SLEEP_VARS].isin([-1]).any(axis=1)]
print(f"Sleep dataset filtered to remove 'Do not know' responses with shape: {SLEEP_DF.shape}")
# Check again the range of values for remaining individuals
for var in SLEEP_VARS:
    print(f"Variable {var} unique values after filtering: {SLEEP_DF[var].unique()}")
# Check to which cluster the excluded individuals belonged
excluded_eids = null_eids + unsuitable_eids
excluded_cluster_info = COHORT_DF[COHORT_DF["eid"].isin(excluded_eids)][["eid", "sfc_external_cluster"]]
print(f"Excluded individuals from sleep dataset due to null or unsuitable values:\n{excluded_cluster_info['sfc_external_cluster'].value_counts()}")
# Now set the correct data type for the variable p1200 (sleeplessness) to categorical, as it is a categorical variable
SLEEP_DF["p1200_i2"] = SLEEP_DF["p1200_i2"].astype("str")
# Now add the sfc_external_cluster column to the sleep dataset for association analysis
SLEEP_DF = SLEEP_DF.merge(COHORT_DF[["eid", "sfc_external_cluster"]], on="eid", how="left")
print(f"Sleep dataset after merging with SFC-EXT clusters: {SLEEP_DF.info()}")
# Append total depression cohort (Cluster 0 + Cluster 1 combined)
dep_df = SLEEP_DF[SLEEP_DF["sfc_external_cluster"].isin(["Cluster 0", "Cluster 1"])].copy()
dep_df["sfc_external_cluster"] = "Depression"
SLEEP_DF = pd.concat([SLEEP_DF, dep_df], ignore_index=True)

# Now ensure the sfc_external_cluster column is categorical with the correct order (correct for histplots)
SLEEP_plots_df = SLEEP_DF.copy() # We will use this copy for plotting, while keeping the original SLEEP_DF for analysis
SLEEP_plots_df["sfc_external_cluster"] = pd.Categorical(
    SLEEP_plots_df["sfc_external_cluster"], categories=["Control", "Depression", "Cluster 0", "Cluster 1"], ordered=True
)
# Now we have a clean sleep dataset with relevant variables for the MDD and control cohort, ready for association analysis with SFC-EXT derived subtypes.

# SOCIAL SUPPORT DATASET 
print("\n[5/6] Processing social support dataset...")
SOCIAL_SUPPORT_DF = pd.read_csv(os.path.join(LIFESTYLE_ENVIRONMENT_DATA_PATH, "social_support.csv"))
print(f"Social support dataset loaded with shape: {SOCIAL_SUPPORT_DF.shape}")
# Get just the variables we need for the analysis
SOCIAL_SUPPORT_VARS = ["p1031_i2", "p2110_i2"] # frequency of friend/family visits, ability to confide in someone close
SOCIAL_SUPPORT_DF = SOCIAL_SUPPORT_DF[["eid"] + SOCIAL_SUPPORT_VARS]
print(f"Social support dataset filtered to relevant variables with shape: {SOCIAL_SUPPORT_DF.shape}")
# Now check if MDD and control people have these variables and that values are non-null
# First ensure both eid columns are of the same type
COHORT_DF["eid"] = COHORT_DF["eid"].astype(str)
SOCIAL_SUPPORT_DF["eid"] = SOCIAL_SUPPORT_DF["eid"].astype(str)
# Now check if the eids in SOCIAL_SUPPORT_DF are in COHORT_DF
SOCIAL_SUPPORT_DF = SOCIAL_SUPPORT_DF[SOCIAL_SUPPORT_DF["eid"].isin(COHORT_DF["eid"])]
print(f"Social support dataset filtered to MDD and control cohort with shape: {SOCIAL_SUPPORT_DF.shape}")
# Check for null-values, if any individual dropped due to null values, show which eid they had and which variable was null
null_eids = SOCIAL_SUPPORT_DF[SOCIAL_SUPPORT_DF[SOCIAL_SUPPORT_VARS].isnull().any(axis=1)]["eid"].tolist()
if null_eids:
    print(f"Warning: The following eids had null values for relevant social support variables and were dropped: {null_eids}")
    SOCIAL_SUPPORT_DF = SOCIAL_SUPPORT_DF.dropna(subset=SOCIAL_SUPPORT_VARS)
    print(f"Social support dataset filtered to remove null values with shape: {SOCIAL_SUPPORT_DF.shape}")
# Now check range of values for remaining individuals
for var in SOCIAL_SUPPORT_VARS:
    print(f"Variable {var} unique values: {SOCIAL_SUPPORT_DF[var].unique()}")
# Some individuals have the values -1 and -3, which indicates "Do not know" and "Prefer not to answer" in UKB coding. We will drop these individuals from the analysis.
unsuitable_eids = SOCIAL_SUPPORT_DF[SOCIAL_SUPPORT_DF[SOCIAL_SUPPORT_VARS].isin([-1, -3]).any(axis=1)]["eid"].tolist()
SOCIAL_SUPPORT_DF = SOCIAL_SUPPORT_DF[~SOCIAL_SUPPORT_DF[SOCIAL_SUPPORT_VARS].isin([-1, -3]).any(axis=1)]
print(f"Social support dataset filtered to remove 'Do not know' and 'Prefer not to answer' responses with shape: {SOCIAL_SUPPORT_DF.shape}")
# Check again the range of values for remaining individuals
for var in SOCIAL_SUPPORT_VARS:
    print(f"Variable {var} unique values after filtering: {SOCIAL_SUPPORT_DF[var].unique()}")
# Check to which cluster the excluded individuals belonged
excluded_eids = null_eids + unsuitable_eids
excluded_cluster_info = COHORT_DF[COHORT_DF["eid"].isin(excluded_eids)][["eid", "sfc_external_cluster"]]
print(f"Excluded individuals from social support dataset due to null or unsuitable values:\n{excluded_cluster_info['sfc_external_cluster'].value_counts()}")
# Now add the sfc_external_cluster column to the social support dataset for association analysis
SOCIAL_SUPPORT_DF = SOCIAL_SUPPORT_DF.merge(COHORT_DF[["eid", "sfc_external_cluster"]], on="eid", how="left")
print(f"Social support dataset after merging with SFC-EXT clusters: {SOCIAL_SUPPORT_DF.info()}")
# Reverse the order of the social support variable (p1031_i2) so that higher values indicate more frequent social support, for easier interpretation in plots
SOCIAL_SUPPORT_DF["p1031_i2"] = SOCIAL_SUPPORT_DF["p1031_i2"].map({
    1: 7,  # "Almost daily" -> 7
    2: 6,  # "Two to four times a week" -> 6
    3: 5,  # "About once a week" -> 5
    4: 4,  # "About once a month" -> 4
    5: 3,  # "Once every few months" -> 3
    6: 2,  # "Never or almost never" -> 2
    7: 1,  # "No friends/family outside household" -> 1
})
# Append total depression cohort (Cluster 0 + Cluster 1 combined)
dep_df = SOCIAL_SUPPORT_DF[SOCIAL_SUPPORT_DF["sfc_external_cluster"].isin(["Cluster 0", "Cluster 1"])].copy()
dep_df["sfc_external_cluster"] = "Depression"
SOCIAL_SUPPORT_DF = pd.concat([SOCIAL_SUPPORT_DF, dep_df], ignore_index=True)
# Now ensure the sfc_external_cluster column is categorical with the correct order (correct for histplots)
SOCIAL_SUPPORT_plots_df = SOCIAL_SUPPORT_DF.copy() # We will use this copy for plotting, while keeping the original SOCIAL_SUPPORT_DF for analysis
SOCIAL_SUPPORT_plots_df["sfc_external_cluster"] = pd.Categorical(
    SOCIAL_SUPPORT_plots_df["sfc_external_cluster"], categories=["Control", "Depression", "Cluster 0", "Cluster 1"], ordered=True
)
# Now we have a clean social support dataset with relevant variables for the MDD and control cohort, ready for association analysis with SFC-EXT derived subtypes.

# SUN EXPOSURE DATASET
print("\n[6/6] Processing sun exposure dataset...")
SUN_EXPOSURE_DF = pd.read_csv(os.path.join(LIFESTYLE_ENVIRONMENT_DATA_PATH, "sun_exposure.csv"))
print(f"Sun exposure dataset loaded with shape: {SUN_EXPOSURE_DF.shape}")
# Get just the variables we need for the analysis
SUN_EXPOSURE_VARS = ["p1050_i2", "p1060_i2"] # time spent outdoors in summer, time spent outdoors in winter
SUN_EXPOSURE_DF = SUN_EXPOSURE_DF[["eid"] + SUN_EXPOSURE_VARS]
print(f"Sun exposure dataset filtered to relevant variables with shape: {SUN_EXPOSURE_DF.shape}")
# Now check if MDD and control people have these variables and that values are non-null
# First ensure both eid columns are of the same type
COHORT_DF["eid"] = COHORT_DF["eid"].astype(str)
SUN_EXPOSURE_DF["eid"] = SUN_EXPOSURE_DF["eid"].astype(str)
# Now check if the eids in SUN_EXPOSURE_DF are in COHORT_DF
SUN_EXPOSURE_DF = SUN_EXPOSURE_DF[SUN_EXPOSURE_DF["eid"].isin(COHORT_DF["eid"])]
print(f"Sun exposure dataset filtered to MDD and control cohort with shape: {SUN_EXPOSURE_DF.shape}")
# Check for null-values, if any individual dropped due to null values, show which eid they had and which variable was null
null_eids = SUN_EXPOSURE_DF[SUN_EXPOSURE_DF[SUN_EXPOSURE_VARS].isnull().any(axis=1)]["eid"].tolist()
if null_eids:
    print(f"Warning: The following eids had null values for relevant sun exposure variables and were dropped: {null_eids}")
    SUN_EXPOSURE_DF = SUN_EXPOSURE_DF.dropna(subset=SUN_EXPOSURE_VARS)
    print(f"Sun exposure dataset filtered to remove null values with shape: {SUN_EXPOSURE_DF.shape}")
# Now check range of values for remaining individuals
for var in SUN_EXPOSURE_VARS:
    print(f"Variable {var} unique values: {SUN_EXPOSURE_DF[var].unique()}")
# Some individuals have the values -1 and -10, which indicates "Do not know" and "Less than an hour a day" in UKB coding. We will drop these individuals from the analysis.
unsuitable_eids = SUN_EXPOSURE_DF[SUN_EXPOSURE_DF[SUN_EXPOSURE_VARS].isin([-1, -10]).any(axis=1)]["eid"].tolist()
SUN_EXPOSURE_DF = SUN_EXPOSURE_DF[~SUN_EXPOSURE_DF[SUN_EXPOSURE_VARS].isin([-1, -10]).any(axis=1)]
print(f"Sun exposure dataset filtered to remove 'Do not know' and 'Less than an hour a day' responses with shape: {SUN_EXPOSURE_DF.shape}")
# Check again the range of values for remaining individuals
for var in SUN_EXPOSURE_VARS:
    print(f"Variable {var} unique values after filtering: {SUN_EXPOSURE_DF[var].unique()}")
# Check to which cluster the excluded individuals belonged
excluded_eids = null_eids + unsuitable_eids
excluded_cluster_info = COHORT_DF[COHORT_DF["eid"].isin(excluded_eids)][["eid", "sfc_external_cluster"]]
print(f"Excluded individuals from sun exposure dataset due to null or unsuitable values:\n{excluded_cluster_info['sfc_external_cluster'].value_counts()}")
# Now add the sfc_external_cluster column to the sun exposure dataset for association analysis
SUN_EXPOSURE_DF = SUN_EXPOSURE_DF.merge(COHORT_DF[["eid", "sfc_external_cluster"]], on="eid", how="left")
print(f"Sun exposure dataset after merging with SFC-EXT clusters: {SUN_EXPOSURE_DF.info()}")
# Append total depression cohort (Cluster 0 + Cluster 1 combined)
dep_df = SUN_EXPOSURE_DF[SUN_EXPOSURE_DF["sfc_external_cluster"].isin(["Cluster 0", "Cluster 1"])].copy()
dep_df["sfc_external_cluster"] = "Depression"
SUN_EXPOSURE_DF = pd.concat([SUN_EXPOSURE_DF, dep_df], ignore_index=True)

# Now ensure the sfc_external_cluster column is categorical with the correct order (correct for histplots)
SUN_EXPOSURE_plots_df = SUN_EXPOSURE_DF.copy() # We will use this copy for plotting, while keeping the original SUN_EXPOSURE_DF for analysis
SUN_EXPOSURE_plots_df["sfc_external_cluster"] = pd.Categorical(
    SUN_EXPOSURE_plots_df["sfc_external_cluster"], categories=["Control", "Depression", "Cluster 0", "Cluster 1"], ordered=True
)
# Now we have a clean sun exposure dataset with relevant variables for the MDD and control cohort, ready for association analysis with SFC-EXT derived subtypes.

# ==============================================================================
# Helper functions for capturing significance and drawing brackets on the right
# ==============================================================================

# Master dictionary to hold all corrected p-values for plotting
master_pvals = {}

def store_pvals(adj_pvals, var_names_with_comp, comparisons_dict):
    """
    Parses the variable names like "p20117_i2 (cluster0_vs_control)" 
    and saves the corrected p-value into the master dictionary.
    """
    for name, p in zip(var_names_with_comp, adj_pvals):
        var = name.split(" (")[0]
        comp_name = name.split(" (")[1].replace(")", "")
        ga, gb = comparisons_dict[comp_name]
        # Store in both directions for easy lookup later
        master_pvals[(var, ga, gb)] = p
        master_pvals[(var, gb, ga)] = p

def get_stars(p):
    """Convert a p-value into significance stars."""
    if pd.isna(p): return "ns"
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"

def draw_right_brackets(ax, y_coords, comparisons, p_values):
    """
    Draws vertical significance brackets to the right of a horizontal plot.
    Adjusts the x-axis limits so brackets are contained within the spines,
    and automatically spaces them to prevent overlap.
    """
    xmin, xmax = ax.get_xlim()
    x_range = xmax - xmin
    
    # Start placing brackets a bit past the maximum data value
    x_start = xmax + x_range * 0.03
    x_step = x_range * 0.08  # Distance between distinct bracket levels
    tip = x_range * 0.015    # Length of the inward-pointing tips
    
    # Filter to valid comparisons that exist in the y_coords dictionary
    valid_comps = [(g1, g2, p) for (g1, g2), p in zip(comparisons, p_values) 
                   if g1 in y_coords and g2 in y_coords]
    
    # Sort by distance between groups (adjacent groups first) to nest the brackets nicely
    valid_comps.sort(key=lambda item: abs(y_coords[item[0]] - y_coords[item[1]]))
    
    max_x_needed = xmax
    
    for i, (g1, g2, p) in enumerate(valid_comps):
        y1, y2 = y_coords[g1], y_coords[g2]
        text = get_stars(p)
        
        # Each bracket gets its own dedicated column (offset level)
        x_pos = x_start + i * x_step
        
        # 1. Draw the bracket line (using data coordinates, so it's inside the spines)
        ax.plot([x_pos - tip, x_pos, x_pos, x_pos - tip], 
                [y1, y1, y2, y2], 
                color='black', lw=1.2)
        
        # 2. Add the stars text (slightly to the right of the vertical line)
        text_x = x_pos + x_range * 0.015
        ax.text(text_x, (y1 + y2) / 2, text, 
                ha='left', va='center', color='black', weight='bold', fontsize=12)
        
        # Estimate rightmost boundary needed for the text (approx 8% of x_range for "***")
        max_x_needed = max(max_x_needed, text_x + x_range * 0.08)
        
    # Expand the right spine to safely enclose all brackets and text
    ax.set_xlim(xmin, max_x_needed)

# ==============================================================================
# Print out 25th, 50th, and 75th percentiles for each variable in each dataset
# per group (Control, Depression, Cluster 0, Cluster 1) 
# Also print percentage of each group for each categorical variable level
# ==============================================================================
def print_percentiles(df, variables, dataset_name):
    print(f"\nPercentiles for {dataset_name} dataset:")
    for var in variables:
        if var in df.columns:
            print(f"\nVariable: {var}")
            percentiles = df.groupby("sfc_external_cluster")[var].quantile([0.25, 0.5, 0.75]).unstack()
            percentiles.columns = ['25th Percentile', '50th Percentile (Median)', '75th Percentile']
            print(percentiles)

def print_categorical_percentages(df, categorical_vars, dataset_name):
    print(f"\nCategorical variable percentages for {dataset_name} dataset:")
    for var in categorical_vars:
        if var in df.columns:
            print(f"\nVariable: {var}")
            percentages = df.groupby("sfc_external_cluster")[var].value_counts(normalize=True).unstack() * 100
            print(percentages)

# ALCOHOL DATASET
# Exclude categorical variable alcohol drinker status p20117_i2 
ALCOHOL_VARS_CONTINUOUS = [var for var in ALCOHOL_VARS if var != "p20117_i2"]
print_percentiles(ALCOHOL_DF, ALCOHOL_VARS_CONTINUOUS, "Alcohol")
print_categorical_percentages(ALCOHOL_DF, [var for var in ALCOHOL_VARS if var == "p20117_i2"], "Alcohol")

# NMR METABOLOMICS DATASET
print_percentiles(NMR_METABOLOMICS_DF, NMR_METABOLOMICS_VARS + ["insulin_resistance_proxy"], "NMR Metabolomics")

# PHYSICAL ACTIVITY DATASET
print_percentiles(PHYSICAL_ACTIVITY_DF, PHYSICAL_ACTIVITY_VARS, "Physical Activity")

# SLEEP DATASET
# Exclude categorical variables sleeplessness/insomnia p1200_i2
SLEEP_VARS_CONTINUOUS = [var for var in SLEEP_VARS if var != "p1200_i2"]
print_percentiles(SLEEP_DF, SLEEP_VARS_CONTINUOUS, "Sleep")
print_categorical_percentages(SLEEP_DF, [var for var in SLEEP_VARS if var == "p1200_i2"], "Sleep")

# SOCIAL SUPPORT DATASET
print_percentiles(SOCIAL_SUPPORT_DF, SOCIAL_SUPPORT_VARS, "Social Support")

# SUN EXPOSURE DATASET
print_percentiles(SUN_EXPOSURE_DF, SUN_EXPOSURE_VARS, "Sun Exposure")

# ==============================================================================
# Formally test the associations between each lifestyle/environment variables and 
# control group and SFC-EXT derived clusters
# ==============================================================================
def _chi2_test(df, x_col, group_a, group_b, value_col):
    if value_col not in df.columns:
        return np.nan, np.nan
    df_sub = df[df[x_col].isin([group_a, group_b])].copy()
    if df_sub.empty:
        return np.nan, np.nan
    table = pd.crosstab(df_sub[x_col], df_sub[value_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan
    if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
        return np.nan, np.nan
    try:
        stat, pval, _, _ = chi2_contingency(table.values)
        return float(stat), float(pval)
    except Exception:
        return np.nan, np.nan
    
def _mannwhitney_test(df, x_col, group_a, group_b, value_col):
    if value_col not in df.columns:
        return np.nan, np.nan
    vals_a = df.loc[df[x_col] == group_a, value_col].dropna()
    vals_b = df.loc[df[x_col] == group_b, value_col].dropna()
    if vals_a.empty or vals_b.empty:
        return np.nan, np.nan
    try:
        stat, pval = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
        return float(stat), float(pval)
    except Exception:
        return np.nan, np.nan
    
def _append_to_text_log(log_path: Optional[str], block: str) -> None:
    if not log_path:
        return
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(block)
            if not block.endswith("\n"):
                f.write("\n")
    except Exception:
        pass

def apply_multiple_testing_correction(
    p_values: List[float],
    variable_names: List[str],
    test_methods: List[str],
    method: Literal['fdr_bh', 'bonferroni'] = 'fdr_bh',
    alpha: float = 0.05,
    log_path: Optional[str] = None,
    log_context: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    
    if not (len(p_values) == len(variable_names) == len(test_methods)):
        raise ValueError("Lengths of p_values, variable_names, and test_methods must match")
    if method == 'fdr_bh':
        reject, pvals_corrected = multitest.fdrcorrection(p_values, alpha=alpha)
        correction_name = 'FDR (Benjamini-Hochberg)'
    elif method == 'bonferroni':
        reject, pvals_corrected, _, _ = multitest.multipletests(
            p_values, alpha=alpha, method='bonferroni'
        )
        correction_name = 'Bonferroni'
    else:
        raise ValueError(f"Unknown method: {method}")

    lines: List[str] = []
    lines.append("\n" + "=" * 80)
    lines.append(f"MULTIPLE TESTING CORRECTION: {correction_name}")
    if log_context:
        lines.append(f"Context: {log_context}")
    lines.append("=" * 80)
    lines.append(f"\n{'Variable':<40} {'Test':<20} {'p (raw)':<12} {'p (adj)':<12} {'Sig.':<5}")
    lines.append("-" * 90)

    for i, var in enumerate(variable_names):
        sig_marker = "***" if pvals_corrected[i] < 0.001 else \
                     "**" if pvals_corrected[i] < 0.01 else \
                     "*" if pvals_corrected[i] < 0.05 else "n.s."

        lines.append(
            f"{var:<40} {test_methods[i]:<20} {p_values[i]:<12.6f} "
            f"{pvals_corrected[i]:<12.6f} {sig_marker:<5}"
        )

    lines.append("-" * 90)
    lines.append(f"Significant results: {reject.sum()} / {len(p_values)}")
    lines.append("=" * 80)

    _append_to_text_log(log_path, "\n".join(lines))

    return reject, pvals_corrected

comparisons_cluster = {
    'depression_vs_control': ('Depression', 'Control'),
    'cluster0_vs_control': ('Cluster 0', 'Control'),
    'cluster1_vs_control': ('Cluster 1', 'Control'),
    'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
}

# The list formatting for plotting the brackets
plot_comparisons = [('Control', 'Depression'), ('Control', 'Cluster 0'), ('Control', 'Cluster 1'), ('Cluster 0', 'Cluster 1')]


# ALCOHOL DATASET
alcohol_p_values = []
alcohol_variable_names = []
alcohol_test_methods = []
for var in ALCOHOL_VARS:
    if var == "p20117_i2":
        test_method = "Chi-square"
    else:
        test_method = "Mann-Whitney U"
    for comparison_name, (group_a, group_b) in comparisons_cluster.items():
        if var == "p20117_i2":
            _, p = _chi2_test(ALCOHOL_DF, 'sfc_external_cluster', group_a, group_b, var)
        else:
            _, p = _mannwhitney_test(ALCOHOL_DF, 'sfc_external_cluster', group_a, group_b, var)
        alcohol_p_values.append(p)
        alcohol_variable_names.append(f"{var} ({comparison_name})")
        alcohol_test_methods.append(test_method)

_, alc_p_adj = apply_multiple_testing_correction(
    p_values=alcohol_p_values,
    variable_names=alcohol_variable_names,
    test_methods=alcohol_test_methods,
    log_path=LOG_PATH,
)
store_pvals(alc_p_adj, alcohol_variable_names, comparisons_cluster)


# NMR METABOLOMICS DATASET
nmr_p_values = []
nmr_variable_names = []
nmr_test_methods = []
for var in (NMR_METABOLOMICS_VARS + ["insulin_resistance_proxy"]):
    for comparison_name, (group_a, group_b) in comparisons_cluster.items():
        stat, p = _mannwhitney_test(NMR_METABOLOMICS_DF, 'sfc_external_cluster', group_a, group_b, var)
        nmr_p_values.append(p)
        nmr_variable_names.append(f"{var} ({comparison_name})")
        nmr_test_methods.append("Mann-Whitney U")

_, nmr_p_adj = apply_multiple_testing_correction(
    p_values=nmr_p_values,
    variable_names=nmr_variable_names,
    test_methods=nmr_test_methods,
    log_path=LOG_PATH,
)
store_pvals(nmr_p_adj, nmr_variable_names, comparisons_cluster)


# PHYSICAL ACTIVITY DATASET
physical_activity_p_values = []
physical_activity_variable_names = []
physical_activity_test_methods = []
for var in PHYSICAL_ACTIVITY_VARS:
    for comparison_name, (group_a, group_b) in comparisons_cluster.items():
        stat, p = _mannwhitney_test(PHYSICAL_ACTIVITY_DF, 'sfc_external_cluster', group_a, group_b, var)
        physical_activity_p_values.append(p)
        physical_activity_variable_names.append(f"{var} ({comparison_name})")
        physical_activity_test_methods.append("Mann-Whitney U")

_, pa_p_adj = apply_multiple_testing_correction(
    p_values=physical_activity_p_values,
    variable_names=physical_activity_variable_names,
    test_methods=physical_activity_test_methods,
    log_path=LOG_PATH,
)
store_pvals(pa_p_adj, physical_activity_variable_names, comparisons_cluster)


# SLEEP DATASET
sleep_p_values = []
sleep_variable_names = []
sleep_test_methods = []
for var in SLEEP_VARS:
    if var == "p1200_i2":
        test_method = "Chi-square"
    else:
        test_method = "Mann-Whitney U"
    for comparison_name, (group_a, group_b) in comparisons_cluster.items():
        if var == "p1200_i2":
            _, p = _chi2_test(SLEEP_DF, 'sfc_external_cluster', group_a, group_b, var)
        else:
            _, p = _mannwhitney_test(SLEEP_DF, 'sfc_external_cluster', group_a, group_b, var)
        sleep_p_values.append(p)
        sleep_variable_names.append(f"{var} ({comparison_name})")
        sleep_test_methods.append(test_method)

_, sleep_p_adj = apply_multiple_testing_correction(
    p_values=sleep_p_values,
    variable_names=sleep_variable_names,
    test_methods=sleep_test_methods,
    log_path=LOG_PATH,
)
store_pvals(sleep_p_adj, sleep_variable_names, comparisons_cluster)


# SOCIAL SUPPORT DATASET
social_support_p_values = []
social_support_variable_names = []
social_support_test_methods = []
for var in SOCIAL_SUPPORT_VARS:
    for comparison_name, (group_a, group_b) in comparisons_cluster.items():
        stat, p = _mannwhitney_test(SOCIAL_SUPPORT_DF, 'sfc_external_cluster', group_a, group_b, var)
        social_support_p_values.append(p)
        social_support_variable_names.append(f"{var} ({comparison_name})")
        social_support_test_methods.append("Mann-Whitney U")

_, ss_p_adj = apply_multiple_testing_correction(
    p_values=social_support_p_values,
    variable_names=social_support_variable_names,
    test_methods=social_support_test_methods,
    log_path=LOG_PATH,
)
store_pvals(ss_p_adj, social_support_variable_names, comparisons_cluster)


# SUN EXPOSURE DATASET
sun_exposure_p_values = []
sun_exposure_variable_names = []
sun_exposure_test_methods = []
for var in SUN_EXPOSURE_VARS:
    for comparison_name, (group_a, group_b) in comparisons_cluster.items():
        stat, p = _mannwhitney_test(SUN_EXPOSURE_DF, 'sfc_external_cluster', group_a, group_b, var)
        sun_exposure_p_values.append(p)
        sun_exposure_variable_names.append(f"{var} ({comparison_name})")
        sun_exposure_test_methods.append("Mann-Whitney U")

_, sun_p_adj = apply_multiple_testing_correction(
    p_values=sun_exposure_p_values,
    variable_names=sun_exposure_variable_names,
    test_methods=sun_exposure_test_methods,
    log_path=LOG_PATH,
)
store_pvals(sun_p_adj, sun_exposure_variable_names, comparisons_cluster)


# ==============================================================================
# Visualize the distribution of each lifestyle/environment variable across the
# control group and SFC-EXT derived clusters (with significance highlighted)
# ==============================================================================
# Control and SFC-EXT cluster palette colors
group_colors = {
    "Control": "#2ca02c",
    "Depression": "#6a3d9a",
    "Cluster 0": "#0026ff", 
    "Cluster 1": "#fd7600", 
}
group_order = ["Control", "Depression", "Cluster 0", "Cluster 1"]
palette_list = [group_colors[g] for g in group_order]


# ALCOHOL DATASET
for var in ALCOHOL_VARS:
    # Retrieve corrected p-values from master dictionary for this specific variable
    current_pvals = [master_pvals.get((var, g1, g2), np.nan) for g1, g2 in plot_comparisons]

    if var == "p20117_i2": 
        title_var = "alcohol drinker status"

        plt.figure(figsize=(8, 6))
        ax = sns.histplot(
            y="sfc_external_cluster",
            hue=var,
            hue_order=["0.0", "1.0", "2.0"], 
            data=ALCOHOL_plots_df,
            multiple="fill",
            stat="percent",
            palette="Accent",
            shrink=0.8, 
            legend=True, 
        )

        for p in ax.patches:
            width = p.get_width()
            if width > 0.05:
                x_center = p.get_x() + width / 2
                y_center = p.get_y() + p.get_height() / 2
                ax.text(
                    x_center, y_center, f"{width * 100:.1f}%",
                    ha="center", va="center", color="black", weight="bold", fontsize=8,
                )

        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        legend_labels = {"0.0": "Never", "1.0": "Previous", "2.0": "Current"}
        legend = ax.get_legend()
        handles, labels = legend.legend_handles, [text.get_text() for text in legend.get_texts()]
        new_labels = [legend_labels.get(label, label) for label in labels]
        ax.legend(handles, new_labels, title="Alcohol Drinker Status", loc="upper right")

        # Get actual y-coordinates drawn by sns.histplot
        plt.draw() 
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        y_coords = {lbl: i for i, lbl in enumerate(y_labels)}
        
        draw_right_brackets(ax, y_coords, plot_comparisons, current_pvals)

        plt.title(f"Distribution of {title_var}")
        plt.xlabel("Percentage")
        plt.ylabel("Group")
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{title_var}.svg"), dpi=300)

    else: 
        title_var = "alcohol intake frequency"
        title_x = "Alcohol intake frequency (1=never, 6=daily or almost daily)"

        plt.figure(figsize=(8, 6))
        ax = pt.RainCloud(
            x="sfc_external_cluster",
            y=var,
            hue="sfc_external_cluster",
            order=group_order,
            data=ALCOHOL_plots_df,
            palette=palette_list,
            move=0.25, 
            cut=0, 
            orient="h",
            linewidth=0, # remove the line around the half-violin shape
            box_linewidth=1.5,
            width_viol=0 # remove the half-violin shape
        )

        y_coords = {g: i for i, g in enumerate(group_order)}
        draw_right_brackets(ax, y_coords, plot_comparisons, current_pvals)

        plt.title(f"Distribution of {title_var}")
        plt.xlabel(title_x)
        plt.ylabel("Group")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{title_var}.svg"), dpi=300)


# NMR METABOLOMICS DATASET
for var in (NMR_METABOLOMICS_VARS + ["insulin_resistance_proxy"]): 
    current_pvals = [master_pvals.get((var, g1, g2), np.nan) for g1, g2 in plot_comparisons]

    if var == "insulin_resistance_proxy":
        title_var = "insulin resistance marker"
        title_x = "Insulin resistance marker (TG:HDL-C)"
    elif var == "p23400_i0":
        title_var = "total cholesterol"
        title_x = "Total cholesterol (mmol/L)"
    elif var == "p23407_i0":
        title_var = "total triglycerides"
        title_x = "Total triglycerides (mmol/L)"
    elif var == "p23406_i0":
        title_var = "HDL cholesterol"
        title_x = "HDL cholesterol (mmol/L)"
    elif var == "p23405_i0":
        title_var = "LDL cholesterol"
        title_x = "LDL cholesterol (mmol/L)"

    plt.figure(figsize=(8, 6))
    ax = pt.RainCloud(
        x="sfc_external_cluster",
        y=var,
        hue="sfc_external_cluster",
        order=group_order,
        data=NMR_METABOLOMICS_plots_df,
        palette=palette_list,
        move=0.25, 
        cut=0, 
        orient="h",
        linewidth=1.5,
        box_linewidth=1.5,
    )
    
    y_coords = {g: i for i, g in enumerate(group_order)}
    draw_right_brackets(ax, y_coords, plot_comparisons, current_pvals)

    plt.title(f"Distribution of {title_var}")
    plt.xlabel(title_x)
    plt.ylabel("Group")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{title_var}.svg"), dpi=300)


# PHYSICAL ACTIVITY DATASET
for var in PHYSICAL_ACTIVITY_VARS:
    current_pvals = [master_pvals.get((var, g1, g2), np.nan) for g1, g2 in plot_comparisons]

    if var == "p22040_i2":
        title_var = "total physical activity"
        title_x = "Total physical activity (MET-minutes/week for all activities)"
    elif var == "p22037_i2":
        title_var = "walking physical activity"
        title_x = "Walking physical activity (MET-minutes/week)"
    elif var == "p22038_i2":
        title_var = "moderate physical activity"
        title_x = "Moderate physical activity (MET-minutes/week)"
    elif var == "p22039_i2":
        title_var = "vigorous physical activity"
        title_x = "Vigorous physical activity (MET-minutes/week)"

    plt.figure(figsize=(8, 6))
    ax = pt.RainCloud(
        x="sfc_external_cluster",
        y=var,
        hue="sfc_external_cluster",
        order=group_order,
        data=PHYSICAL_ACTIVITY_plots_df,
        palette=palette_list,
        move=0.25, 
        cut=0, 
        orient="h",
        linewidth=1.5,
        box_linewidth=1.5,
    )

    y_coords = {g: i for i, g in enumerate(group_order)}
    draw_right_brackets(ax, y_coords, plot_comparisons, current_pvals)

    plt.title(f"Distribution of {title_var}")
    plt.xlabel(title_x)
    plt.ylabel("Group")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{title_var}.svg"), dpi=300)


# SLEEP DATASET
for var in SLEEP_VARS:
    current_pvals = [master_pvals.get((var, g1, g2), np.nan) for g1, g2 in plot_comparisons]

    if var == "p1200_i2":
        title_var = "sleeplessness"

        plt.figure(figsize=(8, 6))
        ax = sns.histplot(
            y="sfc_external_cluster",
            hue=var,
            hue_order=["1.0", "2.0", "3.0"], 
            data=SLEEP_plots_df,
            multiple="fill",
            stat="percent",
            palette="Accent",
            shrink=0.8, 
            legend=True, 
        )

        for p in ax.patches:
            width = p.get_width()
            if width > 0.05:
                x_center = p.get_x() + width / 2
                y_center = p.get_y() + p.get_height() / 2
                ax.text(
                    x_center, y_center, f"{width * 100:.1f}%",
                    ha="center", va="center", color="black", weight="bold", fontsize=8,
                )

        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        legend_labels = {"1.0": "Never/rarely", "2.0": "Sometimes", "3.0": "Usually"}
        legend = ax.get_legend()
        handles, labels = legend.legend_handles, [text.get_text() for text in legend.get_texts()]
        new_labels = [legend_labels.get(label, label) for label in labels]
        ax.legend(handles, new_labels, title="Sleeplessness/Insomnia", loc="upper right")
        
        plt.draw() 
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        y_coords = {lbl: i for i, lbl in enumerate(y_labels)}
        draw_right_brackets(ax, y_coords, plot_comparisons, current_pvals)

        plt.title(f"Distribution of {title_var}")
        plt.xlabel("Percentage")
        plt.ylabel("Group")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{title_var}.svg"), dpi=300)

    else: 
        if var == "p1160_i2":
            title_var = "sleep duration"
            title_x = "Sleep duration (hours/day)"
        elif var == "p1180_i2":
            title_var = "chronotype"
            title_x = "Chronotype (1=morning, 4=evening)"
        
        plt.figure(figsize=(8, 6))
        ax = pt.RainCloud(
            x="sfc_external_cluster",
            y=var,
            hue="sfc_external_cluster",
            order=group_order,
            data=SLEEP_plots_df,
            palette=palette_list,
            move=0.25, 
            cut=0, 
            orient="h",
            linewidth=0, # remove the line around the half-violin shape
            box_linewidth=1.5,
            width_viol=0 # remove the half-violin shape
        )

        y_coords = {g: i for i, g in enumerate(group_order)}
        draw_right_brackets(ax, y_coords, plot_comparisons, current_pvals)

        plt.title(f"Distribution of {title_var}")
        plt.xlabel(title_x)
        plt.ylabel("Group")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{title_var}.svg"), dpi=300)


# SOCIAL SUPPORT DATASET
for var in SOCIAL_SUPPORT_VARS:
    current_pvals = [master_pvals.get((var, g1, g2), np.nan) for g1, g2 in plot_comparisons]
    if var == "p1031_i2":
        title_var = "frequency of social visits"
        title_x = "Frequency of friend/family visits (1=no friends/family outside household, 7=almost daily)"
    elif var == "p2110_i2":
        title_var = "ability to confide in someone close"
        title_x = "Ability to confide in someone close (0=never or almost never, 5=almost daily)"

    plt.figure(figsize=(8, 6))
    ax = pt.RainCloud(
        x="sfc_external_cluster",
        y=var,
        hue="sfc_external_cluster",
        order=group_order,
        data=SOCIAL_SUPPORT_plots_df,
        palette=palette_list,
        move=0.25, 
        cut=0, 
        orient="h",
        linewidth=0, # remove the line around the half-violin shape
        box_linewidth=1.5,
        width_viol=0 # remove the half-violin shape
    )

    y_coords = {g: i for i, g in enumerate(group_order)}
    draw_right_brackets(ax, y_coords, plot_comparisons, current_pvals)

    plt.title(f"Distribution of {title_var}")
    plt.xlabel(title_x)
    plt.ylabel("Group")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{title_var}.svg"), dpi=300)


# SUN EXPOSURE DATASET
for var in SUN_EXPOSURE_VARS:
    current_pvals = [master_pvals.get((var, g1, g2), np.nan) for g1, g2 in plot_comparisons]
    if var == "p1050_i2":
        title_var = "sun exposure in summer"
        title_x = "Time spent outdoors in summer (hours/day)"
    elif var == "p1060_i2":
        title_var = "sun exposure in winter"
        title_x = "Time spent outdoors in winter (hours/day)"

    plt.figure(figsize=(8, 6))
    ax = pt.RainCloud(
        x="sfc_external_cluster",
        y=var,
        hue="sfc_external_cluster",
        order=group_order,
        data=SUN_EXPOSURE_plots_df,
        palette=palette_list,
        move=0.25, 
        cut=0, 
        orient="h",
        linewidth=0, # remove the line around the half-violin shape
        box_linewidth=1.5,
        width_viol=0 # remove the half-violin shape
    )

    y_coords = {g: i for i, g in enumerate(group_order)}
    draw_right_brackets(ax, y_coords, plot_comparisons, current_pvals)

    plt.title(f"Distribution of {title_var}")
    plt.xlabel(title_x)
    plt.ylabel("Group")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{title_var}.svg"), dpi=300)

