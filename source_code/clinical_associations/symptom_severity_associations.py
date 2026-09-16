""" MDD (subclinical) symptom and symptom severity associations with SFC-EXT derived MDD subtypes
"""
import os
import pandas as pd
import ptitprince as pt
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chi2_contingency
import numpy as np
from typing import List, Tuple, Optional, Literal
from statsmodels.stats import multitest

# ==============================================================================
# Configuration
# ==============================================================================
GENERAL_DATA_PATH = ".../data/UKB"
COHORT_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "cohorts")
SYMPTOMS_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "mental_health")
LOG_PATH = os.path.join(SYMPTOMS_DATA_PATH, "symptom_severity_associations_log.txt")
PLOTS_DIR = ".../reports/plots/symptom_severity_associations"

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

# SYMPTOMS DATASET
MENTAL_HEALTH_DF = pd.read_csv(os.path.join(SYMPTOMS_DATA_PATH, "mental_health.csv"), sep = ";")
print(f"Mental health dataset loaded with shape: {MENTAL_HEALTH_DF.shape}")
# Get just the variables we need for the analysis
SYMPTOM_VARS = ["p2030_i2", "p2050_i2", "p2060_i2", "p2070_i2", "p2080_i2"] # guilty feelings, freq. of depressed mood in last 2 weeks, freq. of unenthusiasm/disinterest in last 2 weeks, freq. of tenseness/restlessness in last 2 weeks, freq. of tiredness/lethargy in last 2 weeks 
SYMPTOMS_DF = MENTAL_HEALTH_DF[["eid"] + SYMPTOM_VARS].copy()
print(f"Symptom dataset filtered to relevant variables with shape: {SYMPTOMS_DF.shape}")
# Now check if MDD and control people have these variables and that values are non-null
# First ensure both eid columns are of the same type
COHORT_DF["eid"] = COHORT_DF["eid"].astype(str)
SYMPTOMS_DF["eid"] = SYMPTOMS_DF["eid"].astype(str)
# Now check if the eids in SYMPTOMS_DF are in COHORT_DF
SYMPTOMS_DF = SYMPTOMS_DF[SYMPTOMS_DF["eid"].isin(COHORT_DF["eid"])]
print(f"Symptom dataset filtered to MDD and control cohort with shape: {SYMPTOMS_DF.shape}")
# Check for null-values, if any individual dropped due to null values, show which eid they had and which variable was null
null_eids = SYMPTOMS_DF[SYMPTOMS_DF[SYMPTOM_VARS].isnull().any(axis=1)]["eid"].tolist()
if null_eids:
    print(f"Warning: The following eids had null values for relevant symptom variables and were dropped: {null_eids}")
    SYMPTOMS_DF = SYMPTOMS_DF.dropna(subset=SYMPTOM_VARS)
    print(f"Symptom dataset filtered to remove null values with shape: {SYMPTOMS_DF.shape}")
# Now check unique values for remaining individuals
for var in SYMPTOM_VARS:
    print(f"Variable {var} unique values: {SYMPTOMS_DF[var].unique()})")
# All variables show the values -3 and -1, which indicate "Prefer not to answer" and "Do not know" in UKB coding. We will drop these individuals from the analysis.
unsuitable_eids = SYMPTOMS_DF[SYMPTOMS_DF[SYMPTOM_VARS].isin([-3, -1]).any(axis=1)]["eid"].tolist()
SYMPTOMS_DF = SYMPTOMS_DF[~SYMPTOMS_DF[SYMPTOM_VARS].isin([-3, -1]).any(axis=1)]
print(f"Symptom dataset filtered to remove 'Prefer not to answer' and 'Do not know' responses with shape: {SYMPTOMS_DF.shape}")
# Check again the range of values for remaining individuals
for var in SYMPTOM_VARS:
    print(f"Variable {var} unique values: {SYMPTOMS_DF[var].unique()}")
# Check to which cluster the excluded individuals belonged
excluded_eids = null_eids + unsuitable_eids
excluded_cluster_info = COHORT_DF[COHORT_DF["eid"].isin(excluded_eids)][["eid", "sfc_external_cluster"]]
print(f"Excluded individuals from symptom dataset due to null or unsuitable values:\n{excluded_cluster_info['sfc_external_cluster'].value_counts()}")
# Now set the correct data type for the variable p2030_i2 (guilty feelings) to categorical, as it is a categorical variable
SYMPTOMS_DF["p2030_i2"] = SYMPTOMS_DF["p2030_i2"].astype("str")
# Now add the sfc_external_cluster column to the symptom dataset for association analysis
SYMPTOMS_DF = SYMPTOMS_DF.merge(COHORT_DF[["eid", "sfc_external_cluster"]], on="eid", how="left")
print(f"Symptom dataset after merging with SFC-EXT clusters: {SYMPTOMS_DF.info()}")
# Append total depression cohort (Cluster 0 + Cluster 1 combined)
dep_df = SYMPTOMS_DF[SYMPTOMS_DF["sfc_external_cluster"].isin(["Cluster 0", "Cluster 1"])].copy()
dep_df["sfc_external_cluster"] = "Depression"
SYMPTOMS_DF = pd.concat([SYMPTOMS_DF, dep_df], ignore_index=True)
# Now ensure the sfc_external_cluster column is categorical with the correct order (correct for histplots)
SYMPTOMS_plots_df = SYMPTOMS_DF.copy() # We will use this copy for plotting, while keeping the original SYMPTOMS_DF for analysis
SYMPTOMS_plots_df["sfc_external_cluster"] = pd.Categorical(
    SYMPTOMS_plots_df["sfc_external_cluster"], categories=["Control", "Depression", "Cluster 0", "Cluster 1"], ordered=True
)
# Now we have a clean symptom dataset with relevant variables for the MDD and control cohort, ready for association analysis with SFC-EXT derived subtypes.

# SYMPTOM SEVERITY DATASET
MENTAL_HEALTH_DF = pd.read_csv(os.path.join(SYMPTOMS_DATA_PATH, "mental_health.csv"), sep = ";")
print(f"Mental health dataset loaded with shape: {MENTAL_HEALTH_DF.shape}")
# Get just the variables we need for the analysis
SYMPTOM_SEVERITY_VARS = ["p4609_i2", "p4620_i2", "p5375_i2", "p5386_i2"] # longest period of depression episode, number of depression episodes, longest period of unenthusiasm/disinterest episode, number of unenthusiasm/disinterest episodes
SYMPTOMS_SEVERITY_DF = MENTAL_HEALTH_DF[["eid"] + SYMPTOM_SEVERITY_VARS].copy()
print(f"Symptom severity dataset filtered to relevant variables with shape: {SYMPTOMS_SEVERITY_DF.shape}")
# Now check if MDD and control people have these variables and that values are non-null
# First ensure both eid columns are of the same type
COHORT_DF["eid"] = COHORT_DF["eid"].astype(str)
SYMPTOMS_SEVERITY_DF["eid"] = SYMPTOMS_SEVERITY_DF["eid"].astype(str)
# Now check if the eids in SYMPTOMS_SEVERITY_DF are in COHORT_DF
SYMPTOMS_SEVERITY_DF = SYMPTOMS_SEVERITY_DF[SYMPTOMS_SEVERITY_DF["eid"].isin(COHORT_DF["eid"])]
print(f"Symptom severity dataset filtered to MDD and control cohort with shape: {SYMPTOMS_SEVERITY_DF.shape}")
# Check for null-values, if any individual dropped due to null values, show which eid they had and which variable was null
null_eids = SYMPTOMS_SEVERITY_DF[SYMPTOMS_SEVERITY_DF[SYMPTOM_SEVERITY_VARS].isnull().any(axis=1)]["eid"].tolist()
if null_eids:
    print(f"Warning: The following eids had null values for relevant symptom variables and were dropped: {null_eids}")
    SYMPTOMS_SEVERITY_DF = SYMPTOMS_SEVERITY_DF.dropna(subset=SYMPTOM_SEVERITY_VARS)
    print(f"Symptom severity dataset filtered to remove null values with shape: {SYMPTOMS_SEVERITY_DF.shape}")
# Now check unique values for remaining individuals
for var in SYMPTOM_SEVERITY_VARS:
    print(f"Variable {var} unique values: {SYMPTOMS_SEVERITY_DF[var].unique()})")
# All variables show the values -3 and -1, which indicate "Prefer not to answer" and "Do not know" in UKB coding. We will drop these individuals from the analysis.
unsuitable_eids = SYMPTOMS_SEVERITY_DF[SYMPTOMS_SEVERITY_DF[SYMPTOM_SEVERITY_VARS].isin([-3, -1]).any(axis=1)]["eid"].tolist()
SYMPTOMS_SEVERITY_DF = SYMPTOMS_SEVERITY_DF[~SYMPTOMS_SEVERITY_DF[SYMPTOM_SEVERITY_VARS].isin([-3, -1]).any(axis=1)]
print(f"Symptom severity dataset filtered to remove 'Prefer not to answer' and 'Do not know' responses with shape: {SYMPTOMS_SEVERITY_DF.shape}")
# Check again the range of values for remaining individuals
for var in SYMPTOM_SEVERITY_VARS:
    print(f"Variable {var} unique values: {SYMPTOMS_SEVERITY_DF[var].unique()}")
# Check to which cluster the excluded individuals belonged
excluded_eids = null_eids + unsuitable_eids
excluded_cluster_info = COHORT_DF[COHORT_DF["eid"].isin(excluded_eids)][["eid", "sfc_external_cluster"]]
print(f"Excluded individuals from symptom severity dataset due to null or unsuitable values:\n{excluded_cluster_info['sfc_external_cluster'].value_counts()}")
# Now add the sfc_external_cluster column to the symptom severity dataset for association analysis
SYMPTOMS_SEVERITY_DF = SYMPTOMS_SEVERITY_DF.merge(COHORT_DF[["eid", "sfc_external_cluster"]], on="eid", how="left")
print(f"Symptom severity dataset after merging with SFC-EXT clusters: {SYMPTOMS_SEVERITY_DF.info()}")
# Append total depression cohort (Cluster 0 + Cluster 1 combined)
dep_df = SYMPTOMS_SEVERITY_DF[SYMPTOMS_SEVERITY_DF["sfc_external_cluster"].isin(["Cluster 0", "Cluster 1"])].copy()
dep_df["sfc_external_cluster"] = "Depression"
SYMPTOMS_SEVERITY_DF = pd.concat([SYMPTOMS_SEVERITY_DF, dep_df], ignore_index=True)
# Now ensure the sfc_external_cluster column is categorical with the correct order (correct for histplots)
SYMPTOMS_SEVERITY_plots_df = SYMPTOMS_SEVERITY_DF.copy() # We will use this copy for plotting, while keeping the original SYMPTOMS_SEVERITY_DF for analysis
SYMPTOMS_SEVERITY_plots_df["sfc_external_cluster"] = pd.Categorical(
    SYMPTOMS_SEVERITY_plots_df["sfc_external_cluster"], categories=["Control", "Depression", "Cluster 0", "Cluster 1"], ordered=True
)
# Now we have a clean symptom severity dataset with relevant variables for the MDD and control cohort, ready for association analysis with SFC-EXT derived subtypes.

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
# ==============================================================================
def print_percentiles(df, variables, dataset_name):
    print(f"\nPercentiles for {dataset_name} dataset:")
    for var in variables:
        if var in df.columns:
            print(f"\nVariable: {var}")
            percentiles = df.groupby("sfc_external_cluster")[var].quantile([0.25, 0.5, 0.75]).unstack()
            percentiles.columns = ['25th Percentile', '50th Percentile (Median)', '75th Percentile']
            print(percentiles)

# SYMPTOMS DATASET
# Exclude categorical variable p2030_i2 (guilty feelings) from percentiles
SYMPTOM_VARS_CONTINUOUS = [var for var in SYMPTOM_VARS if var != "p2030_i2"]
print_percentiles(SYMPTOMS_DF, SYMPTOM_VARS_CONTINUOUS, "MDD Symptoms")

# SYMPTOM SEVERITY DATASET
print_percentiles(SYMPTOMS_SEVERITY_DF, SYMPTOM_SEVERITY_VARS, "MDD Symptoms Severity")

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


# SYMPTOMS DATASET
symptom_p_values = []
symptom_variable_names = []
symptom_test_methods = []
for var in SYMPTOM_VARS:
    if var == "p2030_i2":  # Categorical variable (guilty feelings)
        test_method = "Chi-square"
    else:   
        test_method = "Mann-Whitney U"
    for comparison_name, (group_a, group_b) in comparisons_cluster.items():
        if var == "p2030_i2":  # Categorical variable (guilty feelings)
            _, p = _chi2_test(SYMPTOMS_DF, 'sfc_external_cluster', group_a, group_b, var)
        else:
            _, p = _mannwhitney_test(SYMPTOMS_DF, 'sfc_external_cluster', group_a, group_b, var)
        symptom_p_values.append(p)
        symptom_variable_names.append(f"{var} ({comparison_name})")
        symptom_test_methods.append(test_method)

_, sym_p_adj = apply_multiple_testing_correction(
    p_values=symptom_p_values,
    variable_names=symptom_variable_names,
    test_methods=symptom_test_methods,
    log_path=LOG_PATH,
)
store_pvals(sym_p_adj, symptom_variable_names, comparisons_cluster)

# SYMPTOM SEVERITY DATASET
symptom_severity_p_values = []
symptom_severity_variable_names = []
symptom_severity_test_methods = []
for var in SYMPTOM_SEVERITY_VARS:
    for comparison_name, (group_a, group_b) in comparisons_cluster.items():
        stat, p = _mannwhitney_test(SYMPTOMS_SEVERITY_DF, 'sfc_external_cluster', group_a, group_b, var)
        symptom_severity_p_values.append(p)
        symptom_severity_variable_names.append(f"{var} ({comparison_name})")
        symptom_severity_test_methods.append("Mann-Whitney U")

_, symptom_severity_p_adj = apply_multiple_testing_correction(
    p_values=symptom_severity_p_values,
    variable_names=symptom_severity_variable_names,
    test_methods=symptom_severity_test_methods,
    log_path=LOG_PATH,
)
store_pvals(symptom_severity_p_adj, symptom_severity_variable_names, comparisons_cluster)

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

# SYMPTOMS DATASET
for var in SYMPTOM_VARS: 
    current_pvals = [master_pvals.get((var, g1, g2), np.nan) for g1, g2 in plot_comparisons]

    if var == "p2030_i2":
        title_var = "guilty feelings"

        plt.figure(figsize=(8, 6))
        ax = sns.histplot(
            y="sfc_external_cluster",
            hue=var,
            hue_order=["0.0", "1.0"], 
            data=SYMPTOMS_plots_df,
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

        legend_labels = {"0.0": "No", "1.0": "Yes"}
        legend = ax.get_legend()
        handles, labels = legend.legend_handles, [text.get_text() for text in legend.get_texts()]
        new_labels = [legend_labels.get(label, label) for label in labels]
        ax.legend(handles, new_labels, title="Often Guilty Feelings", loc="upper right")

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
        if var == "p2050_i2":
            title_var = "frequency of (subclinical) depressed mood in last 2 weeks"
            title_x = "Frequency of (subclinical) depressed mood in last 2 weeks (1=not at all, 4=nearly every day)"
        elif var == "p2060_i2":
            title_var = "frequency of (subclinical) disinterest in last 2 weeks"
            title_x = "Frequency of (subclinical) disinterest in last 2 weeks (1=not at all, 4=nearly every day)"
        elif var == "p2070_i2":
            title_var = "frequency of (subclinical) tenseness in last 2 weeks"
            title_x = "Frequency of (subclinical) tenseness in last 2 weeks (1=not at all, 4=nearly every day)"
        elif var == "p2080_i2":
            title_var = "frequency of (subclinical) lethargy in last 2 weeks"
            title_x = "Frequency of (subclinical) lethargy in last 2 weeks (1=not at all, 4=nearly every day)"
            
        plt.figure(figsize=(8, 6))
        ax = pt.RainCloud(
            x="sfc_external_cluster",
            y=var,
            hue="sfc_external_cluster",
            order=group_order,
            data=SYMPTOMS_plots_df,
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

# SYMPTOM SEVERITY DATASET
for var in SYMPTOM_SEVERITY_VARS: 
    current_pvals = [master_pvals.get((var, g1, g2), np.nan) for g1, g2 in plot_comparisons]

    if var == "p4609_i2":
        title_var = "longest period of (subclinical) depression episode"
        title_x = "Longest period of (subclinical) depression episode (weeks)"
    elif var == "p4620_i2":
        title_var = "number of (subclinical) depression episodes"
        title_x = "Number of (subclinical) depression episodes"
    elif var == "p5375_i2":
        title_var = "longest period of (subclinical) disinterest episode"
        title_x = "Longest period of (subclinical) disinterest episode (weeks)"
    elif var == "p5386_i2":
        title_var = "number of (subclinical) disinterest episodes"
        title_x = "Number of (subclinical) disinterest episodes"
    plt.figure(figsize=(8, 6))
    ax = pt.RainCloud(
        x="sfc_external_cluster",
        y=var,
        hue="sfc_external_cluster",
        order=group_order,
        data=SYMPTOMS_SEVERITY_plots_df,
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
