""" Utilities for the lifestyle and environment associations with SFC-EXT derived MDD subtypes

Architecture
------------
The script is organized into functional sections:

- **Utility Functions**

- **Data Loading and Preprocessing**

- **Statistical Testing**

- **Visualization**
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
# UTILITY FUNCTIONS
# ==============================================================================
def get_stars(p):
    """Convert a p-value into significance stars."""
    if pd.isna(p): return "ns"
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"

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

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
def load_and_preprocess_cohort_data(cohort_data_path: str) -> pd.DataFrame:
    """
    Load the combined MDD and control cohort data, check for required columns,
    and preprocess the 'sfc_external_cluster' column for clarity.
    """
    cohort_df = pd.read_csv(os.path.join(cohort_data_path, "module_connectivity_features_with_covariates.csv"))
    print(f"Combined MDD and control cohort loaded with shape: {cohort_df.shape}")

    if "sfc_external_cluster" not in cohort_df.columns:
        raise ValueError("Column 'sfc_external_cluster' not found in COHORT_DF")

    # Replace cluster labels with 'Cluster 0' and 'Cluster 1' for clarity
    cohort_df["sfc_external_cluster"] = cohort_df["sfc_external_cluster"].map({
        "Control": "Control",
        "0": "Cluster 0",
        "1": "Cluster 1"
    })

    return cohort_df

def load_and_preprocess_lifestyle_environment_data(lifestyle_environment_data_path: str, cohort_df: pd.DataFrame) -> dict:
    """
    Load and preprocess lifestyle and environment datasets, ensuring they align with the cohort data.
    Returns a dictionary of processed DataFrames for each dataset.
    """
    datasets = {}
    datasets_plots = {}
    
    # Define datasets to load
    dataset_info = {
        "alcohol": ["p20117_i2", "p1558_i2"],
        "NMR_metabolomics": ["p23407_i0", "p23400_i0", "p23406_i0", "p23405_i0"],
        "physical_activity": ["p22040_i2", "p22037_i2", "p22038_i2", "p22039_i2"],
        "sleep": ["p1160_i2", "p1200_i2", "p1180_i2"],
        "social_support": ["p1031_i2", "p2110_i2"],
        "sun_exposure": ["p1050_i2", "p1060_i2"]
    }

    for dataset_name, vars in dataset_info.items():
        print(f"\nProcessing {dataset_name} dataset...")
        df = pd.read_csv(os.path.join(lifestyle_environment_data_path, f"{dataset_name}.csv"))
        print(f"{dataset_name.capitalize()} dataset loaded with shape: {df.shape}")

        # Filter to relevant variables
        df = df[["eid"] + vars]
        print(f"{dataset_name.capitalize()} dataset filtered to relevant variables with shape: {df.shape}")

        # Ensure eids are strings for merging
        df["eid"] = df["eid"].astype(str)
        cohort_df["eid"] = cohort_df["eid"].astype(str)

        # Filter to individuals present in the cohort
        df = df[df["eid"].isin(cohort_df["eid"])]
        print(f"{dataset_name.capitalize()} dataset filtered to MDD and control cohort with shape: {df.shape}")

        # Handle null values
        null_eids = df[df[vars].isnull().any(axis=1)]["eid"].tolist()
        if null_eids:
            print(f"Warning: The following eids had null values for relevant {dataset_name} variables and were dropped: {null_eids}")
            df = df.dropna(subset=vars)
            print(f"{dataset_name.capitalize()} dataset filtered to remove null values with shape: {df.shape}")

        # Handle unsuitable responses ("Prefer not to answer", "Do not know", "Less than an hour a day") based on UKB coding
        unsuitable_eids = df[df[vars].isin([-3, -1, -10]).any(axis=1)]["eid"].tolist()
        if unsuitable_eids:
            print(f"Warning: The following eids had unsuitable responses for relevant {dataset_name} variables and were dropped: {unsuitable_eids}")
            df = df[~df[vars].isin([-3, -1, -10]).any(axis=1)]
            print(f"{dataset_name.capitalize()} dataset filtered to remove unsuitable responses with shape: {df.shape}")

        # Check to which cluster the excluded individuals belonged
        excluded_eids = null_eids + unsuitable_eids
        excluded_cluster_info = cohort_df[cohort_df["eid"].isin(excluded_eids)][["eid", "sfc_external_cluster"]]
        print(f"Excluded individuals from {dataset_name} dataset due to null or unsuitable values:\n{excluded_cluster_info['sfc_external_cluster'].value_counts()}")

        # Now add the sfc_external_cluster column to the alcohol dataset for association analysis
        df = df.merge(cohort_df[["eid", "sfc_external_cluster"]], on="eid", how="left")
        print(f"{dataset_name.capitalize()} dataset after merging with SFC-EXT clusters: {df.info()}")

        # Append total depression cohort (Cluster 0 + Cluster 1 combined)
        dep_df = df[df["sfc_external_cluster"].isin(["Cluster 0", "Cluster 1"])].copy()
        dep_df["sfc_external_cluster"] = "Depression"
        df = pd.concat([df, dep_df], ignore_index=True)

        # Now ensure the sfc_external_cluster column is categorical with the correct order (correct for histplots)
        df_plots = df.copy() # We will use this copy for plotting, while keeping the original ALCOHOL_DF for analysis
        df_plots["sfc_external_cluster"] = pd.Categorical(
            df_plots["sfc_external_cluster"], categories=["Control", "Depression", "Cluster 0", "Cluster 1"], ordered=True
        )

        # Now comes the dataset-specific processing, like reversing order of variables, creating new variables, etc.
        if dataset_name == "alcohol":
            # Set the correct data type for the variable p20117_i2 (alcohol drinker status) to categorical, as it is a categorical variable
            df["p20117_i2"] = df["p20117_i2"].astype("str")

            # Reverse the order of the alcohol intake frequency variable (p1558_i2) so that higher values indicate more frequent drinking, for easier interpretation in plots
            df["p1558_i2"] = df["p1558_i2"].map({
                1: 6,  # "Daily or almost daily" -> 6
                2: 5,  # "Three or four times a week" -> 5
                3: 4,  # "Once or twice a week" -> 4
                4: 3,  # "One to three times a month" -> 3
                5: 2,  # "Special occasions only" -> 2
                6: 1,  # "Never" -> 1
            })
        
        elif dataset_name == "NMR_metabolomics":
            # Now create the insulin resistance proxy as the ratio of triglycerides to HDL cholesterol (as done in Oliveri et al., 2024)
            df["insulin_resistance_proxy"] = df["p23407_i0"] / df["p23406_i0"]
            print(f"Insulin resistance proxy variable description: {df['insulin_resistance_proxy'].describe()}")

        elif dataset_name == "sleep":
            # Set the correct data type for the variable p1200 (sleeplessness) to categorical, as it is a categorical variable
            df["p1200_i2"] = df["p1200_i2"].astype("str")

        elif dataset_name == "social_support":
            # Reverse the order of the social support variable (p1031_i2) so that higher values indicate more frequent social support, for easier interpretation in plots
            df["p1031_i2"] = df["p1031_i2"].map({
                1: 7,  # "Almost daily" -> 7
                2: 6,  # "Two to four times a week" -> 6
                3: 5,  # "About once a week" -> 5
                4: 4,  # "About once a month" -> 4
                5: 3,  # "Once every few months" -> 3
                6: 2,  # "Never or almost never" -> 2
                7: 1,  # "No friends/family outside household" -> 1
            })

        # Store the processed dataset in the dictionary
        datasets[dataset_name] = df

        # Store the plotting dataset in the dictionary
        datasets_plots[dataset_name] = df_plots

    return datasets, datasets_plots

# ==============================================================================
# STATISTICAL TESTING 
# ==============================================================================
def print_percentiles(df, variables, dataset_name):
    print(f"\nPercentiles for {dataset_name} dataset:")
    for var in variables:
        if var in df.columns:
            print(f"\nVariable: {var}")
            percentiles = df.groupby("sfc_external_cluster")[var].quantile([0.25, 0.5, 0.75]).unstack()
            percentiles.columns = ['25th Percentile', '50th Percentile (Median)', '75th Percentile']
            print(percentiles)
        else:
            print(f"Variable {var} not found in {dataset_name} dataset.")

def print_categorical_percentages(df, categorical_vars, dataset_name):
    print(f"\nCategorical variable percentages for {dataset_name} dataset:")
    for var in categorical_vars:
        if var in df.columns:
            print(f"\nVariable: {var}")
            percentages = df.groupby("sfc_external_cluster")[var].value_counts(normalize=True).unstack() * 100
            print(percentages)
        else:
            print(f"Variable {var} not found in {dataset_name} dataset.")

def print_percentiles_and_categorical_percentages(datasets):
    """
    Print percentiles for continuous variables and categorical percentages for each dataset variable.
    """
    for dataset_name, df in datasets.items():
        if dataset_name == "alcohol":
            continuous_vars = [var for var in df.columns if var not in ["eid", "sfc_external_cluster", "p20117_i2"]]
            categorical_vars = [var for var in df.columns if var == "p20117_i2"]
        elif dataset_name == "NMR_metabolomics":
            continuous_vars = [var for var in df.columns if var not in ["eid", "sfc_external_cluster",]]
            categorical_vars = []
        elif dataset_name == "physical_activity":
            continuous_vars = [var for var in df.columns if var not in ["eid", "sfc_external_cluster",]]
            categorical_vars = []
        elif dataset_name == "sleep":
            continuous_vars = [var for var in df.columns if var not in ["eid", "sfc_external_cluster", "p1200_i2"]]
            categorical_vars = [var for var in df.columns if var == "p1200_i2"]
        elif dataset_name == "social_support":
            continuous_vars = [var for var in df.columns if var not in ["eid", "sfc_external_cluster",]]
            categorical_vars = []
        elif dataset_name == "sun_exposure":
            continuous_vars = [var for var in df.columns if var not in ["eid", "sfc_external_cluster",]]
            categorical_vars = []
    print_percentiles(df, continuous_vars, dataset_name)
    print_categorical_percentages(df, categorical_vars, dataset_name)

def _chi2_test(df, x_col, group_a, group_b, value_col):
    """
    Perform a Chi-square test of independence between two categorical variables for two specified groups.
    """
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
    """
    Perform a Mann-Whitney U test between two groups for a continuous variable.
    """
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

def perform_statistical_tests_and_corrections(datasets, log_path):
    """
    For each dataset, perform statistical tests between the control group and SFC-EXT derived clusters,
    apply multiple testing correction, and store the corrected p-values for plotting.
    """
    # Master dictionary to hold all corrected p-values for plotting
    master_pvals = {}

    comparisons_cluster = {
        'depression_vs_control': ('Depression', 'Control'),
        'cluster0_vs_control': ('Cluster 0', 'Control'),
        'cluster1_vs_control': ('Cluster 1', 'Control'),
        'cluster0_vs_cluster1': ('Cluster 0', 'Cluster 1'),
    }

    for dataset_name, df in datasets.items():
        print(f"\nPerforming statistical tests for {dataset_name} dataset...")
        if dataset_name == "alcohol":
            vars_to_test = [var for var in df.columns if var not in ["eid", "sfc_external_cluster"]]  # All variables in the alcohol dataset
        elif dataset_name == "NMR_metabolomics":
            vars_to_test = [var for var in df.columns if var not in ["eid", "sfc_external_cluster"]]
        elif dataset_name == "physical_activity":
            vars_to_test = [var for var in df.columns if var not in ["eid", "sfc_external_cluster"]]
        elif dataset_name == "sleep":
            vars_to_test = [var for var in df.columns if var not in ["eid", "sfc_external_cluster"]]
        elif dataset_name == "social_support":
            vars_to_test = [var for var in df.columns if var not in ["eid", "sfc_external_cluster"]]
        elif dataset_name == "sun_exposure":
            vars_to_test = [var for var in df.columns if var not in ["eid", "sfc_external_cluster"]]
        else:
            print(f"Unknown dataset: {dataset_name}. Skipping.")
            continue

        p_values = []
        variable_names = []
        test_methods = []

        for var in vars_to_test:
            if var not in df.columns:
                print(f"Variable {var} not found in {dataset_name} dataset. Skipping.")
                continue

            if var in ["p20117_i2", "p1200_i2"]:  # Categorical variables
                test_method = "Chi-square"
                test_func = _chi2_test
            else:  # Continuous variables
                test_method = "Mann-Whitney U"
                test_func = _mannwhitney_test

            for comparison_name, (group_a, group_b) in comparisons_cluster.items():
                stat, p = test_func(df, 'sfc_external_cluster', group_a, group_b, var)
                p_values.append(p)
                variable_names.append(f"{var} ({comparison_name})")
                test_methods.append(test_method)

        _, adj_pvals = apply_multiple_testing_correction(
            p_values=p_values,
            variable_names=variable_names,
            test_methods=test_methods,
            log_path=log_path,
        )

        for name, p in zip(variable_names, adj_pvals):
            var = name.split(" (")[0]
            comp_name = name.split(" (")[1].replace(")", "")
            ga, gb = comparisons_cluster[comp_name]
            # Store in both directions for easy lookup later
            master_pvals[(var, ga, gb)] = p
            master_pvals[(var, gb, ga)] = p
    
    return master_pvals

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def get_variable_title_and_xlabel(var):
    """
    Returns a human-readable title and x-axis label for a given variable.
    """
    if var == "p20117_i2":
        return "alcohol drinker status", "Alcohol drinker status"
    elif var == "p1558_i2":
        return "alcohol intake frequency", "Alcohol intake frequency (1=never, 6=daily or almost daily)"
    elif var == "insulin_resistance_proxy":
        return "insulin resistance marker", "Insulin resistance marker (TG:HDL-C)"
    elif var == "p23400_i0":
        return "total cholesterol", "Total cholesterol (mmol/L)"
    elif var == "p23407_i0":
        return "total triglycerides", "Total triglycerides (mmol/L)"
    elif var == "p23406_i0":
        return "HDL cholesterol", "HDL cholesterol (mmol/L)"
    elif var == "p23405_i0":
        return "LDL cholesterol", "LDL cholesterol (mmol/L)"
    elif var == "p22040_i2":
        return "total physical activity", "Total physical activity (MET-minutes/week for all activities)"
    elif var == "p22037_i2":
        return "walking physical activity", "Walking physical activity (MET-minutes/week)"
    elif var == "p22038_i2":
        return "moderate physical activity", "Moderate physical activity (MET-minutes/week)"
    elif var == "p22039_i2":
        return "vigorous physical activity", "Vigorous physical activity (MET-minutes/week)"
    elif var == "p1160_i2":
        return "sleep duration", "Sleep duration (hours/day)"
    elif var == "p1200_i2":
        return "sleeplessness", "Sleeplessness/Insomnia"
    elif var == "p1180_i2":
        return "chronotype", "Chronotype (1=morning, 4=evening)"
    elif var == "p1031_i2":
        return "frequency of social visits", "Frequency of friend/family visits (1=no friends/family outside household, 7=almost daily)"
    elif var == "p2110_i2":
        return "ability to confide in someone close", "Ability to confide in someone close (0=never or almost never, 5=almost daily)"
    elif var == "p1050_i2":
        return "time spent outdoors in summer", "Time spent outdoors in summer (hours/day)"
    elif var == "p1060_i2":
        return "time spent outdoors in winter", "Time spent outdoors in winter (hours/day)"

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

def plot_lifestyle_environment_distributions(datasets_plots, master_pvals, plots_dir):
    """
    Generate and save distribution plots for each lifestyle and environment variable,
    with significance brackets indicating statistical differences between groups.
    """
    # Ensure the plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Control and SFC-EXT cluster palette colors
    group_colors = {
        "Control": "#2ca02c",
        "Depression": "#6a3d9a",
        "Cluster 0": "#0026ff", 
        "Cluster 1": "#fd7600", 
    }
    group_order = ["Control", "Depression", "Cluster 0", "Cluster 1"]
    palette_list = [group_colors[g] for g in group_order]

    # The list formatting for plotting the brackets
    plot_comparisons = [('Control', 'Depression'), ('Control', 'Cluster 0'), ('Control', 'Cluster 1'), ('Cluster 0', 'Cluster 1')]

    for dataset_name, df in datasets_plots.items():
        for var in df.columns:
            if var in ["eid", "sfc_external_cluster"]:
                continue  # Skip non-variable columns

            current_pvals = [master_pvals.get((var, g1, g2), np.nan) for g1, g2 in plot_comparisons]

            if var in ["p20117_i2", "p1200_i2"]:  # Categorical variables
                title_var, title_x = get_variable_title_and_xlabel(var)
                plt.figure(figsize=(8, 6))
                ax = sns.histplot(
                    y="sfc_external_cluster",
                    hue=var,
                    hue_order=sorted(df[var].dropna().unique()), 
                    data=df,
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

                if var == "p20117_i2":
                    legend_labels = {"0.0": "Never", "1.0": "Previous", "2.0": "Current"}
                elif var == "p1200_i2":
                    legend_labels = {"1.0": "Never/rarely", "2.0": "Sometimes", "3.0": "Usually"}

                legend = ax.get_legend()
                handles, labels = legend.legend_handles, [text.get_text() for text in legend.get_texts()]
                new_labels = [legend_labels.get(label, label) for label in labels]
                ax.legend(handles, new_labels, title=title_x, loc="upper right")

                plt.draw() 
                y_labels = [t.get_text() for t in ax.get_yticklabels()]
                y_coords = {lbl: i for i, lbl in enumerate(y_labels)}
                
                draw_right_brackets(ax, y_coords, plot_comparisons, current_pvals)

                plt.title(f"Distribution of {title_var}")
                plt.xlabel("Percentage")
                plt.ylabel("Group")
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{title_var}.svg"), dpi=300)
            
            else:  # Continuous variables
                title_var, title_x = get_variable_title_and_xlabel(var)
                plt.figure(figsize=(8, 6))
                ax = pt.RainCloud(
                    x="sfc_external_cluster",
                    y=var,
                    hue="sfc_external_cluster",
                    order=group_order,
                    data=df,
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
                plt.savefig(os.path.join(plots_dir, f"{title_var}.svg"), dpi=300)

