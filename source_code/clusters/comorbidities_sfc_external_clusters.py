""" Determining comorbidities (and their nature) for SFC external clusters
"""
import os
import pandas as pd
from collections import Counter
from typing import Optional
import re
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
import logging

# ==============================================================================
# Configuration
# ==============================================================================
GENERAL_DATA_PATH = ".../data/UKB"
SFC_EXT_CLUSTERS_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "cohorts", "module_connectivity_features_with_covariates.csv")
COMORBIDITIES_DATA_PATH = os.path.join(GENERAL_DATA_PATH, "cohorts", "depression_cohort_F32.csv")
PLOTS_DIR = ".../reports/plots/comorbidities_sfc_external_clusters"

# Set up logging to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(".../comorbidity_sfc_external_clusters.log"),
    ]
)

# ==============================================================================
# Load datasets
# ==============================================================================
# Combined MDD and control cohort, with SFC-EXT derived subtypes
SFC_EXT_DF = pd.read_csv(SFC_EXT_CLUSTERS_DATA_PATH)
print(f"Combined MDD and control cohort loaded with shape: {SFC_EXT_DF.shape}")
# Check if subtype column exists
if "sfc_external_cluster" not in SFC_EXT_DF.columns:
    raise ValueError("Column 'sfc_external_cluster' not found in SFC_EXT_DF")
# Replace cluster labels with 'Cluster 0' and 'Cluster 1' for clarity
SFC_EXT_DF["sfc_external_cluster"] = SFC_EXT_DF["sfc_external_cluster"].map({"Control": "Control", "0": "Cluster 0", "1": "Cluster 1"})

# Comorbidity data for the MDD cohort 
COMORBIDITIES_DF = pd.read_csv(COMORBIDITIES_DATA_PATH)
print(f"Comorbidity data loaded with shape: {COMORBIDITIES_DF.shape}")

# ==============================================================================
# Extract relevant columns from each dataset and make new dataset
# ==============================================================================
# Extract only the necessary columns from the comorbidities dataset
comorbidities_subset = COMORBIDITIES_DF[['eid', 'codes']].copy()
# Extract only the necessary columns from the sfc_external_clusters dataset and ensure no 'Control' entries are included
sfc_clusters_subset = SFC_EXT_DF[['eid', 'sfc_external_cluster']].copy()
sfc_clusters_subset = sfc_clusters_subset[sfc_clusters_subset['sfc_external_cluster'] != 'Control']
# Ensure that 'eid' is of the same type in both datasets for merging
sfc_clusters_subset['eid'] = sfc_clusters_subset['eid'].astype(str)
comorbidities_subset['eid'] = comorbidities_subset['eid'].astype(str)
# Merge the two datasets on 'eid' to create a combined dataset
combined_df = pd.merge(sfc_clusters_subset, comorbidities_subset, on='eid', how='left')
print(f"Combined dataset created with shape: {combined_df.shape}")
print(f"Columns in combined dataset: {combined_df.columns.tolist()}")
print(f"Sample of combined dataset:\n{combined_df.head()}")

# ==============================================================================
# Some helper functions for processing codes and determining stats and plotting
# ==============================================================================
def split_codestring(codestring: str) -> List[str]:
    """
    Split a codes string into individual codes.

    Notes
    -----
    The current implementation splits only on the vertical bar character
    (``'|'``). This matches the original legacy behavior encountered in
    UKB exports where multiple codes for an individual are joined by ``|``.
    Callers with data that uses commas/semicolons/whitespace as delimiters
    should pre-process their strings or update this helper accordingly.

    Parameters
    ----------
    codestring : str
        String containing one or more codes separated by the vertical bar
        character (e.g. ``'F32|F33|G40'``).

    Returns
    -------
    List[str]
        List of individual non-empty codes (strings), or an empty list for
        blank input.

    Examples
    --------
    >>> split_codestring('F32|F33|G40')
    ['F32', 'F33', 'G40']
    """
    codes = re.split(r'[|]+', codestring.strip())
    codes = [code for code in codes if code]
    return codes

def count_codes_in_cohort(
    cohort_df: pd.DataFrame,
    codes_column_cohort: str = 'codes',
    coding_filepath: str = '.../data/UKB/coding19.tsv',
    output_path: str = None
) -> pd.DataFrame:
    """
    Count occurrences of each ICD-10 code in the specified codes column of a cohort DataFrame.
    
    Parameters
    ----------
    cohort_df : pd.DataFrame
        Cohort DataFrame containing a column with ICD-10 codes.
    codes_column_cohort : str, optional
        Name of the column containing ICD-10 codes (default: 'codes').
    coding_filepath : str, optional
        Filepath to the coding file mapping ICD-10 codes to titles (default: '.../data/UKB/coding19.tsv', from UKB showcase).
    output_path : str, optional
        If provided, save the resulting code distribution csv file to this directory (default: None)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with ICD-10 codes and their counts, proportions, and human-readable meanings, sorted by count in descending order.
    
    Notes
    -----
    - This function relies on ``split_codestring`` to parse the individual
        code tokens from the cohort's code strings; ensure that the codes are
        formatted consistently for accurate counting.
    - The optional ``coding_filepath`` is used to map ICD-10 codes to human
        readable meanings; if the file is missing or malformed an exception may
        be raised by ``pandas.read_csv``.
    """
    
    # Get sample size of cohort
    n_subjects = len(cohort_df)

    # Robustly load ICD-10 code titles (TSV with code\tmeaning)
    icd_texts_df = pd.read_csv(
        coding_filepath,
        sep='\t',
        header=None,
        usecols=[0, 1],
        names=['coding', 'meaning'],
        dtype=str,
        comment='#',
        engine='python'
    )
    icd_texts_df['coding'] = icd_texts_df['coding'].str.strip()
    icd_texts_df['meaning'] = icd_texts_df['meaning'].fillna('').astype(str).str.strip()

    # mapping for fast lookup
    icd_to_title = dict(zip(icd_texts_df['coding'], icd_texts_df['meaning']))

    all_codes = []
    code_distribution = {"code": [], "count": [], "proportion": [], "meaning": []}

    for codes in cohort_df[codes_column_cohort].fillna(''):
        codes_list = split_codestring(codes)
        all_codes.extend(codes_list)

    code_counter = Counter(all_codes)
    for code, count in code_counter.items():
        code_distribution["code"].append(code)
        code_distribution["count"].append(count)
        code_distribution["proportion"].append(count / max(1, n_subjects))
        # exact match
        title = icd_to_title.get(code)
        # fallback to category (e.g., F32.1 -> F32)
        if title is None and '.' in code:
            title = icd_to_title.get(code.split('.', 1)[0])
        code_distribution["meaning"].append(title if title is not None and title != '' else "N/A")

    code_distribution = pd.DataFrame(code_distribution).sort_values(by='count', ascending=False).reset_index(drop=True)
    
    if output_path:
        code_distribution.to_csv(output_path, index=False)
        print(f"ICD-10 code distribution saved to {output_path}")

    return code_distribution

def build_comorbidity_indicator_matrix(
    cohort_df: pd.DataFrame,
    *,
    eid_column: str = 'eid',
    codes_column_cohort: str = 'codes',
    proportion_threshold: float = 0.10,
    coding_filepath: str = '.../data/UKB/coding19.tsv',
    max_comorbidities: Optional[int] = None,
    include_codes: Optional[List[str]] = None,
    exclude_codes: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Add comorbidity indicator columns to a cohort DataFrame.

    This function is designed to work together with `count_codes_in_cohort`.

     It:
    1) Preserves all rows/eIDs in `cohort_df`.
    2) Selects which comorbidities (ICD-10 codes) to include based on a
       prevalence threshold (proportion of subjects with the code) and/or an
       explicit include list.
     3) Adds one column per selected code, with 1 if the eID has that code (or
         0 otherwise).

    Parameters
    ----------
    cohort_df : pd.DataFrame
        Cohort DataFrame containing at least `eid_column` and `codes_column_cohort`.
    eid_column : str
        Column name containing subject IDs (default: 'eid').
    codes_column_cohort : str
        Column name containing ICD-10 codes strings (default: 'codes').
    proportion_threshold : float
        Minimum prevalence (count / n_subjects, as in `count_codes_in_cohort`) for a code
        to be included when `include_codes` is not provided (default: 0.10).
    coding_filepath : str
        Passed through to `count_codes_in_cohort` (default points to UKB coding file).
    max_comorbidities : int | None
        If provided, only keep the top-N most frequent codes after thresholding.
    include_codes : list[str] | None
        If provided, *only* these codes are used (threshold is ignored).
    exclude_codes : list[str] | None
        Code prefixes to always drop from the selected set (e.g., excluding 'F32'
        will also exclude 'F32.0', 'F32.1', etc.).
    output_path : str | None
        If provided, save the augmented cohort to this CSV path. If a directory
        is provided, a default filename is used.

    Returns
    -------
    pd.DataFrame
        A copy of `cohort_df` with added 0/1 comorbidity columns.

    Notes
    -----
    - If no comorbidity codes meet the selection criteria the function
        returns a copy of ``cohort_df`` unchanged (but may write the copy to
        ``output_path`` if provided).
    - When ``output_path`` is provided and is a directory, a default filename
        ``'cohort_with_comorbidity_indicators.csv'`` is used; otherwise the
        provided path is treated as a file path and will be overwritten.
    """
    if pd.Index(cohort_df.columns).duplicated().any():
        dupes = pd.Index(cohort_df.columns)[pd.Index(cohort_df.columns).duplicated()].unique().tolist()
        raise ValueError(
            "Input cohort_df contains duplicate column names, which is not supported. "
            f"Duplicate columns: {dupes}"
        )

    if eid_column not in cohort_df.columns:
        raise ValueError(f"Column '{eid_column}' not found in cohort_df")
    if codes_column_cohort not in cohort_df.columns:
        raise ValueError(f"Column '{codes_column_cohort}' not found in cohort_df")
    if proportion_threshold < 0 or proportion_threshold > 1:
        raise ValueError("proportion_threshold must be between 0 and 1")

    # Work on a copy so callers can decide whether to reassign.
    cohort_base = cohort_df.copy()
    cohort_base[eid_column] = cohort_base[eid_column].astype(str)
    cohort_base[codes_column_cohort] = cohort_base[codes_column_cohort].fillna('').astype(str)

    # Build a per-eID codes table. Ensure each subject contributes at most one
    # count per code for prevalence calculation.
    def _uniq_codestring(s: str) -> str:
        """Return a sorted, pipe-delimited unique set of codes for a single subject.

        Used to normalise multiple rows per subject into a single canonical
        codes string when computing prevalences. This helper preserves the
        legacy ``'|'`` delimiter used elsewhere in the codebase.
        """
        codes = set(split_codestring(s))
        return '|'.join(sorted(codes))

    codes_by_eid = (
        cohort_base[[eid_column, codes_column_cohort]]
        .groupby(eid_column, as_index=False)[codes_column_cohort]
        .agg(lambda vals: '|'.join([v for v in vals if isinstance(v, str) and v != '']))
    )
    codes_by_eid[codes_column_cohort] = codes_by_eid[codes_column_cohort].fillna('').astype(str).map(_uniq_codestring)

    

    code_distribution = count_codes_in_cohort(
        codes_by_eid,
        codes_column_cohort=codes_column_cohort,
        coding_filepath=coding_filepath,
        output_path=None,
    )

    selected = code_distribution.copy()
    if include_codes is not None:
        include_set = set(map(str, include_codes))
        selected = selected[selected['code'].astype(str).isin(include_set)]
    else:
        selected = selected[selected['proportion'] >= proportion_threshold]

    if exclude_codes is not None:
        exclude_prefixes = [str(x) for x in exclude_codes]

        def _excluded(code: str) -> bool:
            """Return True if `code` matches or starts with any excluded prefix.

            This helper considers exact matches and prefix matches (including
            dotted extensions, e.g., ``F32`` matches ``F32.1``).
            """
            for pref in exclude_prefixes:
                if code == pref or code.startswith(pref) or code.startswith(f"{pref}."):
                    return True
            return False

        selected_codes = selected['code'].astype(str)
        selected = selected[~selected_codes.map(_excluded)]

    if max_comorbidities is not None:
        if max_comorbidities <= 0:
            raise ValueError("max_comorbidities must be a positive integer")
        selected = selected.sort_values('count', ascending=False).head(int(max_comorbidities))

    comorbidity_codes: List[str] = selected['code'].astype(str).tolist()
    code_set = set(comorbidity_codes)

    eids = codes_by_eid[eid_column].astype(str).tolist()
    codes_series = codes_by_eid[codes_column_cohort]

    # If nothing selected, just return cohort with no added columns.
    if not comorbidity_codes:
        if output_path:
            out_path = os.fspath(output_path)
            if os.path.isdir(out_path):
                out_path = os.path.join(out_path, 'cohort_with_comorbidity_indicators.csv')
            cohort_base.to_csv(out_path, index=False)
            print(f"Cohort with comorbidity indicators saved to {out_path}")
        return cohort_base

    # Build a long table of (eid, code) for selected codes, then pivot.
    long_eids: List[str] = []
    long_codes: List[str] = []
    for eid, codes_str in zip(eids, codes_series.tolist()):
        # Use a set to ensure each (eid, code) appears at most once.
        for code in set(split_codestring(codes_str)):
            if code in code_set:
                long_eids.append(eid)
                long_codes.append(code)

    if long_eids:
        long_df = pd.DataFrame({'eid': long_eids, 'code': long_codes, 'value': 1})
        mat = long_df.drop_duplicates(subset=['eid', 'code']).pivot_table(
            index='eid',
            columns='code',
            values='value',
            aggfunc='max',
            fill_value=0,
        )
    else:
        mat = pd.DataFrame(index=pd.Index([], name='eid'))

    # Ensure all eIDs exist and all selected code columns exist (fill missing with 0)
    mat = mat.reindex(index=eids, fill_value=0)
    for code in comorbidity_codes:
        if code not in mat.columns:
            mat[code] = 0

    # Re-order columns and cast to int 0/1
    mat = mat[comorbidity_codes].astype(int)
    indicators = mat.reset_index().rename(columns={'eid': eid_column})
    indicators = indicators[[eid_column] + comorbidity_codes]

    # Merge back onto the cohort (broadcasts to all rows with the same eID).
    # If code columns already exist, overwrite them with the recomputed values.
    overlap_cols = [c for c in comorbidity_codes if c in cohort_base.columns]
    if overlap_cols:
        cohort_base = cohort_base.drop(columns=overlap_cols)

    cohort_out = cohort_base.merge(indicators, on=eid_column, how='left', validate='m:1')
    cohort_out[comorbidity_codes] = cohort_out[comorbidity_codes].fillna(0).astype(int)

    # Safety check: ensure we didn't drop any original columns.
    missing_cols = [c for c in cohort_df.columns if c not in cohort_out.columns]
    if missing_cols:
        # Reattach any missing original columns by index alignment.
        for c in missing_cols:
            cohort_out[c] = cohort_df[c].values
        # Keep original column order first, then new comorbidity columns.
        ordered = list(cohort_df.columns) + [c for c in comorbidity_codes if c not in cohort_df.columns]
        cohort_out = cohort_out[ordered]

    # Enforce unique column names in the final output.
    if pd.Index(cohort_out.columns).duplicated().any():
        dupes = pd.Index(cohort_out.columns)[pd.Index(cohort_out.columns).duplicated()].unique().tolist()
        raise RuntimeError(
            "build_comorbidity_indicator_matrix produced duplicate column names, "
            f"which is not allowed. Duplicate columns: {dupes}"
        )

    if output_path:
        out_path = os.fspath(output_path)
        if os.path.isdir(out_path):
            out_path = os.path.join(out_path, 'cohort_with_comorbidity_indicators.csv')
        cohort_out.to_csv(out_path, index=False)
        print(f"Cohort with comorbidity indicators saved to {out_path}")

    return cohort_out

def compare_comorbidities(
    df,
    group_col,
    comorbidity_cols,
    top_n=None,
    correction_method="fdr_bh",
):
    """
    Parameters
    ----------
    df : pd.DataFrame
        One row per patient. Comorbidity columns must be binary (0/1).
    group_col : str
        Column name identifying the two groups (must have exactly 2 unique values).
    comorbidity_cols : list of str
        Column names of the comorbidities to test.
    top_n : int, optional
        If given, only the n most prevalent comorbidities (overall, across
        both groups) are tested. If None, all comorbidity_cols are tested.
    correction_method : str
        Passed to statsmodels multipletests. Common choices:
        'fdr_bh' (Benjamini-Hochberg, recommended default),
        'bonferroni', 'holm'.
 
    Returns
    -------
    pd.DataFrame sorted by raw p-value, with columns:
        comorbidity, n/prop per group, test used, raw p-value,
        adjusted p-value, significant (bool), smd
    """
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"group_col must have exactly 2 groups, found {len(groups)}")
    g1, g2 = groups[0], groups[1]
 
    if top_n is not None:
        prevalence = df[comorbidity_cols].mean().sort_values(ascending=False)
        comorbidity_cols = prevalence.head(top_n).index.tolist()
 
    rows = []
    for col in comorbidity_cols:
        tab = pd.crosstab(df[group_col], df[col])
        tab = tab.reindex(index=[g1, g2], columns=[0, 1], fill_value=0)
 
        n1, n2 = tab.loc[g1].sum(), tab.loc[g2].sum()
        a, b = tab.loc[g1, 1], tab.loc[g2, 1]  # has comorbidity, per group
        p1, p2 = a / n1, b / n2
 
        _, p_chi2, _, expected = chi2_contingency(tab, correction=False)
 
        if (expected < 5).any():
            _, p_value = fisher_exact(tab)
            test_used = "Fisher exact"
        else:
            p_value = p_chi2
            test_used = "Chi-square"
 
        pooled_sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        smd = (p1 - p2) / pooled_sd if pooled_sd > 0 else np.nan
 
        rows.append(
            {
                "comorbidity": col,
                f"n_{g1}": n1,
                f"prop_{g1}": round(p1, 4),
                f"n_{g2}": n2,
                f"prop_{g2}": round(p2, 4),
                "test": test_used,
                "p_value": p_value,
                "smd": round(smd, 4),
            }
        )
 
    results = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(results["p_value"], method=correction_method)
    results["p_adj"] = p_adj
    results["significant"] = reject
 
    return results.sort_values("p_value").reset_index(drop=True)

def compare_category_proportions(
    counts_df,
    category_col,
    count_col_g1,
    count_col_g2,
    n_g1,
    n_g2,
    group_labels=("Group 1", "Group 2"),
    top_n=None,
    correction_method="fdr_bh",
):
    """
    Parameters
    ----------
    counts_df : pd.DataFrame
        One row per ICD-10 category, with a count column for each group.
    category_col : str
        Column with the category/chapter name.
    count_col_g1, count_col_g2 : str
        Columns with the counts of each category in each group.
    n_g1, n_g2 : int
        Total number of letters (ICD-10 categories) in each group (the denominator for proportions).
    group_labels : tuple of str
        Display names for the two groups.
    top_n : int, optional
        If given, keep only the n categories with the highest combined count
        (across both groups) before testing.
    correction_method : str
        Passed to statsmodels multipletests: 'fdr_bh' (default), 'bonferroni', 'holm', etc.
 
    Returns
    -------
    pd.DataFrame sorted by raw p-value, with counts, proportions, test used,
    raw/adjusted p-values, significance flag, and SMD per category.
    """
    df = counts_df.copy()
    missing = [c for c in (category_col, count_col_g1, count_col_g2) if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in counts_df: {missing}")
 
    if top_n is not None:
        df = df.assign(_combined=df[count_col_g1] + df[count_col_g2])
        df = df.sort_values("_combined", ascending=False).head(top_n).drop(columns="_combined")
 
    if len(df) == 0:
        raise ValueError("No categories left to test — check top_n and input data.")
 
    g1_label, g2_label = group_labels
    rows = []
    for _, r in df.iterrows():
        a, b = int(r[count_col_g1]), int(r[count_col_g2])
        if a > n_g1 or b > n_g2:
            raise ValueError(
                f"Category '{r[category_col]}': count exceeds group total "
                f"({a}/{n_g1}, {b}/{n_g2}) — check n_g1/n_g2."
            )
 
        table = [[a, n_g1 - a], [b, n_g2 - b]]
        p1, p2 = a / n_g1, b / n_g2
 
        _, p_chi2, _, expected = chi2_contingency(table, correction=False)
        if (expected < 5).any():
            _, p_value = fisher_exact(table)
            test_used = "Fisher exact"
        else:
            p_value = p_chi2
            test_used = "Chi-square"
 
        pooled_sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
        smd = (p1 - p2) / pooled_sd if pooled_sd > 0 else np.nan
 
        rows.append(
            {
                "category": r[category_col],
                f"count_{g1_label}": a,
                f"prop_{g1_label}": round(p1, 4),
                f"count_{g2_label}": b,
                f"prop_{g2_label}": round(p2, 4),
                "test": test_used,
                "p_value": p_value,
                "smd": round(smd, 4),
            }
        )
 
    results = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(results["p_value"], method=correction_method)
    results["p_adj"] = p_adj
    results["significant"] = reject
 
    return results.sort_values("p_value").reset_index(drop=True)

def _extract_letters(data: str, exclude_prefixes: tuple = ('F32',)):
    """
    Extract the leading letter of each '|'-separated token in a string,
    skipping any token that starts with one of exclude_prefixes.
    """
    if pd.isna(data):
        return []
    tokens = data.split('|')
    letters = []
    for t in tokens:
        if any(t.upper().startswith(p.upper()) for p in exclude_prefixes):
            continue  # skip excluded codes entirely
        m = re.match(r'[A-Za-z]+', t)
        if m:
            letters.append(m.group().upper())
    return letters

def letter_stats_overall(
    df: pd.DataFrame, 
    codes_column_cohort: str
) -> dict:
    """
    Compute aggregate letter stats across ALL rows combined (excluding primary diagnosis codes like F32).
    """
    all_letters = df[codes_column_cohort].apply(_extract_letters).sum()  # flattens lists
    total = len(all_letters)
    counts = Counter(all_letters)
    e_count = counts.get('E', 0)
    f_count = counts.get('F', 0)
    g_count = counts.get('G', 0)
    i_count = counts.get('I', 0)
    k_count = counts.get('K', 0)
    r_count = counts.get('R', 0)
    z_count = counts.get('Z', 0)

    return {
        'total_letters': total,
        'e_count': e_count,
        'f_count': f_count,
        'g_count': g_count,
        'i_count': i_count,
        'k_count': k_count,
        'r_count': r_count,
        'z_count': z_count,
        'e_proportion': e_count / total if total else 0.0,
        'f_proportion': f_count / total if total else 0.0,
        'g_proportion': g_count / total if total else 0.0,
        'i_proportion': i_count / total if total else 0.0,
        'k_proportion': k_count / total if total else 0.0,
        'r_proportion': r_count / total if total else 0.0,
        'z_proportion': z_count / total if total else 0.0,
        'letter_counts': dict(counts.most_common()),  # sorted most → least),
    }

def _extract_codes(data: str, exclude_prefixes: tuple = ('F32',)):
    """
    Extract the letter + 2-digit category for each '|'-separated token. Excluding any token that starts with one of exclude_prefixes.
    e.g. 'R401' -> 'R40', 'F171' -> 'F17', 'I48' -> 'I48'
    """
    if pd.isna(data):
        return []
    tokens = data.split('|')
    codes = []
    for t in tokens:
        if any(t.upper().startswith(p.upper()) for p in exclude_prefixes):
            continue  # skip excluded codes entirely
        m = re.match(r'([A-Za-z]+)(\d+)', t)
        if not m:
            continue
        letter, digits = m.group(1).upper(), m.group(2)
        if len(digits) < 2:
            continue  # not enough digits to determine a category
        codes.append(f"{letter}{digits[:2]}")
    return codes


def range_stats_overall(
    df: pd.DataFrame, 
    codes_column_cohort: str,
    letter: str = 'R', 
    low: int = 40, 
    high: int = 46
) -> dict:
    """
    Compute aggregate stats for codes in [letter+low, letter+high]
    (e.g. R40-R46) vs all other codes, across ALL rows combined. Excludes primary diagnosis codes like F32.
    """
    all_codes = df[codes_column_cohort].apply(_extract_codes).sum()  # flatten lists
    total = len(all_codes)

    target_categories = {f"{letter}{n:02d}" for n in range(low, high + 1)}
    counts = Counter(all_codes)
    target_count = sum(v for k, v in counts.items() if k in target_categories)

    # Breakdown of just the target range, sorted most -> least,
    # including categories with zero occurrences.
    range_breakdown = dict(
        sorted(
            ((cat, counts.get(cat, 0)) for cat in target_categories),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return {
        'total_letters': total,
        'range_count': target_count,
        'other_count': total - target_count,
        'range_proportion': target_count / total if total else 0.0,
        'range_breakdown': range_breakdown,
    }

def plot_comorbidity_distribution(
    code_distribution: pd.DataFrame,
    proportion_threshold: float = 0.10,
    title: str = "ICD-10 Code Distribution",
    figsize: tuple = (10, 10),
    output_path: str = None,
    wrap_width: int = 40,       # max chars per line for y-labels
    y_label_rotation: float = 15.0,  # degrees to tilt y-tick labels
    y_spacing: float = 3.0      # multiplier to increase vertical spacing between y-tick labels
) -> None:
    """
    Plot a horizontal bar chart of ICD-10 code distribution from a DataFrame.

    Parameters
    ----------
    code_distribution : pd.DataFrame
        DataFrame returned by `count_codes_in_cohort`, containing 'code', 'count', 'proportion', and 'meaning'.
    proportion_threshold : float, optional
        Minimum proportion threshold to include a code in the plot (default: 0.10).
    title : str, optional
        Title of the plot (default: "ICD-10 Code Distribution").
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6)).
    output_path : str, optional
        If provided, save the plot to this file path (default: None).
    wrap_width : int, optional
        Maximum number of characters per line for y-axis labels (default: 40).
    y_label_rotation : float, optional
        Rotation angle for y-axis labels in degrees (default: 15.0).
    y_spacing : float, optional
        Multiplier to increase vertical spacing between y-axis labels (default: 3.0).

    Returns
    -------
    None
    
    Notes
    -----
    - The function calls ``plt.show()`` to display the figure in interactive
        sessions. When ``output_path`` is provided the figure is saved to disk
        before being shown.
    - The function attempts to provide extra left margin for long wrapped
        y-axis labels; callers running in headless/non-interactive environments
        may want to set a non-default ``figsize`` and skip calling ``plt.show``
        by capturing the figure object instead.
    """

    # Sort by proportion and take top N (exclude primary diagnosis)
    data_to_plot = code_distribution.sort_values('proportion', ascending=False).copy()
    data_to_plot = data_to_plot.iloc[1:]  # skip first row (primary diagnosis)
    data_to_plot = data_to_plot[data_to_plot['proportion'] >= proportion_threshold]
    if data_to_plot.empty:
        return

    # Increase figure height according to y_spacing so labels don't overlap
    fig = plt.figure(figsize=(figsize[0], max(figsize[1], figsize[1] * y_spacing)))
    ax = fig.add_subplot(1, 1, 1)

    # Prepare labels and positions
    labels = data_to_plot['meaning'].astype(str).tolist()
    def _wrap_label(s: str, width: int) -> str:
        if not isinstance(s, str) or s.strip() == "":
            return ""
        return textwrap.fill(s, width=width)
    wrapped_labels = [_wrap_label(s, wrap_width) for s in labels]

    n = len(data_to_plot)
    base_pos = np.arange(n)
    y_positions = base_pos * y_spacing

    # Draw horizontal bars manually (more control over y positions than seaborn)
    proportions = data_to_plot['proportion'].tolist()[::-1]  # reverse to have largest on top
    colors = sns.color_palette("viridis", n)
    ax.barh(y_positions, proportions, color=colors)

    # Set yticks to our spaced positions and labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(wrapped_labels[::-1], rotation=y_label_rotation, ha='right')  # reverse labels back to match bars

    # Add labels and title
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Proportion of Cohort", fontsize=12)
    ax.set_ylabel("ICD-10 Code And Meaning", fontsize=12)

    # Add percentage labels to the end of bars
    for y, prop in zip(y_positions, proportions):
        ax.text(prop + 0.001, y, f'{prop:.1%}', va='center', fontsize=10)

    # Adjust limits and layout
    ax.set_ylim(y_positions.min() - 0.5 * y_spacing, y_positions.max() + 0.5 * y_spacing)
    plt.tight_layout()
    # leave enough left margin for wrapped, rotated labels
    plt.subplots_adjust(left=0.25)

    if output_path:
        plt.savefig(output_path, dpi='figure', bbox_inches='tight', format='svg')
        print(f"Plot saved to {output_path}")

    plt.show()

def plot_comorbidity_categories(
    letter_stats: dict,
    categories: list = ['E', 'F', 'G', 'I', 'K', 'R', 'Z'],
    title: str = "ICD-10 Letter Category Distribution",
    figsize: tuple = (10, 6),
    output_path: str = None
) -> None:
    """
    Plot a bar chart with a line plot for ICD-10 letter category distribution from a letter stats dictionary.

    Parameters
    ----------
    letter_stats : dict
        Dictionary returned by `letter_stats_overall`, containing counts and proportions for each letter category.
    categories : list
        List of letter categories to include in the plot (default: ['E', 'F', 'G', 'I']).
    title : str, optional
        Title of the plot (default: "ICD-10 Letter Category Distribution").
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (10, 6)).
    output_path : str, optional
        If provided, save the plot to this file path (default: None).

    Returns
    -------
    None
    """

    icd_10_chapters = {
        'E': 'Endocrine, nutritional and metabolic diseases',
        'F': 'Mental and behavioural disorders',
        'G': 'Diseases of the nervous system',
        'I': 'Diseases of the circulatory system',
        'K': 'Diseases of the digestive system',
        'R': 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
        'Z': 'Factors influencing health status and contact with health services'
    }

    counts = [letter_stats.get(f"{cat.lower()}_count", 0) for cat in categories]
    proportions = [letter_stats.get(f"{cat.lower()}_proportion", 0.0) for cat in categories]
    percentages = [p * 100 for p in proportions]  # convert to percentage

    se = np.sqrt(np.array(percentages) * (100 - np.array(percentages)) / max(1, letter_stats.get('total_letters', 1)))

    fig, ax1 = plt.subplots(figsize=figsize)

    # Prepare labels with wrapping for better readability
    labels = [textwrap.fill(icd_10_chapters.get(cat, cat), width=15) for cat in categories]

    # Bar plot for counts
    ax1.bar(labels, counts, color='black', label='Count')
    ax1.set_xlabel('ICD-10 Category')
    ax1.set_ylabel('Count', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    # Line plot for proportions on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(labels, percentages, color='blue', marker='d', label='Percentage')
    ax2.set_ylabel('Percentage', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.errorbar(labels, percentages, yerr=se, fmt='none', color='red', capsize=5, label='Standard Error')

    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi='figure', bbox_inches='tight', format='svg')
        print(f"Plot saved to {output_path}")

    plt.show()

# ==============================================================================
# Actually processing codes and determining stats and plotting
# ==============================================================================
# 1: Comorbidity distribution for each SFC external cluster
# Get the ICD-10 code distribution for each SFC external cluster
code_distribution_cluster_0 = count_codes_in_cohort(
    combined_df[combined_df['sfc_external_cluster'] == 'Cluster 0'],
)
logging.info(f"Code distribution for Cluster 0:\n{code_distribution_cluster_0.head()}")

code_distribution_cluster_1 = count_codes_in_cohort(
    combined_df[combined_df['sfc_external_cluster'] == 'Cluster 1'],
)
logging.info(f"Code distribution for Cluster 1:\n{code_distribution_cluster_1.head()}")
# Test significant differences in code distributions between clusters
# First add comorbidity indicator columns to the combined_df for each cluster
combined_df_with_indicators = build_comorbidity_indicator_matrix(
    combined_df,
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
# Determine stats of distinct ICD-10 categories (letter counts) for each cluster
f_letter_stats_overall_cluster_0 = letter_stats_overall(
    combined_df[combined_df['sfc_external_cluster'] == 'Cluster 0'],
    codes_column_cohort='codes'
)
logging.info(f"Letter stats for Cluster 0: {f_letter_stats_overall_cluster_0}")

f_letter_stats_overall_cluster_1 = letter_stats_overall(
    combined_df[combined_df['sfc_external_cluster'] == 'Cluster 1'],
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

# 3: Range distribution for each SFC external cluster
# Determine stats for specific ICD-10 code ranges (e.g., R40-R46) for each cluster
range_stats_overall_cluster_0 = range_stats_overall(
    combined_df[combined_df['sfc_external_cluster'] == 'Cluster 0'],
    codes_column_cohort='codes',
    letter='R',
    low=40,
    high=46
)
logging.info(f"Range stats for Cluster 0 (R40-R46): {range_stats_overall_cluster_0}")

range_stats_overall_cluster_1 = range_stats_overall(
    combined_df[combined_df['sfc_external_cluster'] == 'Cluster 1'],
    codes_column_cohort='codes',
    letter='R',
    low=40,
    high=46
)
logging.info(f"Range stats for Cluster 1 (R40-R46): {range_stats_overall_cluster_1}")
# Test significant differences in range distributions between clusters
# First, create a DataFrame with counts of the target range and other codes for each cluster
range_counts_df = pd.DataFrame({
    'category': ['R40-R46'],
    'count_cluster_0': [range_stats_overall_cluster_0.get('range_count', 0)],
    'count_cluster_1': [range_stats_overall_cluster_1.get('range_count', 0)],
})
n_cluster_0 = range_stats_overall_cluster_0.get('total_letters', 1)
n_cluster_1 = range_stats_overall_cluster_1.get('total_letters', 1)
# Now test
range_results = compare_category_proportions(
    range_counts_df,
    category_col='category',
    count_col_g1='count_cluster_0',
    count_col_g2='count_cluster_1',
    n_g1=n_cluster_0,
    n_g2=n_cluster_1,
    group_labels=('Cluster 0', 'Cluster 1'),
)
logging.info(f"Range comparison results:\n{range_results}")  # all insignificant (p_adj > 0.05) after FDR correction, so no significant differences between clusters 