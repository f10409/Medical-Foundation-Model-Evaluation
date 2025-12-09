import pandas as pd
import numpy as np
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
from scipy import stats
import math


def calculate_sensitivity(group):
    tp = ((group['preds'] == 1) & (group['labels'] == 1)).sum()
    fn = ((group['preds'] == 0) & (group['labels'] == 1)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calculate_iou_mean(group):

    return group.iou.mean()

def calculate_dice_mean(group):

    return group.dice.mean()

def calculate_CI_width(scores):
    # Calculate 95% confidence intervals using t-distribution
    n_folds = 5
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, n_folds - 1)
    
    # confidence interval
    std = np.std(scores, ddof=1)
    se = std / np.sqrt(n_folds)
    
    return t_critical * se

def median_confidence_interval(group, confidence_level=0.95):
    """
    Calculate median confidence interval for grouped data.
    Returns results that will become separate columns when used with groupby().apply()
    
    Parameters:
    -----------
    group : pandas.Series
        The grouped data (pandas passes this automatically)
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95% confidence interval)
    
    Returns:
    --------
    pandas.Series
        Index will become column names: median, lower_ci, upper_ci
    """
    # Remove NaN values and sort
    clean_data = group.dropna().sort_values()
    n = len(clean_data)
    
    # Handle edge cases
    if n == 0:
        return pd.Series({
            'median': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan
        })
    
    if n == 1:
        value = clean_data.iloc[0]
        return pd.Series({
            'median': value,
            'ci_lower': value,
            'ci_upper': value
        })
    
    # Calculate confidence interval positions
    q = 0.5  # For median
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha/2)
    
    nq = n * q
    se_term = z * math.sqrt(n * q * (1 - q))
    
    j_raw = nq - se_term
    k_raw = nq + se_term
    
    # Round up (ceiling) and bound within [1, n]
    j = max(1, min(n, math.ceil(j_raw)))
    k = max(1, min(n, math.ceil(k_raw)))
    
    # Get actual values (convert to 0-indexed)
    lower_bound = clean_data.iloc[j - 1]
    upper_bound = clean_data.iloc[k - 1]
    median_value = clean_data.median()
    
    return pd.Series({
        'median': median_value,
        'ci_lower': lower_bound,
        'ci_upper': upper_bound
    })

def calculate_subgroup_metric_stats(df_sub, groupby_column, metric_function, calculate_CI_width, metric_name='Metric'):
    """
    Calculate metric statistics with confidence intervals for subgroups.
    
    Parameters:
    -----------
    df_sub : pandas.DataFrame
        The input dataframe containing the data
    groupby_column : str
        The column name to group by (e.g., 'Volume Quartile')
    metric_function : function
        Function to calculate the metric for each group (e.g., calculate_sensitivity, calculate_specificity, etc.)
    calculate_CI_width : function
        Function to calculate confidence interval width (t*se)
    metric_name : str, optional
        Name of the metric for the output column (default: 'Metric')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with metric statistics including confidence intervals
    """
    
    # Calculate metric for each subgroup
    subgroup_metric = df_sub.groupby(['fold', 'model', 'freeze', groupby_column]).apply(metric_function)
    subgroup_metric = subgroup_metric.reset_index(name=metric_name)
    
    # Calculate average metric
    avg_metric = subgroup_metric[['model', 'freeze', groupby_column, metric_name]].groupby(['model', 'freeze', groupby_column]).mean().reset_index()
    
    # Calculate confidence interval width
    t_se = subgroup_metric[['model', 'freeze', groupby_column, metric_name]].groupby(['model', 'freeze', groupby_column]).apply(calculate_CI_width)
    t_se = t_se.reset_index().rename(columns={metric_name: 't*se'})
    
    # Merge and calculate confidence intervals
    avg_metric = avg_metric.merge(t_se, on=['model', 'freeze', groupby_column])
    avg_metric['ci_lower'] = avg_metric[metric_name] - avg_metric['t*se']
    avg_metric['ci_upper'] = avg_metric[metric_name] + avg_metric['t*se']
    avg_metric = avg_metric.drop(columns=['t*se'])
    
    return avg_metric 


def calculate_subgroup_metric_median_stats(df_sub, groupby_column, metric_function, median_confidence_interval, metric_name='Metric'):
    """
    Calculate metric statistics with median confidence intervals for subgroups.
    
    Parameters:
    -----------
    df_sub : pandas.DataFrame
        The input dataframe containing the data
    groupby_column : str
        The column name to group by (e.g., 'Volume Quartile')
    metric_function : function
        Function to calculate the metric for each group (e.g., calculate_sensitivity, calculate_specificity, etc.)
    metric_name : str, optional
        Name of the metric for the output column (default: 'Metric')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with median metric statistics including confidence intervals
    """
    
    # Calculate metric for each subgroup
    subgroup_metric = df_sub.groupby(['fold', 'model', 'freeze', groupby_column]).apply(metric_function)
    subgroup_metric = subgroup_metric.reset_index(name=metric_name)
    
    # Calculate median and confidence intervals using the median_confidence_interval function
    median_stats = (subgroup_metric[['model', 'freeze', groupby_column, metric_name]]
                   .groupby(['model', 'freeze', groupby_column])[metric_name]
                   .apply(median_confidence_interval)
                   .unstack().reset_index())
    
    # Rename columns to match expected output format
    median_stats = median_stats.rename(columns={
        'median': metric_name
    })
    
    return median_stats


def friedman_with_posthoc(data, block_col='Block', treatment_col='Treatment', score_col='Score', alpha=0.05):
    """
    Perform Friedman test with optional Nemenyi post-hoc test.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the experimental data
    block_col : str, default 'Block'
        Name of the column containing block/subject identifiers
    treatment_col : str, default 'Treatment'
        Name of the column containing treatment/condition identifiers
    score_col : str, default 'Score'
        Name of the column containing the response values
    alpha : float, default 0.05
        Significance level for the test
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'friedman_stat': Friedman test statistic
        - 'friedman_p': p-value from Friedman test
        - 'significant': boolean indicating if result is significant
        - 'nemenyi_results': Nemenyi post-hoc results (if applicable) or None
    """
    
    # Validate input data
    required_cols = [block_col, treatment_col, score_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    # Reshape data for Friedman test
    try:
        pivot_data = data.pivot(index=block_col, columns=treatment_col, values=score_col)
    except Exception as e:
        raise ValueError(f"Error reshaping data: {e}")
    
    # Check for missing values
    if pivot_data.isnull().any().any():
        print("Warning: Missing values detected. Consider removing incomplete blocks.")
    
    # Perform Friedman test
    stat, p_value = friedmanchisquare(*[pivot_data[col] for col in pivot_data.columns])
    
    # Print Friedman test results
    print(f"Friedman Test statistic: {stat:.4f}, p-value: {p_value:.4f}")
    
    # Prepare results dictionary
    results = {
        'friedman_stat': stat,
        'friedman_p': p_value,
        'significant': p_value < alpha,
        'nemenyi_results': None
    }
    
    # Perform post-hoc test if significant
    if p_value < alpha:
        print(f"\nFriedman test significant (p < {alpha}). Performing Nemenyi post-hoc test...")
        try:
            nemenyi_results = sp.posthoc_nemenyi_friedman(
                data.pivot(index=block_col, columns=treatment_col, values=score_col)
            )
            print("\nNemenyi Post-Hoc Test Results:")
            print(nemenyi_results)
            results['nemenyi_results'] = nemenyi_results
        except Exception as e:
            print(f"Error performing Nemenyi test: {e}")
            # Try alternative approach with different parameter names
            try:
                print("Trying alternative Nemenyi approach...")
                nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_data)
                print("\nNemenyi Post-Hoc Test Results:")
                print(nemenyi_results)
                results['nemenyi_results'] = nemenyi_results
            except Exception as e2:
                print(f"Alternative approach also failed: {e2}")
                print("Nemenyi post-hoc test could not be performed.")
    else:
        print(f"\nFriedman test not significant (p >= {alpha}). Nemenyi test not required.")
    
    return results