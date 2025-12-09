import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, roc_auc_score, r2_score, 
                           mean_squared_error, precision_recall_curve)
from sklearn.utils import resample


def bootstrap_classification_metrics(y_true, y_pred, n_bootstraps=1000, confidence_level=0.95, random_state=5656):
    """
    Calculate bootstrapped AUC and AUPRC for classification with confidence intervals.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth binary labels
    y_pred : array-like
        Predicted probabilities or scores
    n_bootstraps : int, optional (default=1000)
        Number of bootstrap samples
    confidence_level : float, optional (default=0.95)
        Confidence level for the intervals
    random_state : int, optional (default=None)
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Contains mean AUC, mean AUPRC, standard errors, and confidence intervals
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Original AUC
    original_auc = roc_auc_score(y_true, y_pred)
    
    # Original AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    original_auprc = auc(recall, precision)
    
    # Bootstrap
    bootstrap_aucs = []
    bootstrap_auprcs = []
    n_samples = len(y_true)
    
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Check if both classes are present
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        # Calculate AUC for this bootstrap sample
        bootstrap_auc = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrap_aucs.append(bootstrap_auc)
        
        # Calculate AUPRC for this bootstrap sample
        precision, recall, _ = precision_recall_curve(y_true[indices], y_pred[indices])
        bootstrap_auprc = auc(recall, precision)
        bootstrap_auprcs.append(bootstrap_auprc)
    
    # Calculate statistics
    bootstrap_aucs = np.array(bootstrap_aucs)
    mean_auc = np.mean(bootstrap_aucs)
    
    bootstrap_auprcs = np.array(bootstrap_auprcs)
    mean_auprc = np.mean(bootstrap_auprcs)
    
    # Confidence interval
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # AUC confidence interval
    auc_lower_bound = np.percentile(bootstrap_aucs, lower_percentile)
    auc_upper_bound = np.percentile(bootstrap_aucs, upper_percentile)
    
    # AUPRC confidence interval
    auprc_lower_bound = np.percentile(bootstrap_auprcs, lower_percentile)
    auprc_upper_bound = np.percentile(bootstrap_auprcs, upper_percentile)
    
    return {
        'AUROC': {
            #'original_mean': original_auc,
            'bootstrap_mean': mean_auc,
            #'bootstrap_means': list(bootstrap_aucs),
            'lower_ci': auc_lower_bound, 
            'upper_ci': auc_upper_bound
        },
        'AUPRC': {
            #'original_mean': original_auprc,
            'bootstrap_mean': mean_auprc,
            #'bootstrap_means': list(bootstrap_auprcs),
            'lower_ci': auprc_lower_bound,
            'upper_ci': auprc_upper_bound
        }
    }


def bootstrap_regression_metrics(y_true, y_pred, n_bootstraps=1000, confidence_level=0.95, random_state=5656):
    """
    Calculate bootstrapped R² and MSE for regression with confidence intervals.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    n_bootstraps : int, optional (default=1000)
        Number of bootstrap samples
    confidence_level : float, optional (default=0.95)
        Confidence level for the intervals
    random_state : int, optional (default=None)
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Contains mean R², MSE, standard errors, and confidence intervals
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Original metrics
    original_r2 = r2_score(y_true, y_pred)
    original_mse = mean_squared_error(y_true, y_pred)
    
    # Bootstrap
    bootstrap_r2s = []
    bootstrap_mses = []
    n_samples = len(y_true)
    
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
            
        # Calculate metrics for this bootstrap sample
        bootstrap_r2 = r2_score(y_true[indices], y_pred[indices])
        bootstrap_mse = mean_squared_error(y_true[indices], y_pred[indices])
        
        bootstrap_r2s.append(bootstrap_r2)
        bootstrap_mses.append(bootstrap_mse)
    
    # Calculate statistics
    bootstrap_r2s = np.array(bootstrap_r2s)
    bootstrap_mses = np.array(bootstrap_mses)
    
    mean_r2 = np.mean(bootstrap_r2s)
    mean_mse = np.mean(bootstrap_mses)
    
    # Confidence interval
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    r2_lower_bound = np.percentile(bootstrap_r2s, lower_percentile)
    r2_upper_bound = np.percentile(bootstrap_r2s, upper_percentile)
    
    mse_lower_bound = np.percentile(bootstrap_mses, lower_percentile)
    mse_upper_bound = np.percentile(bootstrap_mses, upper_percentile)
    
    return {
        'R²': {
            #'original_mean': original_r2,
            'bootstrap_mean': mean_r2, 
            #'bootstrap_means': list(bootstrap_r2s), 
            'lower_ci': r2_lower_bound, 
            'upper_ci': r2_upper_bound
        },
        'MSE': {
            #'original_mean': original_mse, 
            'bootstrap_mean': mean_mse, 
            #'bootstrap_means': list(bootstrap_mses), 
            'lower_ci': mse_lower_bound, 
            'upper_ci': mse_upper_bound
        }
    }


def bootstrap_metrics(metrics, n_iterations=1000, confidence_level=0.95, random_seed=5656):
    """
    Perform bootstrap analysis on segmentation metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary where keys are metric names (e.g., 'dice', 'iou') and 
        values are arrays of metric values for each test image
    n_iterations : int
        Number of bootstrap iterations
    confidence_level : float
        Confidence level for the intervals (0-1)
    random_seed : int or None
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary with bootstrap results for each metric
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        
    results = {}
    alpha = (1 - confidence_level) / 2
    
    for metric_name, metric_values in metrics.items():
        # Convert to numpy array if not already
        metric_values = np.array(metric_values)
        
        # Original mean
        original_mean = np.mean(metric_values)
        
        # Initialize array to store bootstrap means
        bootstrap_means = np.zeros(n_iterations)
        
        # Perform bootstrap iterations
        for i in range(n_iterations):
            # Sample with replacement
            bootstrap_sample = resample(metric_values, replace=True, n_samples=len(metric_values))
            bootstrap_means[i] = np.mean(bootstrap_sample)
        
        # Calculate confidence intervals
        lower_ci = np.percentile(bootstrap_means, 100 * alpha)
        upper_ci = np.percentile(bootstrap_means, 100 * (1 - alpha))
        
        # Store results
        results[metric_name] = {
            #'original_mean': original_mean,
            'bootstrap_mean': np.mean(bootstrap_means),
            #'bootstrap_means': list(bootstrap_means),
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        }
    
    return results


def plot_confidence_intervals(results_dict, condition=None, metric='AUROC', 
                            figsize=(12, 8), save_path=None):
    """
    Plot confidence intervals for all models and methods.
    
    Parameters:
    -----------
    results_dict : dict
        Nested dictionary with bootstrap results
    condition : str, optional
        Specific condition to plot. If None, plots all conditions
    metric : str
        Which metric to plot ('AUROC' or 'AUPRC')
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot
    """
    conditions_to_plot = [condition] if condition else list(results_dict.keys())
    
    fig, axes = plt.subplots(1, len(conditions_to_plot), figsize=figsize)
    if len(conditions_to_plot) == 1:
        axes = [axes]
    
    for idx, cond in enumerate(conditions_to_plot):
        ax = axes[idx]
        
        models = list(results_dict[cond].keys())
        methods = list(results_dict[cond][models[0]].keys())
        
        y_pos = np.arange(len(models))
        
        for i, method in enumerate(methods):
            means = []
            lower_cis = []
            upper_cis = []
            
            for model in models:
                data = results_dict[cond][model][method][metric]
                means.append(data['bootstrap_mean'])
                lower_cis.append(data['lower_ci'])
                upper_cis.append(data['upper_ci'])
            
            # Plot confidence intervals - reverse the offset so Linear Probing appears above Fine-tuning
            ax.errorbar(means, y_pos + (len(methods)-1-i)*0.2 - 0.1, 
                       xerr=[np.array(means) - np.array(lower_cis), 
                             np.array(upper_cis) - np.array(means)],
                       fmt='o', label=method, capsize=5, markersize=8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel(metric)
        ax.set_title(f'{cond} - {metric} with 95% CI')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_model_comparison_table(results_dict, metric='AUROC'):
    """
    Print a formatted table comparing all models and methods.
    
    Parameters:
    -----------
    results_dict : dict
        Nested dictionary with bootstrap results
    metric : str
        Which metric to display ('AUROC' or 'AUPRC')
    """
    print(f"\n{metric} Performance Comparison")
    print("=" * 50)
    
    for condition in results_dict.keys():
        print(f"\n{condition}:")
        print("-" * 30)
        
        models = list(results_dict[condition].keys())
        all_methods = list(results_dict[condition][models[0]].keys())
        # Ensure Linear Probing comes before Fine-tuning
        methods = []
        if 'Linear Probing' in all_methods:
            methods.append('Linear Probing')
        if 'Fine-tuning' in all_methods:
            methods.append('Fine-tuning')
        # Add any other methods that might exist
        for method in all_methods:
            if method not in methods:
                methods.append(method)
        
        # Create table
        data = []
        for model in models:
            row = [model]
            for method in methods:
                result = results_dict[condition][model][method][metric]
                row.append(f"{result['bootstrap_mean']:.3f} ({result['lower_ci']:.3f}-{result['upper_ci']:.3f})")
            data.append(row)
        
        # Print table
        headers = ['Model'] + methods
        col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *data)]
        
        # Print header
        header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_row)
        print("-" * len(header_row))
        
        # Print data rows
        for row in data:
            print(" | ".join(str(item).ljust(w) for item, w in zip(row, col_widths)))



if __name__ == "__main__":
    print("Bootstrap Metrics Analysis Module")

