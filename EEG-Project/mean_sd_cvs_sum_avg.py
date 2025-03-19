from all_function import load_csv_data
from all_function import load_video_data 

def compute_summary_stats(dataset):
    """
    Compute summary statistics for each column in a dataset.
    
    Parameters:
        dataset (np.ndarray): Array with shape (voxels, time).
    
    Returns:
        dict: Dictionary containing arrays for:
              "Mean", "SD", "CV", "SUM", and "Average" (across time for each voxel).
    """
    means = np.mean(dataset, axis=0)
    sds = np.std(dataset, axis=0)
    cvs = sds / means
    sums = np.sum(dataset, axis=0)
    averages = np.mean(dataset, axis=1)
    
    return {
        "Mean": means,
        "SD": sds,
        "CV": cvs,
        "SUM": sums,
        "Average": averages
    }

def compute_all_summary_stats(datasets, dataset_labels):
    """
    Compute summary statistics for a list of datasets.
    
    Parameters:
        datasets (list of np.ndarray): List of video datasets.
        dataset_labels (list of str): Labels for each dataset.
    
    Returns:
        dict: Dictionary mapping each label to its computed summary statistics.
    """
    summary_stats = {}
    for i, dataset in enumerate(datasets):
        summary_stats[dataset_labels[i]] = compute_summary_stats(dataset)
    return summary_stats

def plot_summary_stats(summary_stats):
    """
    Visualize the Mean, and SUM metrics for each dataset using bar charts.
    
    Parameters:
        summary_stats (dict): Dictionary of summary statistics for each dataset.
    """
    for label, stats in summary_stats.items():
        # Plot Mean
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(stats["Mean"])), stats["Mean"], color='skyblue', label="Mean")
        plt.xlabel('Columns')
        plt.ylabel('Mean Values')
        plt.title(f'Mean for {label}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot SUM
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(stats["SUM"])), stats["SUM"], color='green', label="SUM")
        plt.xlabel('Columns')
        plt.ylabel('Sum Values')
        plt.title(f'SUM for {label}')
        plt.legend()
        plt.tight_layout()
        plt.show()

def perform_statistical_tests(summary_stats, dataset_labels):
    """
    Perform ANOVA and pairwise t-tests on the Mean and SUM metrics across datasets.
    
    Parameters:
        summary_stats (dict): Dictionary of summary statistics for each dataset.
        dataset_labels (list of str): List of dataset labels.
    """
    print("ANOVA Results (Mean Comparison):")
    anova_results_mean = f_oneway(
        summary_stats[dataset_labels[0]]["Mean"],
        summary_stats[dataset_labels[1]]["Mean"],
        summary_stats[dataset_labels[2]]["Mean"],
        summary_stats[dataset_labels[3]]["Mean"]
    )
    print(f"F-statistic: {anova_results_mean.statistic:.4f}, P-value: {anova_results_mean.pvalue:.4f}")
    if anova_results_mean.pvalue < 0.05:
        print("Statistically significant difference between means.")
    else:
        print("No statistically significant difference between means.")
    
    print("\nANOVA Results (SUM Comparison):")
    anova_results_sum = f_oneway(
        summary_stats[dataset_labels[0]]["SUM"],
        summary_stats[dataset_labels[1]]["SUM"],
        summary_stats[dataset_labels[2]]["SUM"],
        summary_stats[dataset_labels[3]]["SUM"]
    )
    print(f"F-statistic: {anova_results_sum.statistic:.4f}, P-value: {anova_results_sum.pvalue:.4f}")
    if anova_results_sum.pvalue < 0.05:
        print("Statistically significant difference between sums.")
    else:
        print("No statistically significant difference between sums.")
    
    print("\nPairwise T-tests (Mean Comparison):")
    for i in range(len(dataset_labels)):
        for j in range(i + 1, len(dataset_labels)):
            t_stat, t_pval = ttest_ind(
                summary_stats[dataset_labels[i]]["Mean"],
                summary_stats[dataset_labels[j]]["Mean"]
            )
            print(f"{dataset_labels[i]} vs {dataset_labels[j]}: t-stat = {t_stat:.4f}, p-value = {t_pval:.4f}")
    
    print("\nPairwise T-tests (SUM Comparison):")
    for i in range(len(dataset_labels)):
        for j in range(i + 1, len(dataset_labels)):
            t_stat, t_pval = ttest_ind(
                summary_stats[dataset_labels[i]]["SUM"],
                summary_stats[dataset_labels[j]]["SUM"]
            )
            print(f"{dataset_labels[i]} vs {dataset_labels[j]}: t-stat = {t_stat:.4f}, p-value = {t_pval:.4f}")

def main():
    """
    Main execution flow:
    1. Load the datasets using load_video_data().
    2. Compute summary statistics for each dataset.
    3. Visualize the summary statistics.
    4. Perform statistical tests.
    """
    # Here you can load your actual datasets. For now, we'll simulate with dummy data.
    videodata = load_video_data()
    videodata1 = load_video_data()
    videodata2 = load_video_data()
    videodata3 = load_video_data()
    
    datasets = [videodata, videodata1, videodata2, videodata3]
    dataset_labels = ["videodata", "videodata1", "videodata2", "videodata3"]
    
    # Step 2: Calculate summary statistics for each dataset.
    summary_stats = compute_all_summary_stats(datasets, dataset_labels)
    
    # Step 3: Visualize the summary statistics.
    plot_summary_stats(summary_stats)
    
    # Step 4: Perform statistical comparisons.
    perform_statistical_tests(summary_stats, dataset_labels)
    
    # Optionally, combine all features into one CSV for further analysis.
    combined = []
    for label, stats in summary_stats.items():
        df = pd.DataFrame(stats)
        df["Dataset"] = label
        combined.append(df)
    combined_df = pd.concat(combined, ignore_index=True)
    combined_df.to_csv("combined_features.csv", index=False)
    print("Combined features saved to 'combined_features.csv'.")

if __name__ == "__main__":
    main()