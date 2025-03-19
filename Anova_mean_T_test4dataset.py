
from all_function import load_csv_data
from all_function import load_video_data 

def main():
    # Load four datasets
    print("Loading dataset: videodata")
    videodata = load_video_data()
    print("Loading dataset: videodata1")
    videodata1 = load_video_data()
    print("Loading dataset: videodata2")
    videodata2 = load_video_data()
    print("Loading dataset: videodata3")
    videodata3 = load_video_data()

    # List of datasets and corresponding labels
    datasets = [videodata, videodata1, videodata2, videodata3]
    dataset_labels = ["videodata", "videodata1", "videodata2", "videodata3"]

    # Step 2: Calculate Mean, SD, CV, SUM, and Average (across voxels) for each column of each dataset.
    summary_stats = {}
    for i, dataset in enumerate(datasets):
        means = np.mean(dataset, axis=0)
        sds = np.std(dataset, axis=0)
        cvs = sds / means
        sums = np.sum(dataset, axis=0)
        averages = np.mean(dataset, axis=1)
        summary_stats[dataset_labels[i]] = {
            "Mean": means,
            "SD": sds,
            "CV": cvs,
            "SUM": sums,
            "Average": averages
        }
    
    # Step 3: Visualize the Mean for each dataset using a bar chart.
    for label, stats in summary_stats.items():
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(stats["Mean"])), stats["Mean"], color='skyblue', label="Mean")
        plt.xlabel('Columns')
        plt.ylabel('Values')
        plt.title(f'Mean for {label}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Step 4: Statistical Comparison using ANOVA.
    print("ANOVA Results:")
    anova_results = f_oneway(
        summary_stats["videodata"]["Mean"],
        summary_stats["videodata1"]["Mean"],
        summary_stats["videodata2"]["Mean"],
        summary_stats["videodata3"]["Mean"]
    )
    print(f"F-statistic: {anova_results.statistic:.4f}, P-value: {anova_results.pvalue:.4f}")
    if anova_results.pvalue < 0.05:
        print("There is a statistically significant difference between the datasets.")
    else:
        print("No statistically significant difference between the datasets.")
    
    # Step 5: Pairwise Comparison using t-tests.
    print("\nPairwise T-tests:")
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            t_stat, t_pval = ttest_ind(
                summary_stats[dataset_labels[i]]["Mean"],
                summary_stats[dataset_labels[j]]["Mean"]
            )
            print(f"{dataset_labels[i]} vs {dataset_labels[j]}: t-stat = {t_stat:.4f}, p-value = {t_pval:.4f}")

if __name__ == "__main__":
    main()
