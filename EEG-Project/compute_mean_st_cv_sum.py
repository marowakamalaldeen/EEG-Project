from all_function import load_csv_data
from all_function import load_video_data 

def compute_video_metrics(videodata):
    """
    Compute basic metrics for the video data along the time axis.

    Parameters:
        videodata (np.ndarray): The video data with shape (voxels, time).

    Returns:
        dict: A dictionary containing:
            - 'mean': Mean activation over voxels for each time point.
            - 'std': Standard deviation over voxels for each time point.
            - 'cv': Coefficient of variation (std/mean) for each time point.
            - 'sum': Sum of activation over voxels for each time point.
    """
    mean_values = np.mean(videodata, axis=0)
    std_values = np.std(videodata, axis=0)
    cv_values = std_values / mean_values
    sum_values = np.sum(videodata, axis=0)

    return {
        "mean": mean_values,
        "std": std_values,
        "cv": cv_values,
        "sum": sum_values
    }

def plot_metrics(metrics):
    """
    Plot each metric (mean, std, cv, sum) as a line plot.

    Parameters:
        metrics (dict): Dictionary containing arrays for 'mean', 'std', 'cv', 'sum'.
    """
    for key, values in metrics.items():
        plt.figure(figsize=(10, 5))
        plt.plot(values, marker='o')
        plt.xlabel("Time Axis")
        plt.ylabel("Value")
        plt.title(f"{key.capitalize()} of video data")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

def main():
    """
    Main execution: Load the video data, compute metrics, and plot them.
    """
    videodata = load_video_data()
    metrics = compute_video_metrics(videodata)
    plot_metrics(metrics)

if __name__ == "__main__":