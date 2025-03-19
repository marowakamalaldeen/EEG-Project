from all_function import load_csv_data
from all_function import load_video_data 

def load_video_data():
    """
    Manually enter the file path to load brain activation video data (.npy file).
    Returns:
        np.ndarray: Loaded video data (if not found, dummy data is returned).
    """
    file_path = input("Enter the full file path for the video data (.npy file): ").strip()
    if not os.path.exists(file_path):
        print(f"❌ Warning: File not found at {file_path}. Returning dummy data.")
        return np.random.randn(31335, 11250)  # Dummy fallback data (voxels x time)
    return np.load(file_path)

def visualize_heatmap():
    """
    Load the video data using the file path provided by the user and visualize it as a heatmap.
    """
    videodata = load_video_data()
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca()
    ax.imshow(videodata, aspect='auto')
    plt.tight_layout()
    plt.xlabel('X-axis', fontsize=10)
    plt.ylabel('Y-axis', fontsize=10)
    plt.title('Heat map of brain activation over time')
    plt.xticks(rotation=90)
    plt.show()

visualize_heatmap()
