

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


def load_csv_data():
    """
    Manually enter the file path to load the region mapping CSV file.
    Returns:
        pd.DataFrame: Loaded CSV data, or None if file is not found.
    """
    file_path = input("Enter the full file path for the region CSV file: ").strip()
    if not os.path.exists(file_path):
        print(f"❌ Warning: File not found at {file_path}. Please enter a valid path.")
        return None
    return pd.read_csv(file_path)