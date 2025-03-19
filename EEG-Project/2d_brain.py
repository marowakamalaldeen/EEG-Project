from all_function import load_csv_data
from all_function import load_video_data 

# Clone and install the repository first!
# !git clone https://github.com/kdotdot/cerebra_atlas_python.git
# %cd cerebra_atlas_python
# !pip install .

# Load video eLORETA data
try:
    videodata = np.load("/content/drive/MyDrive/Data TU PHD DUBLIN/First data/video1_eLORETA.npy")  # Shape: (time, 11250)
    print(f"Loaded video data with shape: {videodata.shape}")
except FileNotFoundError:
    print("Error: video1_eLORETA.npy file not found. Please check the file path.")
    raise
videodata = videodata.T

# Load the region mapping
try:
    map_voxel = np.load("/content/drive/MyDrive/Data TU PHD DUBLIN/LABEL DETAILS/map_voxel.npy")  # Shape: (31553,)
    print(f"Loaded map_voxel data with shape: {map_voxel.shape}")
except FileNotFoundError:
    print("Error: map_voxel.npy file not found. Please check the file path.")
    raise

# Align map_voxel with videodata
if len(map_voxel) != videodata.shape[1]:
    map_voxel = map_voxel[:videodata.shape[1]]  # Trim map_voxel to match videodata's voxels
    print(f"Aligned map_voxel shape: {map_voxel.shape}")

# Load region names from a CSV file
try:
    region_names = pd.read_csv("/content/drive/MyDrive/Data TU PHD DUBLIN/LABEL DETAILS/results_per_region (MY 101124).csv")
    print(f"Loaded region names with shape: {region_names.shape}")
except FileNotFoundError:
    print("Error: results_per_region (MY 101124).csv file not found. Please check the file path.")
    raise

# Debugging CSV column names
print("Region names CSV columns:", region_names.columns)

# Ensure region names CSV contains 'Cerebra_ID' and 'Region_name' columns
if "Cerebra_ID" not in region_names.columns or "Region_name" not in region_names.columns:
    raise ValueError("CSV file must contain 'Cerebra_ID' and 'Region_name' columns.")

# Create a dictionary for region ID to region name mapping
region_id_to_name = region_names.set_index("Cerebra_ID")["Region_name"].to_dict()

# Compute mean over time for each region
unique_regions = np.unique(map_voxel)
region_time_means = {}

for region in unique_regions:
    # Get indices for all voxels in the region
    region_indices = map_voxel == region
    # Compute the mean over time for voxels in this region
    region_time_means[region] = np.mean(videodata[:, region_indices], axis=1)

# Prepare data for 3D visualization
voxel_colors = np.zeros((map_voxel.shape[0], 3))  # Initialize voxel colors as black

for region in unique_regions:
    region_indices = map_voxel == region
    mean_intensity = np.mean(region_time_means[region])  # Mean intensity for the region
    color_value = plt.cm.viridis(mean_intensity / max(region_time_means[region]))[:3]  # Normalize and use colormap
    voxel_colors[region_indices] = color_value  # Assign colors to voxels



#Plots3D.plot_data_3d(plot_data=plot_data)
# Visualize 3D brain with Plots3D
try:
    Plots3D.plot_data_3d(region_time_means)
    print("3D visualization generated successfully.")
except Exception as e:
    print(f"Error generating 3D visualization: {e}")