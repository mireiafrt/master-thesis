import os

def clean_mac_junk(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.startswith("._") or filename == ".DS_Store":
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Could not delete {file_path}: {e}")

# Replace with the desireed data folder path
data_folder = '/Volumes/MIREIA/M-THESIS/DATA/filtered_images'
clean_mac_junk(data_folder)
