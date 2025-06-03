import kagglehub

# to use Kaggle's api, need to put api credentials files kaggle.json in:
# /home/vito/fortunom/.config/kaggle --> /home/vito/fortunom/.config/kaggle/kaggle.json

# after download, move the data from cache to desired path:
# mv /path/to/current/folder /path/to/new/folder/

# Download latest version
path = kagglehub.dataset_download("hgunraj/covidxct")

print("Path to dataset files:", path)
