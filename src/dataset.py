import kagglehub

# Download latest version
path = kagglehub.dataset_download("robertnowak/bowel-sounds")

print("Path to dataset files:", path)
