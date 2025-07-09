# Get the data
# To use in google colab via cloud
import kagglehub
path = kagglehub.dataset_download("rijulshr/pneumoniamnist")
print("Path to dataset files:", path)
path = f"{path}/pneumoniamnist.npz"

# or manual load file locally
path = "~/projects/biostats/pneumoniamnist.npz"

