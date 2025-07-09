import pickle
import numpy as np
import matplotlib.pyplot as plt

# Path to your training data
data_path = "GTO_Halo_DM/data/training_data_boundary_100000.pkl"

# Load the data
with open(data_path, "rb") as f:
    data = pickle.load(f)

# If your data is a dict, adjust accordingly
if isinstance(data, dict):
    arr = data.get("data") or data.get("X") or next(iter(data.values()))
else:
    arr = data

arr = np.array(arr, dtype=np.float32)
N = arr.shape[0]

# Pad and reshape: [N, 67] -> [N, 1, 8, 9]
def pad_and_reshape(vec):
    padded = np.pad(vec, (0, 72 - len(vec)), 'constant')
    return padded.reshape(1, 8, 9)

images = np.stack([pad_and_reshape(arr[i]) for i in range(N)])

# Print statistics
print(f"Mean: {images.mean():.4f}, Std: {images.std():.4f}, Min: {images.min():.4f}, Max: {images.max():.4f}")

# Plot a grid of 100 random samples (10x10)
num_samples = 100
np.random.seed(42)  # For reproducibility
indices = np.random.choice(N, num_samples, replace=False)

plt.figure(figsize=(12, 12))
for i, idx in enumerate(indices):
    plt.subplot(10, 10, i+1)
    plt.imshow(images[idx, 0], cmap='gray', aspect='auto')
    plt.axis('off')
plt.suptitle("Grid of 100 Random Training Data Images (as input to model)")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show() 