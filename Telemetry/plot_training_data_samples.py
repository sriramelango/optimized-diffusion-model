import pickle
import numpy as np
import matplotlib.pyplot as plt

# Path to your training data (relative to the main project directory)
data_path = "../GTO_Halo_DM/data/training_data_boundary_100000.pkl"

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

# Same normalization parameters as in the training dataset
mean = 0.4652
std = 0.1811

# Pad and reshape: [N, 67] -> [N, 1, 9, 9] with proper normalization
def pad_and_reshape(vec):
    # Pad to 81 values (9×9 = 81) instead of 72 (8×9)
    padded = np.pad(vec, (0, 81 - len(vec)), 'constant')
    # Apply the same normalization as the training dataset
    normalized = (padded - mean) / std
    return normalized.reshape(1, 9, 9)

images = np.stack([pad_and_reshape(arr[i]) for i in range(N)])

# Print statistics
print(f"Data shape: {arr.shape}")
print(f"Images shape: {images.shape}")
print(f"Mean: {images.mean():.4f}, Std: {images.std():.4f}, Min: {images.min():.4f}, Max: {images.max():.4f}")

# Check for extreme values after normalization
extreme_indices = []
for i in range(N):
    img = images[i]
    if (img < -3).any() or (img > 3).any():  # More than 3 standard deviations
        extreme_indices.append(i)
        print(f"Image {i} has extreme values: min={img.min():.4f}, max={img.max():.4f}")

if extreme_indices:
    print(f"Total images with extreme values: {len(extreme_indices)} / {N}")
else:
    print("All images have reasonable normalized values.")

# Check thrust values (indices 4-63) for boundary violations in original data
thrust_out_of_bounds_indices = []
for i in range(N):
    vec = arr[i]  # Original 67-vector
    thrust_values = vec[4:64]  # Indices 4-63 are thrust values
    if (thrust_values < 0).any() or (thrust_values > 1).any():
        thrust_out_of_bounds_indices.append(i)
        print(f"Sample {i} thrust values out of bounds: min={thrust_values.min():.4f}, max={thrust_values.max():.4f}")

if thrust_out_of_bounds_indices:
    print(f"Total samples with out-of-bounds thrust values: {len(thrust_out_of_bounds_indices)} / {N}")
else:
    print("All thrust values are within the [0, 1] boundary.")

# Check classifier_norm values (index 0) to verify they are different
classifier_values = []
for i in range(N):
    vec = arr[i]  # Original 67-vector
    classifier_norm = vec[0]  # Index 0 is classifier_norm
    classifier_values.append(classifier_norm)

print(f"Classifier values range: min={min(classifier_values):.4f}, max={max(classifier_values):.4f}")
print(f"Number of unique classifier values: {len(set(classifier_values))}")
print(f"First 10 classifier values: {[f'{x:.4f}' for x in classifier_values[:10]]}")

# Verify the padding is correct
print(f"\nVerifying padding:")
print(f"Original vector length: {len(arr[0])}")
print(f"Padded vector length: {len(pad_and_reshape(arr[0]).flatten())}")
print(f"Expected padded length: 81 (9×9)")

# Plot a grid of 100 random samples (10x10)
num_samples = 100
np.random.seed(42)  # For reproducibility
indices = np.random.choice(N, num_samples, replace=False)

plt.figure(figsize=(15, 15))
for i, idx in enumerate(indices):
    plt.subplot(10, 10, i+1)
    plt.imshow(images[idx, 0], cmap='gray', aspect='equal')
    plt.axis('off')
    # Add sample index as title
    plt.title(f'{idx}', fontsize=6, pad=2)
plt.suptitle("Grid of 100 Random Training Data Images (9×9, Normalized)\nWhite = higher values, Black = lower values", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Plot a few individual samples with their original values
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i in range(6):
    row = i // 3
    col = i % 3
    idx = indices[i]
    
    # Show the 9×9 image
    im = axes[row, col].imshow(images[idx, 0], cmap='gray', aspect='equal')
    axes[row, col].set_title(f'Sample {idx}\nImage (9×9, normalized)')
    axes[row, col].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[row, col], shrink=0.8)

plt.suptitle("Individual Training Samples (9×9 Normalized Images)", fontsize=14)
plt.tight_layout()
plt.show()

# Plot the distribution of normalized values
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(images.flatten(), bins=50, alpha=0.7, edgecolor='black')
plt.title('Distribution of Normalized Values')
plt.xlabel('Normalized Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(arr.flatten(), bins=50, alpha=0.7, edgecolor='black')
plt.title('Distribution of Original Values')
plt.xlabel('Original Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
# Show the first 67 values (original data) vs padded values
original_67 = arr[0][:67]
padded_81 = pad_and_reshape(arr[0]).flatten()
plt.plot(range(67), original_67, 'b-', label='Original 67 values', linewidth=2)
plt.plot(range(81), padded_81, 'r--', label='Padded 81 values (normalized)', alpha=0.7)
plt.title('Original vs Padded Values (Sample 0)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show() 