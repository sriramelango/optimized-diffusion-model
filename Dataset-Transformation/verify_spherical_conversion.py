#!/usr/bin/env python3
"""
Verify Spherical Dataset Conversion Accuracy

This script loads both the original Cartesian dataset and the new spherical dataset,
converts the spherical dataset back to Cartesian coordinates, and verifies that
they match exactly (within numerical precision).

This is a critical verification step to ensure the conversion process is lossless
and that the new spherical dataset preserves all the original training information.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# Physical constants (matching the conversion script)
THRUST = 1.0
NUM_SEGMENTS = 20


def spherical_to_cartesian_verification(alpha: np.ndarray, beta: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates back to Cartesian coordinates.
    This must match exactly the conversion used in the original conversion script.
    
    Args:
        alpha: Azimuthal angle [0, 2π]
        beta: Polar angle [0, 2π]
        r: Magnitude [0, 1]
        
    Returns:
        ux, uy, uz: Cartesian thrust components
    """
    ux = r * np.cos(alpha) * np.cos(beta)
    uy = r * np.sin(alpha) * np.cos(beta)
    uz = r * np.sin(beta)
    
    return ux, uy, uz


def unnormalize_spherical_coordinates_verification(alpha_norm: np.ndarray, beta_norm: np.ndarray, r_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unnormalize spherical coordinates from [0, 1] back to physical range.
    This must match exactly the unnormalization used in the conversion script.
    
    Args:
        alpha_norm, beta_norm, r_norm: Normalized spherical coordinates [0, 1]
        
    Returns:
        alpha: Azimuthal angle [0, 2π]
        beta: Polar angle [0, 2π]
        r: Magnitude [0, 1]
    """
    alpha = alpha_norm * 2 * np.pi
    beta = beta_norm * 2 * np.pi  
    r = r_norm  # r is already the magnitude
    
    return alpha, beta, r


def convert_spherical_back_to_cartesian(spherical_data: np.ndarray) -> np.ndarray:
    """
    Convert the spherical dataset back to Cartesian format for comparison.
    
    Args:
        spherical_data: Dataset with spherical thrust coordinates (N, 67)
        
    Returns:
        cartesian_data: Dataset converted back to Cartesian thrust coordinates (N, 67)
    """
    print("Converting spherical dataset back to Cartesian format...")
    
    # Create a copy to avoid modifying the original
    cartesian_data = spherical_data.copy()
    
    # Extract thrust variables (indices 4:64) - these are now in spherical format
    thrust_spherical = spherical_data[:, 4:64]  # Shape: (N, 60)
    thrust_reshaped = thrust_spherical.reshape(-1, NUM_SEGMENTS, 3)  # Shape: (N, 20, 3)
    
    print(f"Spherical thrust data shape: {thrust_spherical.shape}")
    print(f"Spherical thrust reshaped: {thrust_reshaped.shape}")
    
    # Extract normalized spherical components
    alpha_norm = thrust_reshaped[:, :, 0]  # Shape: (N, 20)
    beta_norm = thrust_reshaped[:, :, 1]   # Shape: (N, 20)
    r_norm = thrust_reshaped[:, :, 2]      # Shape: (N, 20)
    
    print(f"Spherical components ranges:")
    print(f"  Alpha_norm: [{np.min(alpha_norm):.6f}, {np.max(alpha_norm):.6f}]")
    print(f"  Beta_norm: [{np.min(beta_norm):.6f}, {np.max(beta_norm):.6f}]")
    print(f"  R_norm: [{np.min(r_norm):.6f}, {np.max(r_norm):.6f}]")
    
    # Unnormalize spherical coordinates
    alpha, beta, r = unnormalize_spherical_coordinates_verification(alpha_norm, beta_norm, r_norm)
    
    print(f"Physical spherical components ranges:")
    print(f"  Alpha: [{np.min(alpha):.6f}, {np.max(alpha):.6f}]")
    print(f"  Beta: [{np.min(beta):.6f}, {np.max(beta):.6f}]")
    print(f"  R: [{np.min(r):.6f}, {np.max(r):.6f}]")
    
    # Convert to Cartesian coordinates
    ux, uy, uz = spherical_to_cartesian_verification(alpha, beta, r)
    
    print(f"Physical Cartesian components ranges:")
    print(f"  Ux: [{np.min(ux):.6f}, {np.max(ux):.6f}]")
    print(f"  Uy: [{np.min(uy):.6f}, {np.max(uy):.6f}]")
    print(f"  Uz: [{np.min(uz):.6f}, {np.max(uz):.6f}]")
    
    # Verify magnitudes are within bounds
    magnitudes = np.sqrt(ux**2 + uy**2 + uz**2)
    print(f"Reconstructed magnitudes range: [{np.min(magnitudes):.6f}, {np.max(magnitudes):.6f}]")
    print(f"Magnitudes > 1.0: {np.sum(magnitudes > 1.0)} (should be 0!)")
    
    # Normalize back to [0, 1] range (reverse of: thrust_unnormalized = thrust_reshaped * 2 * THRUST - THRUST)
    ux_norm = (ux + THRUST) / (2 * THRUST)
    uy_norm = (uy + THRUST) / (2 * THRUST)
    uz_norm = (uz + THRUST) / (2 * THRUST)
    
    print(f"Normalized Cartesian components ranges:")
    print(f"  Ux_norm: [{np.min(ux_norm):.6f}, {np.max(ux_norm):.6f}]")
    print(f"  Uy_norm: [{np.min(uy_norm):.6f}, {np.max(uy_norm):.6f}]")
    print(f"  Uz_norm: [{np.min(uz_norm):.6f}, {np.max(uz_norm):.6f}]")
    
    # Reshape back to (N, 60) format
    cartesian_normalized = np.stack([ux_norm, uy_norm, uz_norm], axis=2)  # Shape: (N, 20, 3)
    cartesian_flattened = cartesian_normalized.reshape(-1, 60)  # Shape: (N, 60)
    
    # Replace thrust data in the cartesian dataset
    cartesian_data[:, 4:64] = cartesian_flattened
    
    return cartesian_data


def compare_datasets(original_data: np.ndarray, reconstructed_data: np.ndarray) -> Dict:
    """
    Compare the original and reconstructed datasets to verify they match.
    
    Args:
        original_data: Original Cartesian dataset (N, 67)
        reconstructed_data: Reconstructed Cartesian dataset (N, 67)
        
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "=" * 80)
    print("COMPARING ORIGINAL AND RECONSTRUCTED DATASETS")
    print("=" * 80)
    
    # Check shapes
    if original_data.shape != reconstructed_data.shape:
        print(f"❌ SHAPE MISMATCH: Original {original_data.shape} vs Reconstructed {reconstructed_data.shape}")
        return {'shapes_match': False}
    
    print(f"✅ Shapes match: {original_data.shape}")
    
    # Compare each component separately
    results = {
        'shapes_match': True,
        'halo_energy_match': True,
        'time_vars_match': True,
        'thrust_vars_match': True,
        'mass_vars_match': True,
        'overall_match': True,
        'max_difference': 0,
        'mean_difference': 0,
        'component_differences': {}
    }
    
    # Component ranges (matching the 67-vector structure)
    components = {
        'halo_energy': (0, 1),      # [0:1] - halo energy (class label)
        'time_vars': (1, 4),        # [1:4] - time variables  
        'thrust_vars': (4, 64),     # [4:64] - thrust variables (60 values)
        'mass_vars': (64, 67)       # [64:67] - mass variables (3 values)
    }
    
    all_differences = []
    
    for comp_name, (start, end) in components.items():
        original_comp = original_data[:, start:end]
        reconstructed_comp = reconstructed_data[:, start:end]
        
        # Compute differences
        diff = np.abs(original_comp - reconstructed_comp)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Store results
        results['component_differences'][comp_name] = {
            'max_difference': max_diff,
            'mean_difference': mean_diff,
            'shape': original_comp.shape
        }
        
        all_differences.extend(diff.flatten())
        
        # Check if components match within tolerance
        tolerance = 1e-10  # Very strict tolerance
        matches = max_diff < tolerance
        results[f'{comp_name}_match'] = matches
        
        print(f"{comp_name:12} [{start:2d}:{end:2d}]: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, matches={matches}")
        
        if not matches:
            results['overall_match'] = False
    
    # Overall statistics
    all_differences = np.array(all_differences)
    results['max_difference'] = np.max(all_differences)
    results['mean_difference'] = np.mean(all_differences)
    
    print(f"\nOVERALL COMPARISON:")
    print(f"  Maximum difference: {results['max_difference']:.2e}")
    print(f"  Mean difference: {results['mean_difference']:.2e}")
    print(f"  Overall match: {results['overall_match']}")
    
    # Special focus on thrust variables (most critical)
    if not results['thrust_vars_match']:
        print(f"\n⚠️  THRUST VARIABLES MISMATCH DETECTED!")
        thrust_orig = original_data[:, 4:64]
        thrust_recon = reconstructed_data[:, 4:64]
        thrust_diff = np.abs(thrust_orig - thrust_recon)
        
        # Find worst mismatches
        worst_indices = np.unravel_index(np.argmax(thrust_diff), thrust_diff.shape)
        print(f"  Worst mismatch at sample {worst_indices[0]}, component {worst_indices[1]}")
        print(f"  Original value: {thrust_orig[worst_indices]:.10f}")
        print(f"  Reconstructed value: {thrust_recon[worst_indices]:.10f}")
        print(f"  Difference: {thrust_diff[worst_indices]:.2e}")
    
    return results


def create_verification_plots(original_data: np.ndarray, reconstructed_data: np.ndarray, output_dir: str = "."):
    """
    Create plots to visualize the comparison between original and reconstructed datasets.
    
    Args:
        original_data: Original dataset
        reconstructed_data: Reconstructed dataset  
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("CREATING VERIFICATION PLOTS")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Focus on thrust variables (most critical)
    original_thrust = original_data[:, 4:64]
    reconstructed_thrust = reconstructed_data[:, 4:64]
    differences = np.abs(original_thrust - reconstructed_thrust)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Difference distribution
    axes[0, 0].hist(differences.flatten(), bins=50, alpha=0.7, color='red', density=True)
    axes[0, 0].set_title('Distribution of Absolute Differences')
    axes[0, 0].set_xlabel('Absolute Difference')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of original vs reconstructed (sample of thrust values)
    sample_size = min(10000, original_thrust.size)
    sample_indices = np.random.choice(original_thrust.size, size=sample_size, replace=False)
    orig_flat = original_thrust.flatten()
    recon_flat = reconstructed_thrust.flatten()
    
    axes[0, 1].scatter(orig_flat[sample_indices], recon_flat[sample_indices], alpha=0.1, s=1)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Match (y=x)')
    axes[0, 1].set_title('Original vs Reconstructed Thrust Values')
    axes[0, 1].set_xlabel('Original Value')
    axes[0, 1].set_ylabel('Reconstructed Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Maximum difference per sample
    max_diff_per_sample = np.max(differences, axis=1)
    axes[1, 0].hist(max_diff_per_sample, bins=50, alpha=0.7, color='blue', density=True)
    axes[1, 0].set_title('Maximum Difference per Sample')
    axes[1, 0].set_xlabel('Maximum Absolute Difference')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Difference by component index
    mean_diff_per_component = np.mean(differences, axis=0)
    axes[1, 1].plot(mean_diff_per_component, 'o-', markersize=3)
    axes[1, 1].set_title('Mean Difference by Thrust Component Index')
    axes[1, 1].set_xlabel('Component Index (0-59)')
    axes[1, 1].set_ylabel('Mean Absolute Difference')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'dataset_verification_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Verification plots saved to: {plot_path}")


def main():
    """
    Main verification function.
    """
    print("=" * 80)
    print("SPHERICAL DATASET CONVERSION VERIFICATION")
    print("=" * 80)
    
    # File paths
    original_path = "GTO_Halo_DM/data/training_data_boundary_100000.pkl"
    spherical_path = "GTO_Halo_DM/data/training_data_boundary_100000_spherical.pkl"
    output_dir = "spherical_verification_analysis"
    
    # Check if files exist
    if not os.path.exists(original_path):
        print(f"❌ ERROR: Original dataset not found: {original_path}")
        return
    
    if not os.path.exists(spherical_path):
        print(f"❌ ERROR: Spherical dataset not found: {spherical_path}")
        return
    
    # Load datasets
    print(f"Loading original dataset from: {original_path}")
    with open(original_path, 'rb') as f:
        original_data = pickle.load(f)
    
    print(f"Loading spherical dataset from: {spherical_path}")
    with open(spherical_path, 'rb') as f:
        spherical_data = pickle.load(f)
    
    print(f"Original dataset shape: {original_data.shape}")
    print(f"Spherical dataset shape: {spherical_data.shape}")
    
    # Convert spherical dataset back to Cartesian
    print("\n" + "=" * 80)
    print("CONVERTING SPHERICAL DATASET BACK TO CARTESIAN")
    print("=" * 80)
    
    reconstructed_data = convert_spherical_back_to_cartesian(spherical_data)
    
    print(f"Reconstructed dataset shape: {reconstructed_data.shape}")
    
    # Compare datasets
    comparison_results = compare_datasets(original_data, reconstructed_data)
    
    # Create verification plots
    create_verification_plots(original_data, reconstructed_data, output_dir)
    
    # Save comparison results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'verification_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(comparison_results, f)
    
    # Print final verdict
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    if comparison_results['overall_match']:
        print("🎉 SUCCESS: Spherical conversion is LOSSLESS!")
        print("✅ Original and reconstructed datasets match within numerical precision")
        print("✅ The spherical dataset preserves all original training information")
        print("✅ Safe to use the spherical dataset for training")
    else:
        print("❌ WARNING: Datasets do not match exactly!")
        print("⚠️  There may be precision loss in the conversion process")
        print(f"⚠️  Maximum difference: {comparison_results['max_difference']:.2e}")
        
        # Check if differences are within acceptable tolerance
        if comparison_results['max_difference'] < 1e-6:
            print("ℹ️  However, differences are very small and likely due to floating-point precision")
            print("ℹ️  This level of difference should not affect training")
        else:
            print("⚠️  Differences are significant and may affect training quality")
    
    print(f"📊 Detailed results saved to: {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()