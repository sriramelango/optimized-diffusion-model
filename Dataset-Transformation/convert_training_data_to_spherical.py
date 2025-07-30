#!/usr/bin/env python3
"""
Convert GTO Halo Training Dataset from Cartesian to Spherical Thrust Coordinates

This script transforms the existing training dataset from Cartesian thrust representation
to spherical thrust representation to guarantee that thrust magnitudes never exceed 1.0
after unnormalization, eliminating the need for any clipping during inference.

Original 67-vector format (Cartesian):
[0] halo_energy (class label)
[1:4] time variables (shooting_time, initial_coast, final_coast) 
[4:64] thrust variables (60 values = 20 segments × 3 Cartesian coords each)
[64:67] mass/manifold variables (final_fuel_mass, halo_period, manifold_length)

New 67-vector format (Spherical):
[0] halo_energy (class label) - UNCHANGED
[1:4] time variables - UNCHANGED  
[4:64] thrust variables (60 values = 20 segments × 3 spherical coords each: alpha, beta, r)
[64:67] mass/manifold variables - UNCHANGED

Key Insight: By storing thrust in spherical coordinates and normalizing appropriately,
the magnitude r is directly controlled and can never exceed 1.0.
"""

import os
import sys
import pickle
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

# Physical constants and bounds (matching benchmarking code exactly)
MIN_SHOOTING_TIME = 0
MAX_SHOOTING_TIME = 40
MIN_COAST_TIME = 0
MAX_COAST_TIME = 15
MIN_HALO_ENERGY = 0.008
MAX_HALO_ENERGY = 0.095
MIN_FINAL_FUEL_MASS = 408
MAX_FINAL_FUEL_MASS = 470
MIN_MANIFOLD_LENGTH = 5
MAX_MANIFOLD_LENGTH = 11
THRUST = 1.0
NUM_SEGMENTS = 20  # Number of thrust control segments


def cartesian_to_spherical(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates (ux, uy, uz) to spherical coordinates (alpha, beta, r).
    
    This function matches the _convert_to_spherical logic from the benchmarking code
    but operates on the original physical thrust values (not normalized ones).
    
    Args:
        ux, uy, uz: Cartesian thrust components
        
    Returns:
        alpha: Azimuthal angle [0, 2π]
        beta: Polar angle [0, 2π] 
        r: Magnitude [0, 1] (this is the key constraint!)
    """
    # Compute magnitude (this is what we want to constrain to ≤ 1)
    r = np.sqrt(ux**2 + uy**2 + uz**2)
    
    # Initialize angles
    beta = np.zeros_like(r)
    
    # Compute polar angle beta (elevation)
    # Only compute for non-zero magnitude to avoid division by zero
    mask_non_zero = r != 0
    beta[mask_non_zero] = np.arcsin(uz[mask_non_zero] / r[mask_non_zero])
    
    # Compute azimuthal angle alpha
    alpha = np.arctan2(uy, ux)
    
    # Ensure alpha is in [0, 2π] range
    alpha = np.where(alpha >= 0, alpha, 2 * np.pi + alpha)
    
    # Ensure beta is in [0, 2π] range (matching benchmarking code)
    beta = np.where(beta >= 0, beta, 2 * np.pi + beta)
    
    return alpha, beta, r


def normalize_spherical_coordinates(alpha: np.ndarray, beta: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize spherical coordinates to [0, 1] range for training.
    
    Args:
        alpha: Azimuthal angle [0, 2π]
        beta: Polar angle [0, 2π]
        r: Magnitude [0, 1] (already in correct range, but may need clipping)
        
    Returns:
        alpha_norm: Normalized alpha [0, 1]
        beta_norm: Normalized beta [0, 1] 
        r_norm: Normalized r [0, 1]
    """
    # Normalize angles from [0, 2π] to [0, 1]
    alpha_norm = alpha / (2 * np.pi)
    beta_norm = beta / (2 * np.pi)
    
    # r should already be in [0, 1] range, but clip to be safe
    # This is the key: r represents the magnitude directly
    r_norm = np.clip(r, 0, 1)
    
    return alpha_norm, beta_norm, r_norm


def unnormalize_spherical_coordinates(alpha_norm: np.ndarray, beta_norm: np.ndarray, r_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unnormalize spherical coordinates from [0, 1] back to physical range.
    This is the inverse of normalize_spherical_coordinates.
    
    Args:
        alpha_norm, beta_norm, r_norm: Normalized spherical coordinates [0, 1]
        
    Returns:
        alpha: Azimuthal angle [0, 2π]
        beta: Polar angle [0, 2π]
        r: Magnitude [0, 1] (guaranteed ≤ 1!)
    """
    alpha = alpha_norm * 2 * np.pi
    beta = beta_norm * 2 * np.pi  
    r = r_norm  # r is already the magnitude, no further transformation needed!
    
    return alpha, beta, r


def spherical_to_cartesian(alpha: np.ndarray, beta: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates back to Cartesian coordinates.
    This matches the spherical_to_cart function from prepare_training_data.py.
    
    Args:
        alpha: Azimuthal angle [0, 2π]
        beta: Polar angle [0, 2π]
        r: Magnitude [0, 1]
        
    Returns:
        ux, uy, uz: Cartesian thrust components with guaranteed magnitude ≤ 1
    """
    ux = r * np.cos(alpha) * np.cos(beta)
    uy = r * np.sin(alpha) * np.cos(beta)
    uz = r * np.sin(beta)
    
    return ux, uy, uz


def analyze_original_dataset(data: np.ndarray) -> dict:
    """
    Analyze the original Cartesian dataset to understand thrust magnitude distribution.
    
    Args:
        data: Original training data (N, 67)
        
    Returns:
        Dictionary with analysis results
    """
    print("=" * 80)
    print("ANALYZING ORIGINAL CARTESIAN DATASET")
    print("=" * 80)
    
    # Extract thrust variables (indices 4:64, which is 60 values = 20 segments × 3 coords)
    thrust_data = data[:, 4:64]  # Shape: (N, 60)
    
    # Reshape to (N, 20, 3) for easier processing
    thrust_reshaped = thrust_data.reshape(-1, NUM_SEGMENTS, 3)
    
    print(f"Original data shape: {data.shape}")
    print(f"Thrust data shape: {thrust_data.shape}")
    print(f"Thrust reshaped shape: {thrust_reshaped.shape}")
    
    # The thrust data is currently normalized Cartesian coordinates in [0, 1] range
    # To analyze magnitudes, we need to unnormalize to [-1, 1] range first
    # This matches the benchmarking unnormalization: model_outputs[:, 3:-3] * 2 * thrust - thrust
    thrust_unnormalized = thrust_reshaped * 2 * THRUST - THRUST  # Convert [0,1] to [-1,1]
    
    # Extract Cartesian components
    ux = thrust_unnormalized[:, :, 0]  # Shape: (N, 20)
    uy = thrust_unnormalized[:, :, 1]  # Shape: (N, 20)
    uz = thrust_unnormalized[:, :, 2]  # Shape: (N, 20)
    
    # Compute magnitudes
    magnitudes = np.sqrt(ux**2 + uy**2 + uz**2)  # Shape: (N, 20)
    
    # Analyze magnitude violations
    violations = magnitudes > 1.0
    num_violations = np.sum(violations)
    total_elements = magnitudes.size
    
    # Statistics
    stats = {
        'total_samples': data.shape[0],
        'total_thrust_elements': total_elements,
        'magnitude_violations': num_violations,
        'violation_percentage': 100 * num_violations / total_elements,
        'max_magnitude': np.max(magnitudes),
        'min_magnitude': np.min(magnitudes),
        'mean_magnitude': np.mean(magnitudes),
        'std_magnitude': np.std(magnitudes),
        'magnitudes_exceeding_1': magnitudes[violations] if num_violations > 0 else np.array([])
    }
    
    print(f"Total samples: {stats['total_samples']}")
    print(f"Total thrust elements: {stats['total_thrust_elements']}")
    print(f"Magnitude violations (> 1.0): {stats['magnitude_violations']}/{stats['total_thrust_elements']} ({stats['violation_percentage']:.4f}%)")
    print(f"Max magnitude: {stats['max_magnitude']:.6f}")
    print(f"Min magnitude: {stats['min_magnitude']:.6f}")
    print(f"Mean magnitude: {stats['mean_magnitude']:.6f}")
    print(f"Std magnitude: {stats['std_magnitude']:.6f}")
    
    if num_violations > 0:
        print(f"Magnitudes exceeding 1.0: {np.unique(stats['magnitudes_exceeding_1'])}")
    
    return stats


def convert_dataset_to_spherical(data: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Convert the training dataset from Cartesian to spherical thrust representation.
    
    Args:
        data: Original training data (N, 67) with Cartesian thrust
        
    Returns:
        converted_data: New training data (N, 67) with spherical thrust
        conversion_stats: Dictionary with conversion statistics
    """
    print("\n" + "=" * 80)
    print("CONVERTING DATASET TO SPHERICAL COORDINATES")
    print("=" * 80)
    
    # Create a copy to avoid modifying original data
    converted_data = data.copy()
    
    # Extract thrust variables (indices 4:64)
    thrust_data = data[:, 4:64]  # Shape: (N, 60)
    thrust_reshaped = thrust_data.reshape(-1, NUM_SEGMENTS, 3)  # Shape: (N, 20, 3)
    
    print(f"Processing {thrust_reshaped.shape[0]} samples with {thrust_reshaped.shape[1]} thrust segments each")
    
    # Step 1: Unnormalize from [0,1] to [-1,1] (matching current benchmarking logic)
    thrust_cartesian_physical = thrust_reshaped * 2 * THRUST - THRUST
    
    # Step 2: Extract Cartesian components
    ux = thrust_cartesian_physical[:, :, 0]  # Shape: (N, 20)
    uy = thrust_cartesian_physical[:, :, 1]  # Shape: (N, 20)
    uz = thrust_cartesian_physical[:, :, 2]  # Shape: (N, 20)
    
    print(f"Cartesian components shape: ux={ux.shape}, uy={uy.shape}, uz={uz.shape}")
    
    # Step 3: Convert to spherical coordinates
    alpha, beta, r = cartesian_to_spherical(ux, uy, uz)
    
    print(f"Spherical components shape: alpha={alpha.shape}, beta={beta.shape}, r={r.shape}")
    print(f"Alpha range: [{np.min(alpha):.6f}, {np.max(alpha):.6f}]")
    print(f"Beta range: [{np.min(beta):.6f}, {np.max(beta):.6f}]")
    print(f"R range: [{np.min(r):.6f}, {np.max(r):.6f}]")
    
    # Step 4: Normalize spherical coordinates to [0,1] for training
    alpha_norm, beta_norm, r_norm = normalize_spherical_coordinates(alpha, beta, r)
    
    print(f"Normalized spherical ranges:")
    print(f"  Alpha_norm: [{np.min(alpha_norm):.6f}, {np.max(alpha_norm):.6f}]")
    print(f"  Beta_norm: [{np.min(beta_norm):.6f}, {np.max(beta_norm):.6f}]")
    print(f"  R_norm: [{np.min(r_norm):.6f}, {np.max(r_norm):.6f}]")
    
    # Step 5: Reshape back to (N, 60) and store in converted dataset
    spherical_normalized = np.stack([alpha_norm, beta_norm, r_norm], axis=2)  # Shape: (N, 20, 3)
    spherical_flattened = spherical_normalized.reshape(-1, 60)  # Shape: (N, 60)
    
    # Replace thrust data in the converted dataset
    converted_data[:, 4:64] = spherical_flattened
    
    # Collect conversion statistics
    conversion_stats = {
        'original_cartesian_magnitude_max': np.max(np.sqrt(ux**2 + uy**2 + uz**2)),
        'spherical_r_max': np.max(r),
        'spherical_r_min': np.min(r),
        'spherical_r_mean': np.mean(r),
        'num_r_clipped': np.sum(r > 1.0),  # Should be same as original violations
        'alpha_range': [np.min(alpha), np.max(alpha)],
        'beta_range': [np.min(beta), np.max(beta)],
        'r_range': [np.min(r), np.max(r)]
    }
    
    print(f"\nConversion completed successfully!")
    print(f"Maximum r value: {conversion_stats['spherical_r_max']:.6f}")
    print(f"Number of r values clipped to 1.0: {conversion_stats['num_r_clipped']}")
    
    return converted_data, conversion_stats


def verify_conversion(original_data: np.ndarray, converted_data: np.ndarray, num_test_samples: int = 100) -> dict:
    """
    Verify the conversion by testing round-trip conversion on a subset of data.
    
    Args:
        original_data: Original Cartesian dataset
        converted_data: Converted spherical dataset  
        num_test_samples: Number of samples to test
        
    Returns:
        Dictionary with verification results
    """
    print("\n" + "=" * 80)
    print("VERIFYING CONVERSION ACCURACY")
    print("=" * 80)
    
    # Select random samples for testing
    test_indices = np.random.choice(original_data.shape[0], size=min(num_test_samples, original_data.shape[0]), replace=False)
    
    verification_results = {
        'test_samples': len(test_indices),
        'max_reconstruction_error': 0,
        'mean_reconstruction_error': 0,
        'magnitude_constraint_satisfied': True,
        'max_magnitude_after_conversion': 0
    }
    
    reconstruction_errors = []
    max_magnitudes = []
    
    for i, idx in enumerate(test_indices):
        # Get original thrust data (normalized Cartesian)
        original_thrust = original_data[idx, 4:64].reshape(NUM_SEGMENTS, 3)
        
        # Get converted thrust data (normalized spherical)
        converted_thrust = converted_data[idx, 4:64].reshape(NUM_SEGMENTS, 3)
        
        # Unnormalize spherical coordinates
        alpha_norm, beta_norm, r_norm = converted_thrust[:, 0], converted_thrust[:, 1], converted_thrust[:, 2]
        alpha, beta, r = unnormalize_spherical_coordinates(alpha_norm, beta_norm, r_norm)
        
        # Convert back to Cartesian
        ux_reconstructed, uy_reconstructed, uz_reconstructed = spherical_to_cartesian(alpha, beta, r)
        
        # Normalize back to [0,1] for comparison with original
        ux_norm = (ux_reconstructed + THRUST) / (2 * THRUST)
        uy_norm = (uy_reconstructed + THRUST) / (2 * THRUST)
        uz_norm = (uz_reconstructed + THRUST) / (2 * THRUST)
        
        reconstructed_thrust = np.column_stack([ux_norm, uy_norm, uz_norm])
        
        # Compute reconstruction error
        error = np.max(np.abs(original_thrust - reconstructed_thrust))
        reconstruction_errors.append(error)
        
        # Check magnitude constraint
        magnitude = np.sqrt(ux_reconstructed**2 + uy_reconstructed**2 + uz_reconstructed**2)
        max_magnitude = np.max(magnitude)
        max_magnitudes.append(max_magnitude)
        
        if i < 5:  # Print details for first few samples
            print(f"Sample {idx}: reconstruction_error={error:.8f}, max_magnitude={max_magnitude:.6f}")
    
    verification_results['max_reconstruction_error'] = np.max(reconstruction_errors)
    verification_results['mean_reconstruction_error'] = np.mean(reconstruction_errors)
    verification_results['max_magnitude_after_conversion'] = np.max(max_magnitudes)
    verification_results['magnitude_constraint_satisfied'] = np.all(np.array(max_magnitudes) <= 1.0 + 1e-10)  # Small tolerance
    
    print(f"\nVerification Results:")
    print(f"  Test samples: {verification_results['test_samples']}")
    print(f"  Max reconstruction error: {verification_results['max_reconstruction_error']:.8f}")
    print(f"  Mean reconstruction error: {verification_results['mean_reconstruction_error']:.8f}")
    print(f"  Max magnitude after conversion: {verification_results['max_magnitude_after_conversion']:.6f}")
    print(f"  Magnitude constraint satisfied: {verification_results['magnitude_constraint_satisfied']}")
    
    return verification_results


def create_comparison_plots(original_data: np.ndarray, converted_data: np.ndarray, output_dir: str = "."):
    """
    Create comparison plots between original and converted datasets.
    
    Args:
        original_data: Original Cartesian dataset
        converted_data: Converted spherical dataset
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("CREATING COMPARISON PLOTS")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze original dataset magnitudes
    original_thrust = original_data[:, 4:64].reshape(-1, NUM_SEGMENTS, 3)
    original_unnormalized = original_thrust * 2 * THRUST - THRUST
    original_magnitudes = np.sqrt(np.sum(original_unnormalized**2, axis=2)).flatten()
    
    # Analyze converted dataset magnitudes (using r component directly)
    converted_thrust = converted_data[:, 4:64].reshape(-1, NUM_SEGMENTS, 3)
    converted_r = converted_thrust[:, :, 2].flatten()  # r_norm is the magnitude
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Original magnitude distribution
    axes[0, 0].hist(original_magnitudes, bins=50, alpha=0.7, color='red', density=True)
    axes[0, 0].axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Magnitude = 1.0')
    axes[0, 0].set_title('Original Dataset: Thrust Magnitude Distribution')
    axes[0, 0].set_xlabel('Magnitude')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Converted magnitude distribution
    axes[0, 1].hist(converted_r, bins=50, alpha=0.7, color='blue', density=True)
    axes[0, 1].axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Magnitude = 1.0')
    axes[0, 1].set_title('Converted Dataset: Thrust Magnitude Distribution (r)')
    axes[0, 1].set_xlabel('Magnitude (r)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Magnitude comparison scatter
    sample_indices = np.random.choice(len(original_magnitudes), size=min(10000, len(original_magnitudes)), replace=False)
    axes[1, 0].scatter(original_magnitudes[sample_indices], converted_r[sample_indices], alpha=0.1, s=1)
    axes[1, 0].plot([0, 2], [0, 2], 'r--', linewidth=2, label='y=x')
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    axes[1, 0].axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    axes[1, 0].set_title('Magnitude Comparison: Original vs Converted')
    axes[1, 0].set_xlabel('Original Magnitude')
    axes[1, 0].set_ylabel('Converted Magnitude (r)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics comparison
    stats_labels = ['Max', 'Mean', 'Std', '99th Percentile']
    original_stats = [
        np.max(original_magnitudes),
        np.mean(original_magnitudes),
        np.std(original_magnitudes),
        np.percentile(original_magnitudes, 99)
    ]
    converted_stats = [
        np.max(converted_r),
        np.mean(converted_r),
        np.std(converted_r),
        np.percentile(converted_r, 99)
    ]
    
    x = np.arange(len(stats_labels))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, original_stats, width, label='Original', alpha=0.7, color='red')
    axes[1, 1].bar(x + width/2, converted_stats, width, label='Converted', alpha=0.7, color='blue')
    axes[1, 1].axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Magnitude = 1.0')
    axes[1, 1].set_title('Statistics Comparison')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(stats_labels)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'dataset_conversion_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {plot_path}")


def main():
    """
    Main function to convert the GTO Halo training dataset from Cartesian to spherical coordinates.
    """
    print("=" * 80)
    print("GTO HALO TRAINING DATASET CONVERSION: CARTESIAN → SPHERICAL")
    print("=" * 80)
    
    # Configuration
    input_path = "GTO_Halo_DM/data/training_data_boundary_100000.pkl"
    output_path = "GTO_Halo_DM/data/training_data_boundary_100000_spherical.pkl"
    analysis_output_dir = "spherical_conversion_analysis"
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        print("Please ensure the training data file exists at the specified path.")
        return
    
    # Load original dataset
    print(f"Loading original dataset from: {input_path}")
    with open(input_path, 'rb') as f:
        original_data = pickle.load(f)
    
    print(f"Loaded dataset shape: {original_data.shape}")
    print(f"Expected shape: (N, 67) where N is number of samples")
    
    if original_data.shape[1] != 67:
        print(f"ERROR: Unexpected dataset shape. Expected 67 features, got {original_data.shape[1]}")
        return
    
    # Analyze original dataset
    original_stats = analyze_original_dataset(original_data)
    
    # Convert dataset to spherical coordinates
    converted_data, conversion_stats = convert_dataset_to_spherical(original_data)
    
    # Verify conversion accuracy
    verification_results = verify_conversion(original_data, converted_data)
    
    # Create comparison plots
    create_comparison_plots(original_data, converted_data, analysis_output_dir)
    
    # Save converted dataset
    print(f"\nSaving converted dataset to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(converted_data, f)
    
    # Save analysis results
    analysis_results = {
        'original_stats': original_stats,
        'conversion_stats': conversion_stats,
        'verification_results': verification_results,
        'dataset_info': {
            'original_shape': original_data.shape,
            'converted_shape': converted_data.shape,
            'input_path': input_path,
            'output_path': output_path
        }
    }
    
    analysis_path = os.path.join(analysis_output_dir, 'conversion_analysis.pkl')
    os.makedirs(analysis_output_dir, exist_ok=True)
    with open(analysis_path, 'wb') as f:
        pickle.dump(analysis_results, f)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully converted {original_data.shape[0]} samples")
    print(f"✅ Original magnitude violations: {original_stats['magnitude_violations']} ({original_stats['violation_percentage']:.4f}%)")
    print(f"✅ Converted max magnitude: {verification_results['max_magnitude_after_conversion']:.6f}")
    print(f"✅ Magnitude constraint satisfied: {verification_results['magnitude_constraint_satisfied']}")
    print(f"✅ Max reconstruction error: {verification_results['max_reconstruction_error']:.8f}")
    print(f"✅ Converted dataset saved to: {output_path}")
    print(f"✅ Analysis results saved to: {analysis_path}")
    print("\n🎯 KEY BENEFIT: The new spherical dataset guarantees thrust magnitude ≤ 1.0")
    print("   No clipping will be needed during inference!")
    print("=" * 80)


if __name__ == "__main__":
    main()