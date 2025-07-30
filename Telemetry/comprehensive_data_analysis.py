#!/usr/bin/env python3
"""
Comprehensive Data Analysis for GTO Halo Training Data
======================================================

This script conducts an in-depth analysis of the training data using the same
dataset class as the training model to provide essential information for designing
an optimal NCSN++ model with maximum performance.

Analysis includes:
1. Basic statistics and data shape analysis
2. Distribution analysis of raw and normalized values
3. Spatial pattern analysis in 9x9 images
4. Classifier value analysis (first value as label)
5. Boundary violation analysis
6. Correlation analysis between features
7. Outlier detection
8. Recommendations for model architecture
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add the Reflected-Diffusion directory to path to import datasets
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Reflected-Diffusion'))
from datasets import GTOHaloImageDataset

class ComprehensiveDataAnalyzer:
    def __init__(self, data_path):
        """
        Initialize the analyzer with the training data path.
        
        Args:
            data_path (str): Path to the training data pickle file
        """
        self.data_path = data_path
        self.dataset = GTOHaloImageDataset(data_path)
        self.raw_data = None
        self.normalized_data = None
        self.classifier_values = None
        self.analysis_results = {}
        
        print("🔍 Loading and preprocessing data...")
        self._load_data()
        
    def _load_data(self):
        """Load raw data and extract key components."""
        # Load raw data from pickle
        with open(self.data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.raw_data = np.array(raw_data, dtype=np.float32)
        print(f"📊 Raw data shape: {self.raw_data.shape}")
        
        # Extract classifier values (first column)
        self.classifier_values = self.raw_data[:, 0]
        
        # Extract thrust values (indices 4-63)
        self.thrust_values = self.raw_data[:, 4:64]
        
        # Extract other features
        self.other_features = np.concatenate([
            self.raw_data[:, 1:4],  # indices 1-3
            self.raw_data[:, 64:]   # indices 64+
        ], axis=1)
        
        # Load normalized data using the dataset class
        normalized_images = []
        normalized_classifiers = []
        
        for i in range(len(self.dataset)):
            img, classifier = self.dataset[i]
            normalized_images.append(img.numpy())
            normalized_classifiers.append(classifier.numpy())
        
        self.normalized_data = np.array(normalized_images)
        self.normalized_classifiers = np.array(normalized_classifiers)
        
        print(f"📊 Normalized data shape: {self.normalized_data.shape}")
        print(f"📊 Classifier values shape: {self.normalized_classifiers.shape}")
        
    def calculate_true_mean_std(self):
        """Calculate the true mean and std of the dataset."""
        print("\n" + "="*60)
        print("🔍 TRUE MEAN AND STD CALCULATION")
        print("="*60)
        
        # Calculate mean and std of the entire dataset
        true_mean = np.mean(self.raw_data)
        true_std = np.std(self.raw_data)
        
        print(f"📊 True mean of entire dataset: {true_mean:.6f}")
        print(f"📊 True std of entire dataset: {true_std:.6f}")
        
        # Compare with current values used in dataset
        current_mean = 0.4652
        current_std = 0.1811
        
        print(f"\n📊 Current values used in dataset:")
        print(f"   Mean: {current_mean:.6f}")
        print(f"   Std:  {current_std:.6f}")
        
        print(f"\n📊 Difference:")
        print(f"   Mean difference: {abs(true_mean - current_mean):.6f}")
        print(f"   Std difference:  {abs(true_std - current_std):.6f}")
        
        # Calculate mean and std for different components
        print(f"\n📊 Component-wise statistics:")
        print(f"   Classifier values (index 0): mean={np.mean(self.classifier_values):.6f}, std={np.std(self.classifier_values):.6f}")
        print(f"   Thrust values (indices 4-63): mean={np.mean(self.thrust_values):.6f}, std={np.std(self.thrust_values):.6f}")
        print(f"   Other features: mean={np.mean(self.other_features):.6f}, std={np.std(self.other_features):.6f}")
        
        # Store results
        self.analysis_results['true_mean'] = true_mean
        self.analysis_results['true_std'] = true_std
        self.analysis_results['current_mean'] = current_mean
        self.analysis_results['current_std'] = current_std
        
        # Now verify using the GTOHaloImageDataset class
        print(f"\n🔍 VERIFICATION USING GTOHaloImageDataset CLASS:")
        try:
            import sys
            sys.path.append('../Reflected-Diffusion')
            from datasets import GTOHaloImageDataset
            
            # Load dataset using the class
            dataset = GTOHaloImageDataset('../GTO_Halo_DM/data/training_data_boundary_100000_spherical.pkl')
            
            # Get all raw data from the dataset
            all_raw_data = []
            all_padded_data = []
            for i in range(len(dataset)):
                raw_item = dataset.data[i]  # Get raw data before normalization
                all_raw_data.append(raw_item)
                
                # Also get the padded data (what the model actually sees)
                vec = raw_item
                padded = np.pad(vec, (0, 81 - len(vec)), 'constant')
                all_padded_data.append(padded)
            
            all_raw_data = np.array(all_raw_data)
            all_padded_data = np.array(all_padded_data)
            
            # Calculate statistics from the raw data
            raw_mean = np.mean(all_raw_data)
            raw_std = np.std(all_raw_data)
            
            # Calculate statistics from the padded data (what model actually sees)
            padded_mean = np.mean(all_padded_data)
            padded_std = np.std(all_padded_data)
            
            print(f"📊 Raw data (before padding):")
            print(f"   Mean: {raw_mean:.6f}")
            print(f"   Std:  {raw_std:.6f}")
            
            print(f"📊 Padded data (what model sees):")
            print(f"   Mean: {padded_mean:.6f}")
            print(f"   Std:  {padded_std:.6f}")
            
            # Check if they match our calculated values
            raw_mean_match = abs(true_mean - raw_mean) < 1e-6
            raw_std_match = abs(true_std - raw_std) < 1e-6
            
            print(f"\n✅ Verification Results:")
            print(f"   Raw data mean matches: {raw_mean_match}")
            print(f"   Raw data std matches: {raw_std_match}")
            
            if raw_mean_match and raw_std_match:
                print(f"   ✅ Raw data values are consistent!")
            else:
                print(f"   ❌ Raw data values don't match!")
                
            print(f"\n📋 Current values in GTOHaloImageDataset:")
            print(f"   Mean: {dataset.mean:.6f}")
            print(f"   Std:  {dataset.std:.6f}")
            
            # Calculate what the correct values should be for the padded data
            correct_padded_mean = padded_mean
            correct_padded_std = padded_std
            
            print(f"\n🔧 RECOMMENDED VALUES FOR GTOHaloImageDataset:")
            print(f"   Mean: {correct_padded_mean:.6f}")
            print(f"   Std:  {correct_padded_std:.6f}")
                
        except Exception as e:
            print(f"❌ Error loading GTOHaloImage class: {e}")
            print(f"   This might be due to import path issues")
        
        return true_mean, true_std

    def basic_statistics(self):
        """Compute and display basic statistics."""
        print("\n" + "="*60)
        print("📈 BASIC STATISTICS")
        print("="*60)
        
        # Raw data statistics
        raw_stats = {
            'mean': np.mean(self.raw_data),
            'std': np.std(self.raw_data),
            'min': np.min(self.raw_data),
            'max': np.max(self.raw_data),
            'median': np.median(self.raw_data),
            'skewness': stats.skew(self.raw_data.flatten()),
            'kurtosis': stats.kurtosis(self.raw_data.flatten())
        }
        
        # Normalized data statistics
        norm_stats = {
            'mean': np.mean(self.normalized_data),
            'std': np.std(self.normalized_data),
            'min': np.min(self.normalized_data),
            'max': np.max(self.normalized_data),
            'median': np.median(self.normalized_data),
            'skewness': stats.skew(self.normalized_data.flatten()),
            'kurtosis': stats.kurtosis(self.normalized_data.flatten())
        }
        
        # Classifier statistics
        classifier_stats = {
            'mean': np.mean(self.classifier_values),
            'std': np.std(self.classifier_values),
            'min': np.min(self.classifier_values),
            'max': np.max(self.classifier_values),
            'unique_values': len(np.unique(self.classifier_values)),
            'most_common': float(stats.mode(self.classifier_values)[0])
        }
        
        # Thrust statistics
        thrust_stats = {
            'mean': np.mean(self.thrust_values),
            'std': np.std(self.thrust_values),
            'min': np.min(self.thrust_values),
            'max': np.max(self.thrust_values),
            'boundary_violations': np.sum((self.thrust_values < 0) | (self.thrust_values > 1))
        }
        
        print("Raw Data Statistics:")
        for key, value in raw_stats.items():
            print(f"  {key}: {value:.6f}")
            
        print("\nNormalized Data Statistics:")
        for key, value in norm_stats.items():
            print(f"  {key}: {value:.6f}")
            
        print("\nClassifier Statistics:")
        for key, value in classifier_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
                
        print("\nThrust Statistics:")
        for key, value in thrust_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        self.analysis_results['basic_stats'] = {
            'raw': raw_stats,
            'normalized': norm_stats,
            'classifier': classifier_stats,
            'thrust': thrust_stats
        }
        
    def distribution_analysis(self):
        """Analyze distributions of various data components."""
        print("\n" + "="*60)
        print("📊 DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Create comprehensive distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Raw data distribution
        axes[0, 0].hist(self.raw_data.flatten(), bins=100, alpha=0.7, density=True)
        axes[0, 0].set_title('Raw Data Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        
        # Normalized data distribution
        axes[0, 1].hist(self.normalized_data.flatten(), bins=100, alpha=0.7, density=True)
        axes[0, 1].set_title('Normalized Data Distribution')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Density')
        
        # Classifier values distribution
        axes[0, 2].hist(self.classifier_values, bins=50, alpha=0.7, density=True)
        axes[0, 2].set_title('Classifier Values Distribution')
        axes[0, 2].set_xlabel('Classifier Value')
        axes[0, 2].set_ylabel('Density')
        
        # Thrust values distribution
        axes[1, 0].hist(self.thrust_values.flatten(), bins=100, alpha=0.7, density=True)
        axes[1, 0].set_title('Thrust Values Distribution')
        axes[1, 0].set_xlabel('Thrust Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Boundary')
        axes[1, 0].axvline(1, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].legend()
        
        # Other features distribution
        axes[1, 1].hist(self.other_features.flatten(), bins=100, alpha=0.7, density=True)
        axes[1, 1].set_title('Other Features Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Density')
        
        # Normalized classifier distribution
        axes[1, 2].hist(self.normalized_classifiers.flatten(), bins=50, alpha=0.7, density=True)
        axes[1, 2].set_title('Normalized Classifier Distribution')
        axes[1, 2].set_xlabel('Normalized Classifier Value')
        axes[1, 2].set_ylabel('Density')
        
        plt.tight_layout()
        plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def spatial_pattern_analysis(self):
        """Analyze spatial patterns in the 9x9 images."""
        print("\n" + "="*60)
        print("🖼️  SPATIAL PATTERN ANALYSIS")
        print("="*60)
        
        # Calculate mean and std for each pixel position
        mean_image = np.mean(self.normalized_data, axis=0)[0]  # Remove channel dimension
        std_image = np.std(self.normalized_data, axis=0)[0]
        
        # Create spatial analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Mean image
        im1 = axes[0, 0].imshow(mean_image, cmap='viridis', aspect='equal')
        axes[0, 0].set_title('Mean Image (9×9)')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Std image
        im2 = axes[0, 1].imshow(std_image, cmap='viridis', aspect='equal')
        axes[0, 1].set_title('Std Image (9×9)')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Variance image
        variance_image = std_image ** 2
        im3 = axes[0, 2].imshow(variance_image, cmap='viridis', aspect='equal')
        axes[0, 2].set_title('Variance Image (9×9)')
        axes[0, 2].set_xlabel('X')
        axes[0, 2].set_ylabel('Y')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Pixel-wise statistics
        pixel_means = mean_image.flatten()
        pixel_stds = std_image.flatten()
        
        axes[1, 0].scatter(range(81), pixel_means, alpha=0.7)
        axes[1, 0].set_title('Pixel-wise Means')
        axes[1, 0].set_xlabel('Pixel Index')
        axes[1, 0].set_ylabel('Mean Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(range(81), pixel_stds, alpha=0.7)
        axes[1, 1].set_title('Pixel-wise Standard Deviations')
        axes[1, 1].set_xlabel('Pixel Index')
        axes[1, 1].set_ylabel('Std Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Correlation between pixel positions
        pixel_corr = np.corrcoef(self.normalized_data.reshape(len(self.normalized_data), -1).T)
        im4 = axes[1, 2].imshow(pixel_corr, cmap='coolwarm', aspect='equal')
        axes[1, 2].set_title('Pixel-wise Correlation Matrix')
        axes[1, 2].set_xlabel('Pixel Index')
        axes[1, 2].set_ylabel('Pixel Index')
        plt.colorbar(im4, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('spatial_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print spatial statistics
        print(f"Spatial Statistics:")
        print(f"  Mean image range: [{mean_image.min():.4f}, {mean_image.max():.4f}]")
        print(f"  Std image range: [{std_image.min():.4f}, {std_image.max():.4f}]")
        print(f"  High variance pixels (>0.5): {np.sum(variance_image > 0.5)}")
        print(f"  Low variance pixels (<0.1): {np.sum(variance_image < 0.1)}")
        
        self.analysis_results['spatial_stats'] = {
            'mean_image': mean_image,
            'std_image': std_image,
            'variance_image': variance_image,
            'high_variance_pixels': np.sum(variance_image > 0.5),
            'low_variance_pixels': np.sum(variance_image < 0.1)
        }
        
    def outlier_analysis(self):
        """Detect and analyze outliers in the data."""
        print("\n" + "="*60)
        print("🔍 OUTLIER ANALYSIS")
        print("="*60)
        
        # Define outlier thresholds
        raw_data_flat = self.raw_data.flatten()
        norm_data_flat = self.normalized_data.flatten()
        
        # Z-score based outliers
        raw_z_scores = np.abs(stats.zscore(raw_data_flat))
        norm_z_scores = np.abs(stats.zscore(norm_data_flat))
        
        raw_outliers_z = np.sum(raw_z_scores > 3)
        norm_outliers_z = np.sum(norm_z_scores > 3)
        
        # IQR based outliers
        raw_q1, raw_q3 = np.percentile(raw_data_flat, [25, 75])
        raw_iqr = raw_q3 - raw_q1
        raw_lower_bound = raw_q1 - 1.5 * raw_iqr
        raw_upper_bound = raw_q3 + 1.5 * raw_iqr
        raw_outliers_iqr = np.sum((raw_data_flat < raw_lower_bound) | (raw_data_flat > raw_upper_bound))
        
        norm_q1, norm_q3 = np.percentile(norm_data_flat, [25, 75])
        norm_iqr = norm_q3 - norm_q1
        norm_lower_bound = norm_q1 - 1.5 * norm_iqr
        norm_upper_bound = norm_q3 + 1.5 * norm_iqr
        norm_outliers_iqr = np.sum((norm_data_flat < norm_lower_bound) | (norm_data_flat > norm_upper_bound))
        
        # Boundary violations
        thrust_violations = np.sum((self.thrust_values < 0) | (self.thrust_values > 1))
        
        print(f"Outlier Analysis Results:")
        print(f"  Raw data outliers (Z-score > 3): {raw_outliers_z} ({100*raw_outliers_z/len(raw_data_flat):.2f}%)")
        print(f"  Normalized data outliers (Z-score > 3): {norm_outliers_z} ({100*norm_outliers_z/len(norm_data_flat):.2f}%)")
        print(f"  Raw data outliers (IQR method): {raw_outliers_iqr} ({100*raw_outliers_iqr/len(raw_data_flat):.2f}%)")
        print(f"  Normalized data outliers (IQR method): {norm_outliers_iqr} ({100*norm_outliers_iqr/len(norm_data_flat):.2f}%)")
        print(f"  Thrust boundary violations: {thrust_violations} ({100*thrust_violations/self.thrust_values.size:.2f}%)")
        
        # Create outlier visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Raw data outliers
        axes[0, 0].hist(raw_data_flat, bins=100, alpha=0.7, density=True)
        axes[0, 0].axvline(raw_lower_bound, color='red', linestyle='--', label='IQR Bounds')
        axes[0, 0].axvline(raw_upper_bound, color='red', linestyle='--')
        axes[0, 0].set_title('Raw Data with Outlier Bounds')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        
        # Normalized data outliers
        axes[0, 1].hist(norm_data_flat, bins=100, alpha=0.7, density=True)
        axes[0, 1].axvline(norm_lower_bound, color='red', linestyle='--', label='IQR Bounds')
        axes[0, 1].axvline(norm_upper_bound, color='red', linestyle='--')
        axes[0, 1].set_title('Normalized Data with Outlier Bounds')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        
        # Thrust violations
        axes[1, 0].hist(self.thrust_values.flatten(), bins=100, alpha=0.7, density=True)
        axes[1, 0].axvline(0, color='red', linestyle='--', label='Boundary')
        axes[1, 0].axvline(1, color='red', linestyle='--')
        axes[1, 0].set_title('Thrust Values with Boundaries')
        axes[1, 0].set_xlabel('Thrust Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        
        # Z-score distribution
        axes[1, 1].hist(norm_z_scores, bins=100, alpha=0.7, density=True)
        axes[1, 1].axvline(3, color='red', linestyle='--', label='Z=3 Threshold')
        axes[1, 1].set_title('Normalized Data Z-Scores')
        axes[1, 1].set_xlabel('Z-Score')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.analysis_results['outlier_stats'] = {
            'raw_outliers_z': raw_outliers_z,
            'norm_outliers_z': norm_outliers_z,
            'raw_outliers_iqr': raw_outliers_iqr,
            'norm_outliers_iqr': norm_outliers_iqr,
            'thrust_violations': thrust_violations
        }
        
    def correlation_analysis(self):
        """Analyze correlations between different features."""
        print("\n" + "="*60)
        print("🔗 CORRELATION ANALYSIS")
        print("="*60)
        
        # Create feature matrix
        feature_matrix = np.column_stack([
            self.classifier_values,
            self.raw_data[:, 1:4],  # indices 1-3
            self.thrust_values.mean(axis=1),  # mean thrust
            self.thrust_values.std(axis=1),   # thrust std
            self.other_features.mean(axis=1), # mean other features
            self.other_features.std(axis=1)   # other features std
        ])
        
        feature_names = ['Classifier', 'Feature1', 'Feature2', 'Feature3', 
                        'Mean_Thrust', 'Std_Thrust', 'Mean_Other', 'Std_Other']
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(feature_matrix.T)
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=feature_names, yticklabels=feature_names)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print("Strongest correlations:")
        for i, (feat1, feat2, corr) in enumerate(corr_pairs[:10]):
            print(f"  {feat1} ↔ {feat2}: {corr:.4f}")
        
        self.analysis_results['correlation_stats'] = {
            'correlation_matrix': corr_matrix,
            'strongest_correlations': corr_pairs[:10]
        }
        
    def sample_visualization(self):
        """Visualize sample images and their characteristics."""
        print("\n" + "="*60)
        print("🖼️  SAMPLE VISUALIZATION")
        print("="*60)
        
        # Select diverse samples
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.normalized_data), 16, replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        
        for i, idx in enumerate(sample_indices):
            row, col = i // 4, i % 4
            
            # Get the image
            img = self.normalized_data[idx, 0]  # Remove channel dimension
            
            # Create the plot
            im = axes[row, col].imshow(img, cmap='gray', aspect='equal')
            axes[row, col].set_title(f'Sample {idx}\nClassifier: {self.classifier_values[idx]:.4f}')
            axes[row, col].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        plt.suptitle('Sample Training Images (9×9 Normalized)', fontsize=16)
        plt.tight_layout()
        plt.savefig('sample_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Show extreme samples
        extreme_indices = []
        for i in range(len(self.normalized_data)):
            img = self.normalized_data[i, 0]
            if img.min() < -2 or img.max() > 2:  # More than 2 std from mean
                extreme_indices.append(i)
        
        if extreme_indices:
            print(f"Found {len(extreme_indices)} extreme samples")
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            for i, idx in enumerate(extreme_indices[:6]):
                row, col = i // 3, i % 3
                img = self.normalized_data[idx, 0]
                
                im = axes[row, col].imshow(img, cmap='gray', aspect='equal')
                axes[row, col].set_title(f'Extreme Sample {idx}\nRange: [{img.min():.3f}, {img.max():.3f}]')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], shrink=0.8)
            
            plt.suptitle('Extreme Samples (|value| > 2σ)', fontsize=16)
            plt.tight_layout()
            plt.savefig('extreme_samples.png', dpi=300, bbox_inches='tight')
            plt.show()
        
    def model_recommendations(self):
        """Provide recommendations for optimal NCSN++ model design."""
        print("\n" + "="*60)
        print("🎯 MODEL DESIGN RECOMMENDATIONS")
        print("="*60)
        
        # Analyze data characteristics
        norm_std = np.std(self.normalized_data)
        norm_range = np.max(self.normalized_data) - np.min(self.normalized_data)
        spatial_variance = np.var(self.analysis_results['spatial_stats']['mean_image'])
        outlier_ratio = self.analysis_results['outlier_stats']['norm_outliers_z'] / len(self.normalized_data.flatten())
        
        print("Data Characteristics:")
        print(f"  Normalized data std: {norm_std:.4f}")
        print(f"  Normalized data range: {norm_range:.4f}")
        print(f"  Spatial variance: {spatial_variance:.4f}")
        print(f"  Outlier ratio: {outlier_ratio:.4f}")
        print(f"  Image size: 9×9")
        print(f"  Number of samples: {len(self.normalized_data)}")
        
        print("\nNCSN++ Model Recommendations:")
        
        # Architecture recommendations
        if norm_std < 0.5:
            print("  ✅ Low variance data - consider simpler architecture")
            print("  📝 Recommendation: Reduce model complexity")
        else:
            print("  ✅ Moderate variance data - standard architecture should work")
        
        if outlier_ratio > 0.05:
            print("  ⚠️  High outlier ratio - consider robust training")
            print("  📝 Recommendation: Use gradient clipping and robust loss")
        else:
            print("  ✅ Low outlier ratio - standard training should work")
        
        if spatial_variance > 0.1:
            print("  ✅ High spatial variance - attention mechanisms useful")
            print("  📝 Recommendation: Include attention blocks")
        else:
            print("  ⚠️  Low spatial variance - attention may not be necessary")
            print("  📝 Recommendation: Consider simpler architecture")
        
        # Specific recommendations for 9x9 images
        print("\nSpecific Recommendations for 9×9 Images:")
        print("  📝 Use ch_mult = [1, 2, 4] (reduce from typical [1, 2, 4, 8])")
        print("  📝 Set attn_resolutions = [9, 4] (focus on higher resolutions)")
        print("  📝 Increase nf to 128 or 256 (compensate for small spatial size)")
        print("  📝 Use num_res_blocks = 2 (fewer blocks for small images)")
        print("  📝 Consider dropout = 0.2 (moderate regularization)")
        print("  📝 Use batch_size = 64 or 128 (larger batches for stability)")
        
        # Training recommendations
        print("\nTraining Recommendations:")
        print("  📝 Use learning rate = 1e-4 (conservative for small dataset)")
        print("  📝 Use gradient clipping = 1.0 (prevent instability)")
        print("  📝 Use warmup steps = 1000 (gradual learning)")
        print("  📝 Use ema_decay = 0.9999 (stable training)")
        print("  📝 Monitor validation loss closely (small dataset)")
        
        # Data augmentation recommendations
        print("\nData Augmentation Recommendations:")
        print("  📝 Consider adding noise during training")
        print("  📝 Use random horizontal flips if applicable")
        print("  📝 Consider mixup or cutmix for regularization")
        
        self.analysis_results['recommendations'] = {
            'norm_std': norm_std,
            'norm_range': norm_range,
            'spatial_variance': spatial_variance,
            'outlier_ratio': outlier_ratio
        }
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("🚀 Starting Comprehensive Data Analysis")
        print("="*60)
        
        # First calculate true mean and std
        self.calculate_true_mean_std()
        
        self.basic_statistics()
        self.distribution_analysis()
        self.spatial_pattern_analysis()
        self.outlier_analysis()
        self.correlation_analysis()
        self.sample_visualization()
        self.model_recommendations()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        print("📁 Generated files:")
        print("  - distribution_analysis.png")
        print("  - spatial_pattern_analysis.png")
        print("  - outlier_analysis.png")
        print("  - correlation_analysis.png")
        print("  - sample_visualization.png")
        print("  - extreme_samples.png (if applicable)")
        
        return self.analysis_results

def main():
    """Main function to run the analysis."""
    # Path to the training data
    data_path = "../GTO_Halo_DM/data/training_data_boundary_100000_spherical.pkl"
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return
    
    # Create analyzer and run analysis
    analyzer = ComprehensiveDataAnalyzer(data_path)
    results = analyzer.run_complete_analysis()
    
    # Save results to file
    with open('data_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n💾 Analysis results saved to 'data_analysis_results.pkl'")

if __name__ == "__main__":
    main() 