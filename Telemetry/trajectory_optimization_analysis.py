#!/usr/bin/env python3
"""
Trajectory Generation Optimization Analysis
=========================================

This script provides specialized optimization recommendations for NCSN++ models
specifically designed for trajectory generation tasks using the GTO Halo dataset.

Key focus areas:
1. Spatial-temporal pattern analysis
2. Trajectory-specific architecture optimizations
3. Training strategies for trajectory generation
4. Performance metrics for trajectory quality
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

class TrajectoryOptimizationAnalyzer:
    def __init__(self, data_path):
        """Initialize the trajectory optimization analyzer."""
        self.data_path = data_path
        self.dataset = GTOHaloImageDataset(data_path)
        self.raw_data = None
        self.normalized_data = None
        self.classifier_values = None
        self.optimization_results = {}
        
        print("🚀 TRAJECTORY OPTIMIZATION ANALYSIS")
        print("=" * 60)
        print("🔍 Loading and analyzing trajectory data...")
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess trajectory data."""
        # Load raw data
        with open(self.data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.raw_data = np.array(raw_data, dtype=np.float32)
        
        # Extract trajectory components
        self.classifier_values = self.raw_data[:, 0]  # Halo energy
        self.thrust_values = self.raw_data[:, 4:64]   # Thrust profile (60 values)
        self.other_features = np.concatenate([
            self.raw_data[:, 1:4],   # Features 1-3
            self.raw_data[:, 64:]    # Features 64+
        ], axis=1)
        
        # Load normalized data
        normalized_images = []
        normalized_classifiers = []
        
        for i in range(len(self.dataset)):
            img, classifier = self.dataset[i]
            normalized_images.append(img.numpy())
            normalized_classifiers.append(classifier.numpy())
        
        self.normalized_data = np.array(normalized_images)
        self.normalized_classifiers = np.array(normalized_classifiers)
        
        print(f"📊 Dataset loaded: {len(self.raw_data)} trajectories")
        print(f"📊 Trajectory components: {self.raw_data.shape[1]} features")
        print(f"📊 Thrust profile length: {self.thrust_values.shape[1]} steps")
        
    def analyze_trajectory_patterns(self):
        """Analyze spatial-temporal patterns in trajectory data."""
        print("\n" + "="*60)
        print("🛤️  TRAJECTORY PATTERN ANALYSIS")
        print("="*60)
        
        # Analyze thrust profile patterns
        thrust_mean = np.mean(self.thrust_values, axis=0)
        thrust_std = np.std(self.thrust_values, axis=0)
        thrust_correlation = np.corrcoef(self.thrust_values.T)
        
        # Analyze spatial patterns in 9x9 images
        spatial_mean = np.mean(self.normalized_data, axis=0)[0]
        spatial_std = np.std(self.normalized_data, axis=0)[0]
        
        # Calculate trajectory complexity
        trajectory_complexity = np.std(self.thrust_values, axis=1)
        
        # Calculate spatial variance
        spatial_variance = np.var(self.normalized_data, axis=0)[0]
        
        # Print trajectory-specific insights
        print(f"Trajectory Pattern Insights:")
        print(f"  Thrust profile mean range: [{thrust_mean.min():.4f}, {thrust_mean.max():.4f}]")
        print(f"  Thrust profile std range: [{thrust_std.min():.4f}, {thrust_std.max():.4f}]")
        print(f"  High complexity trajectories (>0.1 std): {np.sum(trajectory_complexity > 0.1)}")
        print(f"  Low complexity trajectories (<0.05 std): {np.sum(trajectory_complexity < 0.05)}")
        print(f"  Spatial variance range: [{spatial_variance.min():.4f}, {spatial_variance.max():.4f}]")
        
        self.optimization_results['trajectory_patterns'] = {
            'thrust_mean': thrust_mean,
            'thrust_std': thrust_std,
            'thrust_correlation': thrust_correlation,
            'spatial_mean': spatial_mean,
            'spatial_variance': spatial_variance,
            'trajectory_complexity': trajectory_complexity
        }
        
    def analyze_temporal_dependencies(self):
        """Analyze temporal dependencies in trajectory data."""
        print("\n" + "="*60)
        print("⏰ TEMPORAL DEPENDENCY ANALYSIS")
        print("="*60)
        
        # Analyze temporal correlations in thrust profiles
        thrust_data = self.thrust_values
        
        # Calculate lag correlations (simplified approach)
        max_lag = 5  # Reduced for stability
        lag_correlations = []
        
        for lag in range(1, max_lag + 1):
            try:
                # Use a simpler correlation calculation
                if thrust_data.shape[1] > lag:
                    # Calculate correlation between adjacent time steps
                    x = thrust_data[:, :-lag].flatten()
                    y = thrust_data[:, lag:].flatten()
                    
                    # Remove any NaN values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if np.sum(mask) > 100:  # Need sufficient data
                        corr = np.corrcoef(x[mask], y[mask])[0, 1]
                        lag_correlations.append(corr)
                    else:
                        lag_correlations.append(0.0)
                else:
                    lag_correlations.append(0.0)
            except Exception as e:
                print(f"Warning: Could not calculate lag {lag} correlation: {e}")
                lag_correlations.append(0.0)
        
        # Analyze temporal structure (simplified)
        try:
            # Use a subset of data to avoid memory issues
            subset_size = min(1000, thrust_data.shape[0])
            subset_indices = np.random.choice(thrust_data.shape[0], subset_size, replace=False)
            thrust_subset = thrust_data[subset_indices]
            
            # Calculate temporal autocorrelation on subset
            temporal_autocorr = np.corrcoef(thrust_subset.T)
        except Exception as e:
            print(f"Warning: Could not calculate temporal autocorrelation: {e}")
            temporal_autocorr = np.eye(thrust_data.shape[1])  # Identity matrix as fallback
        
        # Calculate temporal complexity statistics
        temporal_complexity = np.std(thrust_data, axis=1)
        
        # Print temporal dependency insights
        print(f"Temporal Dependency Insights:")
        if len(lag_correlations) > 0:
            print(f"  Average lag correlation (lag=1): {lag_correlations[0]:.4f}")
            print(f"  Lag correlation range: [{min(lag_correlations):.4f}, {max(lag_correlations):.4f}]")
        else:
            print(f"  Average lag correlation (lag=1): N/A")
        
        print(f"  Temporal autocorrelation range: [{temporal_autocorr.min():.4f}, {temporal_autocorr.max():.4f}]")
        print(f"  High temporal complexity (>0.1): {np.sum(temporal_complexity > 0.1)} ({np.sum(temporal_complexity > 0.1)/len(temporal_complexity):.2%})")
        print(f"  Low temporal complexity (<0.05): {np.sum(temporal_complexity < 0.05)} ({np.sum(temporal_complexity < 0.05)/len(temporal_complexity):.2%})")
        print(f"  Mean temporal complexity: {np.mean(temporal_complexity):.4f}")
        print(f"  Std temporal complexity: {np.std(temporal_complexity):.4f}")
        
        # Calculate additional temporal statistics
        temporal_mean = np.mean(thrust_data, axis=1)
        temporal_range = np.max(thrust_data, axis=1) - np.min(thrust_data, axis=1)
        
        print(f"  Mean thrust variation: {np.mean(temporal_range):.4f}")
        print(f"  High variation trajectories (>0.2): {np.sum(temporal_range > 0.2)} ({np.sum(temporal_range > 0.2)/len(temporal_range):.2%})")
        
        self.optimization_results['temporal_dependencies'] = {
            'lag_correlations': lag_correlations,
            'temporal_autocorr': temporal_autocorr,
            'temporal_complexity': temporal_complexity,
            'temporal_range': temporal_range,
            'temporal_mean': temporal_mean
        }
        
    def generate_trajectory_optimizations(self):
        """Generate trajectory-specific optimization recommendations."""
        print("\n" + "="*60)
        print("🎯 TRAJECTORY-SPECIFIC OPTIMIZATIONS")
        print("="*60)
        
        # Extract key insights
        trajectory_complexity = self.optimization_results['trajectory_patterns']['trajectory_complexity']
        temporal_complexity = self.optimization_results['temporal_dependencies']['temporal_complexity']
        spatial_variance = self.optimization_results['trajectory_patterns']['spatial_variance']
        
        # Analyze data characteristics
        high_complexity_ratio = np.sum(trajectory_complexity > 0.1) / len(trajectory_complexity)
        high_temporal_ratio = np.sum(temporal_complexity > 0.1) / len(temporal_complexity)
        high_spatial_variance_ratio = np.sum(spatial_variance > 0.5) / spatial_variance.size
        
        print(f"Trajectory Data Characteristics:")
        print(f"  High complexity trajectories: {high_complexity_ratio:.2%}")
        print(f"  High temporal complexity: {high_temporal_ratio:.2%}")
        print(f"  High spatial variance pixels: {high_spatial_variance_ratio:.2%}")
        print(f"  Dataset size: {len(self.raw_data)} trajectories")
        print(f"  Feature dimensionality: {self.raw_data.shape[1]} features")
        
        print("\n🚀 TRAJECTORY-SPECIFIC NCSN++ OPTIMIZATIONS:")
        
        # Architecture optimizations
        print("\n📐 ARCHITECTURE OPTIMIZATIONS:")
        
        if high_spatial_variance_ratio > 0.1:
            print("  ✅ High spatial variance detected - attention mechanisms crucial")
            print("  📝 Recommendation: Use attention at all resolutions [9, 4, 2]")
        else:
            print("  ⚠️  Low spatial variance - attention may not be necessary")
            print("  📝 Recommendation: Use attention only at higher resolutions [9, 4]")
        
        if high_temporal_ratio > 0.2:
            print("  ✅ High temporal complexity detected - need temporal modeling")
            print("  📝 Recommendation: Increase model capacity for temporal patterns")
        else:
            print("  ✅ Moderate temporal complexity - standard architecture sufficient")
        
        # Specific configuration recommendations
        print("\n⚙️  OPTIMAL CONFIGURATION FOR TRAJECTORY GENERATION:")
        
        # Base features
        if high_complexity_ratio > 0.3:
            print("  📝 nf = 256 (high complexity trajectories need more capacity)")
        else:
            print("  📝 nf = 128 (moderate complexity - standard capacity)")
        
        # Channel multipliers
        print("  📝 ch_mult = [1, 2, 4] (reduced for 9×9 images)")
        
        # Attention resolutions
        if high_spatial_variance_ratio > 0.1:
            print("  📝 attn_resolutions = [9, 4, 2] (full attention for complex patterns)")
        else:
            print("  📝 attn_resolutions = [9, 4] (focused attention)")
        
        # ResNet blocks
        if high_temporal_ratio > 0.2:
            print("  📝 num_res_blocks = 3 (more blocks for temporal complexity)")
        else:
            print("  📝 num_res_blocks = 2 (standard for 9×9 images)")
        
        # Dropout
        if high_complexity_ratio > 0.3:
            print("  📝 dropout = 0.3 (higher regularization for complex data)")
        else:
            print("  📝 dropout = 0.2 (moderate regularization)")
        
        # Training optimizations
        print("\n🎓 TRAINING OPTIMIZATIONS FOR TRAJECTORY GENERATION:")
        
        # Learning rate
        if high_complexity_ratio > 0.3:
            print("  📝 learning_rate = 5e-5 (conservative for complex trajectories)")
        else:
            print("  📝 learning_rate = 1e-4 (standard for trajectory generation)")
        
        # Batch size
        print("  📝 batch_size = 128 (larger batches for trajectory stability)")
        
        # Gradient clipping
        print("  📝 gradient_clip = 1.0 (prevent instability in trajectory training)")
        
        # Warmup
        print("  📝 warmup_steps = 2000 (gradual learning for complex trajectories)")
        
        # EMA decay
        print("  📝 ema_decay = 0.9999 (stable training for trajectory generation)")
        
        # Data augmentation
        print("\n🔄 TRAJECTORY-SPECIFIC DATA AUGMENTATION:")
        print("  📝 Add Gaussian noise (σ=0.01) during training")
        print("  📝 Use trajectory-specific noise injection")
        print("  📝 Consider temporal jittering for robustness")
        print("  📝 Implement trajectory-specific mixup")
        
        # Loss function optimizations
        print("\n📊 LOSS FUNCTION OPTIMIZATIONS:")
        print("  📝 Use weighted MSE loss for trajectory components")
        print("  📝 Implement trajectory-specific loss weighting")
        print("  📝 Consider temporal consistency loss")
        print("  📝 Add boundary constraint loss for thrust values")
        
        # Performance monitoring
        print("\n📈 TRAJECTORY-SPECIFIC METRICS:")
        print("  📝 Monitor thrust profile accuracy")
        print("  📝 Track trajectory feasibility rate")
        print("  📝 Measure temporal consistency")
        print("  📝 Validate boundary constraints")
        print("  📝 Assess trajectory smoothness")
        
        # Advanced optimizations
        print("\n🚀 ADVANCED OPTIMIZATIONS:")
        
        if high_complexity_ratio > 0.3:
            print("  📝 Consider hierarchical attention for complex trajectories")
            print("  📝 Implement multi-scale temporal modeling")
            print("  📝 Use trajectory-specific conditioning")
        else:
            print("  📝 Standard attention mechanisms sufficient")
            print("  📝 Single-scale temporal modeling adequate")
        
        print("  📝 Implement trajectory-specific sampling strategies")
        print("  📝 Use guided diffusion for trajectory generation")
        print("  📝 Consider trajectory-specific guidance")
        
        # Save optimization results
        self.optimization_results['recommendations'] = {
            'high_complexity_ratio': high_complexity_ratio,
            'high_temporal_ratio': high_temporal_ratio,
            'high_spatial_variance_ratio': high_spatial_variance_ratio,
            'optimal_nf': 256 if high_complexity_ratio > 0.3 else 128,
            'optimal_attn_resolutions': [9, 4, 2] if high_spatial_variance_ratio > 0.1 else [9, 4],
            'optimal_num_res_blocks': 3 if high_temporal_ratio > 0.2 else 2,
            'optimal_dropout': 0.3 if high_complexity_ratio > 0.3 else 0.2,
            'optimal_learning_rate': 5e-5 if high_complexity_ratio > 0.3 else 1e-4
        }
        
    def run_complete_optimization_analysis(self):
        """Run the complete trajectory optimization analysis."""
        print("🚀 Starting Trajectory Optimization Analysis")
        print("=" * 60)
        
        self.analyze_trajectory_patterns()
        self.analyze_temporal_dependencies()
        self.generate_trajectory_optimizations()
        
        print("\n" + "="*60)
        print("✅ TRAJECTORY OPTIMIZATION ANALYSIS COMPLETE")
        print("="*60)
        
        return self.optimization_results

def main():
    """Main function to run trajectory optimization analysis."""
    # Path to the training data
    data_path = "GTO_Halo_DM/data/training_data_boundary_100000.pkl"
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return
    
    # Create analyzer and run analysis
    analyzer = TrajectoryOptimizationAnalyzer(data_path)
    results = analyzer.run_complete_optimization_analysis()
    
    # Save results to file
    with open('trajectory_optimization_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n💾 Optimization results saved to 'trajectory_optimization_results.pkl'")

if __name__ == "__main__":
    main() 