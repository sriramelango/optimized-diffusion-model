"""
Machine Learning Statistics Module for Diffusion Model Benchmarking

This module provides standard ML evaluation metrics for diffusion models including:
- Distribution metrics (KL divergence, Wasserstein distance)
- Error metrics (MSE, MAE, mean error, standard deviation)
- Image metrics (PSNR, SSIM) when applicable
- Sampling efficiency metrics
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
import json
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Reflected-Diffusion'))

# Import from Reflected-Diffusion
import sampling
import sde_lib
import utils
import losses
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets


@dataclass
class MLStatisticsConfig:
    """Configuration for ML statistics benchmarking."""
    # Model and data config
    model_path: str
    config_path: str
    data_path: str
    
    # Sampling config
    num_samples: int = 1000
    batch_size: int = 100
    sampling_method: str = "pc"
    guidance_weight: float = 0.0
    
    # Metrics config
    compute_fid: bool = True
    compute_lpips: bool = True
    compute_psnr: bool = True
    compute_ssim: bool = True
    
    # Output config
    output_dir: str = "ml_statistics_results"
    save_samples: bool = True
    save_plots: bool = True
    
    # Device config
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class MLStatisticsBenchmarker:
    """Machine learning statistics benchmarking for diffusion models."""
    
    def __init__(self, config: MLStatisticsConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Default model path - use Training Runs directory structure
        if not hasattr(self.config, 'model_path') or not self.config.model_path:
            # Look for the most recent training run
            training_runs_dir = "../Reflected-Diffusion/Training Runs"
            if os.path.exists(training_runs_dir):
                runs = [d for d in os.listdir(training_runs_dir) if os.path.isdir(os.path.join(training_runs_dir, d))]
                if runs:
                    # Sort by datetime and get the most recent
                    runs.sort(reverse=True)
                    latest_run = runs[0]
                    self.config.model_path = os.path.join(training_runs_dir, latest_run)
                    print(f"Using latest training run: {self.config.model_path}")
                else:
                    # Fallback to old structure
                    self.config.model_path = "runs/GTOHaloImage/2025.07.07/182107"
            else:
                # Fallback to old structure
                self.config.model_path = "runs/GTOHaloImage/2025.07.07/182107"
        
        # Load model and config
        self.load_model()
        
        # Initialize metrics
        self.metrics = {}
        self.samples = []
        self.sampling_times = []
        
    def load_model(self):
        """Load the trained diffusion model."""
        print(f"Loading model from {self.config.model_path}")
        
        # Load config
        if os.path.isdir(self.config.config_path):
            config_path = os.path.join(self.config.config_path, ".hydra", "config.yaml")
        else:
            config_path = self.config.config_path
            
        self.cfg = OmegaConf.load(config_path)
        
        # Create model
        try:
            self.score_model = mutils.create_model(self.cfg).to(self.device)
            self.ema = ExponentialMovingAverage(self.score_model.parameters(), decay=self.cfg.model.ema_rate)
        except Exception as e:
            print(f"Error creating model: {e}")
            print("Trying alternative model creation...")
            # Try to create model with explicit architecture
            if hasattr(self.cfg.model, 'name'):
                print(f"Model architecture: {self.cfg.model.name}")
            raise e
        
        # Load checkpoint
        checkpoint_path = os.path.join(self.config.model_path, "checkpoints-meta", "checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            # Try to find the latest checkpoint
            checkpoint_dir = os.path.join(self.config.model_path, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
                    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle checkpoint structure
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    # Load model state dict
                    self.score_model.load_state_dict(checkpoint['model'])
                    print("Loaded model state dict")
                
                if 'ema' in checkpoint:
                    # Load EMA state dict
                    self.ema.load_state_dict(checkpoint['ema'])
                    print("Loaded EMA state dict")
                
                if 'step' in checkpoint:
                    print(f"Checkpoint step: {checkpoint['step']}")
            else:
                # Fallback: assume it's a direct state dict
                self.score_model.load_state_dict(checkpoint)
                print("Loaded direct state dict")
            
            # Copy EMA parameters to model
            self.ema.copy_to(self.score_model.parameters())
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}")
        
        # Setup sampling
        self.sde = sde_lib.RVESDE(
            sigma_min=self.cfg.sde.sigma_min,
            sigma_max=self.cfg.sde.sigma_max,
            N=self.cfg.sde.num_scales
        )
        
        sampling_shape = (
            self.config.batch_size,
            self.cfg.data.num_channels,
            self.cfg.data.image_size,
            self.cfg.data.image_size
        )
        
        self.sampling_fn = sampling.get_sampling_fn(
            self.cfg, self.sde, sampling_shape, 1e-5, self.device
        )
        
    def generate_samples(self) -> Tuple[np.ndarray, List[float]]:
        """Generate samples and measure sampling time."""
        print(f"Generating {self.config.num_samples} samples...")
        
        samples = []
        sampling_times = []
        
        num_batches = (self.config.num_samples + self.config.batch_size - 1) // self.config.batch_size
        
        for i in range(num_batches):
            batch_size = min(self.config.batch_size, self.config.num_samples - i * self.config.batch_size)
            
            # Prepare class labels for conditional generation
            if hasattr(self.cfg.data, 'classes') and self.cfg.data.classes:
                # For GTO Halo data, use uniform sampling of class labels in [0, 1]
                class_labels = torch.rand(batch_size, 1, device=self.device)
            else:
                class_labels = None
            
            # Generate samples
            start_time = time.time()
            
            self.ema.store(self.score_model.parameters())
            self.ema.copy_to(self.score_model.parameters())
            
            sample, n_steps = self.sampling_fn(
                self.score_model,
                weight=self.config.guidance_weight,
                class_labels=class_labels
            )
            
            self.ema.restore(self.score_model.parameters())
            
            end_time = time.time()
            sampling_time = end_time - start_time
            
            # Convert to numpy and clip to [0, 1]
            sample_np = sample.cpu().numpy()
            sample_np = np.clip(sample_np, 0, 1)
            
            samples.append(sample_np)
            sampling_times.append(sampling_time)
            
            print(f"Batch {i+1}/{num_batches}: Generated {batch_size} samples in {sampling_time:.2f}s")
        
        # Concatenate all samples
        all_samples = np.concatenate(samples, axis=0)
        all_samples = all_samples[:self.config.num_samples]  # Ensure exact number
        
        # --- FLATTEN TO (N, 67) ---
        # Model outputs (N, 1, 9, 9) = (N, 81) total values
        # Original data is 67-dimensional, padded with 14 zeros to make 81
        # So we need to flatten and take only the first 67 values
        all_samples_flat = all_samples.reshape(all_samples.shape[0], -1)  # (N, 81)
        all_samples_flat = all_samples_flat[:, :67]  # Remove padding, keep only first 67 values
        
        print(f"Model output shape: {all_samples.shape}")
        print(f"Flattened shape: {all_samples_flat.shape}")
        
        return all_samples_flat, sampling_times
    
    def compute_standard_metrics(self, samples: np.ndarray, reference_data: np.ndarray) -> Dict[str, float]:
        """Compute standard ML evaluation metrics."""
        metrics = {}
        
        # Ensure we have the same number of samples for comparison
        num_samples = min(samples.shape[0], reference_data.shape[0])
        samples = samples[:num_samples]
        reference_data = reference_data[:num_samples]
        
        # Both should be (N, 67) already
        samples_2d = samples
        reference_2d = reference_data
        
        # Check for matching feature dimension
        if samples_2d.shape[1] != reference_2d.shape[1]:
            print(f"Feature dimension mismatch: samples {samples_2d.shape[1]}, reference {reference_2d.shape[1]}")
            min_dim = min(samples_2d.shape[1], reference_2d.shape[1])
            samples_2d = samples_2d[:, :min_dim]
            reference_2d = reference_2d[:, :min_dim]
        
        # MSE and MAE
        metrics['mse'] = mean_squared_error(reference_2d, samples_2d)
        metrics['mae'] = mean_absolute_error(reference_2d, samples_2d)
        
        # Statistical metrics
        metrics['mean_error'] = np.mean(np.abs(samples_2d - reference_2d))
        metrics['std_error'] = np.std(np.abs(samples_2d - reference_2d))
        
        # Distribution metrics
        metrics['kl_divergence'] = self.compute_kl_divergence(samples_2d, reference_2d)
        metrics['wasserstein_distance'] = self.compute_wasserstein_distance(samples_2d, reference_2d)
        
        # Skip image metrics for vector data
        print("Skipping image metrics: data is not image-like, but a flattened vector.")
        
        return metrics
    
    def compute_kl_divergence(self, samples: np.ndarray, reference: np.ndarray) -> float:
        """Compute KL divergence between sample and reference distributions."""
        try:
            # Use histogram-based KL divergence
            hist_samples, _ = np.histogram(samples.flatten(), bins=50, density=True)
            hist_reference, _ = np.histogram(reference.flatten(), bins=50, density=True)
            
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            hist_samples = hist_samples + eps
            hist_reference = hist_reference + eps
            
            # Normalize
            hist_samples = hist_samples / np.sum(hist_samples)
            hist_reference = hist_reference / np.sum(hist_reference)
            
            kl_div = np.sum(hist_reference * np.log(hist_reference / hist_samples))
            return float(kl_div)
        except:
            return float('inf')
    
    def compute_wasserstein_distance(self, samples: np.ndarray, reference: np.ndarray) -> float:
        """Compute Wasserstein distance between sample and reference distributions."""
        try:
            from scipy.stats import wasserstein_distance
            return float(wasserstein_distance(samples.flatten(), reference.flatten()))
        except:
            return float('inf')
    
    def compute_image_metrics(self, samples: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        """Compute image-specific metrics (PSNR, SSIM)."""
        metrics = {}
        
        try:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            
            # Compute metrics for each sample
            psnr_values = []
            ssim_values = []
            
            for i in range(min(samples.shape[0], reference.shape[0])):
                sample_img = samples[i].transpose(1, 2, 0)  # CHW -> HWC
                reference_img = reference[i].transpose(1, 2, 0)
                
                # Convert to [0, 255] range
                sample_img = (sample_img * 255).astype(np.uint8)
                reference_img = (reference_img * 255).astype(np.uint8)
                
                # Compute PSNR
                psnr = peak_signal_noise_ratio(reference_img, sample_img, data_range=255)
                psnr_values.append(psnr)
                
                # Compute SSIM
                ssim = structural_similarity(reference_img, sample_img, data_range=255, multichannel=True)
                ssim_values.append(ssim)
            
            metrics['psnr_mean'] = np.mean(psnr_values)
            metrics['psnr_std'] = np.std(psnr_values)
            metrics['ssim_mean'] = np.mean(ssim_values)
            metrics['ssim_std'] = np.std(ssim_values)
            
        except ImportError:
            print("Warning: skimage not available. Skipping image metrics.")
        
        return metrics
    
    def compute_sampling_efficiency_metrics(self, sampling_times: List[float]) -> Dict[str, float]:
        """Compute sampling efficiency metrics."""
        metrics = {
            'total_sampling_time': sum(sampling_times),
            'average_sampling_time_per_sample': np.mean(sampling_times),
            'sampling_time_std': np.std(sampling_times),
            'samples_per_second': len(sampling_times) / sum(sampling_times),
            'min_sampling_time': min(sampling_times),
            'max_sampling_time': max(sampling_times)
        }
        
        return metrics
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete ML statistics benchmarking pipeline."""
        print("Starting ML statistics benchmark...")
        
        # Generate samples
        samples, sampling_times = self.generate_samples()
        
        # Load reference data
        reference_data = self.load_reference_data()
        
        # Compute metrics
        results = {}
        
        # Standard ML metrics
        if reference_data is not None:
            results['standard_metrics'] = self.compute_standard_metrics(samples, reference_data)
        
        # Sampling efficiency metrics
        results['sampling_efficiency'] = self.compute_sampling_efficiency_metrics(sampling_times)
        
        # Save results
        self.save_results(results, samples)
        
        # Generate plots
        if self.config.save_plots:
            self.generate_plots(results, samples)
        
        return results
    
    def load_reference_data(self) -> Optional[np.ndarray]:
        """Load reference data for comparison."""
        try:
            # Try to load from the data path
            if os.path.exists(self.config.data_path):
                if self.config.data_path.endswith('.pkl'):
                    import pickle
                    with open(self.config.data_path, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, np.ndarray):
                        return data
                    elif isinstance(data, dict) and 'data' in data:
                        return data['data']
                elif self.config.data_path.endswith('.npy'):
                    return np.load(self.config.data_path)
            
            # Try to load from dataset
            train_ds, eval_ds = datasets.get_dataset(self.cfg)
            eval_iter = iter(eval_ds)
            
            # Get a batch of reference data
            batch = next(eval_iter)
            reference_data = batch[0].cpu().numpy()
            
            return reference_data
            
        except Exception as e:
            print(f"Warning: Could not load reference data: {e}")
            return None
    
    def save_results(self, results: Dict[str, Any], samples: np.ndarray):
        """Save benchmark results."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(self.config.output_dir, 'ml_statistics_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save samples
        if self.config.save_samples:
            np.save(os.path.join(self.config.output_dir, 'generated_samples.npy'), samples)
        
        # Save summary
        self.save_summary(results)
    
    def save_summary(self, results: Dict[str, Any]):
        """Save a human-readable summary of results."""
        summary = []
        summary.append("=" * 60)
        summary.append("ML STATISTICS BENCHMARK RESULTS")
        summary.append("=" * 60)
        
        if 'standard_metrics' in results:
            summary.append("\nSTANDARD METRICS:")
            for key, value in results['standard_metrics'].items():
                summary.append(f"  {key}: {value:.6f}")
        
        if 'sampling_efficiency' in results:
            summary.append("\nSAMPLING EFFICIENCY:")
            for key, value in results['sampling_efficiency'].items():
                summary.append(f"  {key}: {value:.6f}")
        
        summary.append("\n" + "=" * 60)
        
        with open(os.path.join(self.config.output_dir, 'summary.txt'), 'w') as f:
            f.write('\n'.join(summary))
        
        print('\n'.join(summary))
    
    def generate_plots(self, results: Dict[str, Any], samples: np.ndarray):
        """Generate visualization plots."""
        os.makedirs(os.path.join(self.config.output_dir, 'plots'), exist_ok=True)
        
        # Sample distribution plots
        self.plot_sample_distributions(samples)
        
        # Metrics comparison plots
        if 'standard_metrics' in results:
            self.plot_metrics_comparison(results['standard_metrics'])
        
        # Sampling efficiency plots
        if 'sampling_efficiency' in results:
            self.plot_sampling_efficiency(results['sampling_efficiency'])
    
    def plot_sample_distributions(self, samples: np.ndarray):
        """Plot sample distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Flatten samples for distribution analysis
        samples_flat = samples.reshape(samples.shape[0], -1)
        
        # Overall distribution
        axes[0, 0].hist(samples_flat.flatten(), bins=50, alpha=0.7, density=True)
        axes[0, 0].set_title('Overall Sample Distribution')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        
        # Per-dimension statistics
        means = np.mean(samples_flat, axis=0)
        stds = np.std(samples_flat, axis=0)
        
        axes[0, 1].plot(means)
        axes[0, 1].set_title('Mean per Dimension')
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Mean')
        
        axes[1, 0].plot(stds)
        axes[1, 0].set_title('Std per Dimension')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Standard Deviation')
        
        # Value range
        mins = np.min(samples_flat, axis=0)
        maxs = np.max(samples_flat, axis=0)
        
        axes[1, 1].fill_between(range(len(mins)), mins, maxs, alpha=0.5)
        axes[1, 1].set_title('Value Range per Dimension')
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', 'sample_distributions.png'), dpi=300)
        plt.close()
    
    def plot_metrics_comparison(self, metrics: Dict[str, float]):
        """Plot metrics comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        axes[0].bar(metric_names, metric_values)
        axes[0].set_title('Standard Metrics')
        axes[0].set_ylabel('Value')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Log scale for better visualization
        positive_metrics = {k: v for k, v in metrics.items() if v > 0}
        if positive_metrics:
            axes[1].bar(positive_metrics.keys(), positive_metrics.values())
            axes[1].set_yscale('log')
            axes[1].set_title('Standard Metrics (Log Scale)')
            axes[1].set_ylabel('Value (log scale)')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', 'metrics_comparison.png'), dpi=300)
        plt.close()
    
    def plot_sampling_efficiency(self, metrics: Dict[str, float]):
        """Plot sampling efficiency metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time metrics
        time_metrics = ['total_sampling_time', 'average_sampling_time_per_sample', 'min_sampling_time', 'max_sampling_time']
        time_values = [metrics.get(k, 0) for k in time_metrics]
        
        axes[0].bar(time_metrics, time_values)
        axes[0].set_title('Sampling Time Metrics')
        axes[0].set_ylabel('Time (seconds)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Efficiency metrics
        efficiency_metrics = ['samples_per_second', 'sampling_time_std']
        efficiency_values = [metrics.get(k, 0) for k in efficiency_metrics]
        
        axes[1].bar(efficiency_metrics, efficiency_values)
        axes[1].set_title('Efficiency Metrics')
        axes[1].set_ylabel('Value')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', 'sampling_efficiency.png'), dpi=300)
        plt.close() 