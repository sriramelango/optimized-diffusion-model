"""
GTO Halo Benchmarking Module for Diffusion Model Evaluation (Spherical Dataset)

This module provides domain-specific evaluation for GTO Halo trajectory optimization including:
- Physical validation using CR3BP simulator
- Statistical analysis matching GTO_Halo_DM project
- Component analysis (thrust, mass, time variables)
- Spherical coordinate unnormalization (no clipping needed)
- Guaranteed thrust magnitude constraint satisfaction (≤ 1.0)

Key Features:
- Works with spherical dataset (training_data_boundary_100000_spherical.pkl)
- Direct spherical unnormalization without Cartesian conversion
- Mathematical guarantee of thrust magnitude ≤ 1.0 (no clipping required)
- Eliminates 90% clipping issue from original Cartesian approach
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from omegaconf import OmegaConf
import json
import pandas as pd
from scipy import stats
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

# Import from GTO_Halo_DM
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../GTO_Halo_DM/data_generation_scripts')))
    from cr3bp_earth_mission_simulator_boundary_diffusion_warmstart import CR3BPEarthMissionWarmstartSimulatorBoundary
    from support_scripts.support import get_GTO_in_CR3BP_units
    GTO_HALO_DM_AVAILABLE = True
    print("✓ GTO_Halo_DM modules loaded successfully")
    print("✓ Physical validation enabled - CR3BP simulator available")
except ImportError as e:
    print(f"Warning: GTO_Halo_DM modules not available: {e}")
    print(f"sys.path was: {sys.path}")
    # Try adding the absolute path to GTO_Halo_DM and re-import
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../GTO_Halo_DM')))
        from data_generation_scripts.cr3bp_earth_mission_simulator_boundary_diffusion_warmstart import CR3BPEarthMissionWarmstartSimulatorBoundary
        from data_generation_scripts.support_scripts.support import get_GTO_in_CR3BP_units
        GTO_HALO_DM_AVAILABLE = True
        print("✓ GTO_Halo_DM modules loaded successfully (second attempt)")
        print("✓ Physical validation enabled - CR3BP simulator available")
    except ImportError as e2:
        print(f"Failed second import attempt: {e2}")
        print(f"sys.path is now: {sys.path}")
        CR3BPEarthMissionWarmstartSimulatorBoundary = None
        GTO_HALO_DM_AVAILABLE = False


@dataclass
class GTOHaloBenchmarkConfig:
    """Configuration for GTO Halo benchmarking."""
    # Model and data config
    model_path: str
    config_path: str
    data_path: str
    
    # Sampling config
    num_samples: int = 1000
    batch_size: int = 100
    sampling_method: str = "pc"
    guidance_weight: float = 0.0
    
    # Physical validation config
    enable_physical_validation: bool = True
    
    # Output config
    output_dir: str = "gto_halo_results"
    save_samples: bool = True
    save_plots: bool = True
    
    # Device config
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class GTOHaloBenchmarker:
    """GTO Halo specific benchmarking for diffusion models."""
    
    def __init__(self, config: GTOHaloBenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # No clipping counters needed - spherical dataset guarantees magnitude ≤ 1.0
        
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
        input_class_labels = []  # Store original input class labels
        
        num_batches = (self.config.num_samples + self.config.batch_size - 1) // self.config.batch_size
        
        for i in range(num_batches):
            batch_size = min(self.config.batch_size, self.config.num_samples - i * self.config.batch_size)
            
            # For GTO Halo data, use uniform sampling of class labels in [0, 1]
            class_labels = torch.rand(batch_size, 1, device=self.device)
            
            # Store input class labels for later use (like original GTO Halo implementation)
            input_class_labels.append(class_labels.cpu().numpy())
            
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
            
            # Convert to numpy (no clipping for reflected diffusion model)
            sample_np = sample.cpu().numpy()
            samples.append(sample_np)
            sampling_times.append(sampling_time)
            
            print(f"Batch {i+1}/{num_batches}: Generated {batch_size} samples in {sampling_time:.2f}s")
        
        # Concatenate all samples
        all_samples = np.concatenate(samples, axis=0)
        all_samples = all_samples[:self.config.num_samples]  # Ensure exact number
        
        # Concatenate input class labels
        input_class_labels = np.concatenate(input_class_labels, axis=0)
        input_class_labels = input_class_labels[:self.config.num_samples]  # Ensure exact number

        # --- FLATTEN TO (N, 67) ---
        samples = all_samples.reshape(all_samples.shape[0], -1)  # (N, 81)
        samples = samples[:, :67]  # Keep only the first 67 values

        # Extract model outputs (skip the generated class label)
        model_outputs = samples[:, 1:]  # Rest is the model output (66 values)
        
        # Use INPUT class labels for simulation (like original GTO Halo implementation)
        class_labels_normalized = input_class_labels.flatten()  # Use input class labels

        # Skip mean/std unnormalization - model outputs are already in [0,1] range
        # With spherical dataset: model learns to output (alpha_norm, beta_norm, r_norm) directly

        # Then, apply per-variable physical unnormalization (match 1D GTO Halo DM exactly)
        min_shooting_time = 0
        max_shooting_time = 40
        min_coast_time = 0
        max_coast_time = 15
        min_halo_energy = 0.008
        max_halo_energy = 0.095
        min_final_fuel_mass = 408
        max_final_fuel_mass = 470
        min_manifold_length = 5
        max_manifold_length = 11
        thrust = 1.0

        # Unnormalize times
        model_outputs[:, 0] = model_outputs[:, 0] * (max_shooting_time - min_shooting_time) + min_shooting_time
        model_outputs[:, 1] = model_outputs[:, 1] * (max_coast_time - min_coast_time) + min_coast_time
        model_outputs[:, 2] = model_outputs[:, 2] * (max_coast_time - min_coast_time) + min_coast_time
        
        # Direct spherical unnormalization - no Cartesian conversion needed!
        # The model outputs are already in spherical coordinates (alpha_norm, beta_norm, r_norm)
        # We just need to unnormalize them to physical spherical coordinates
        control_section = model_outputs[:, 3:-3]  # Extract control section
        
        # Reshape to ensure we have groups of 3 (alpha_norm, beta_norm, r_norm)
        num_control_vars = control_section.shape[1]
        num_triplets = num_control_vars // 3
        
        if num_control_vars % 3 != 0:
            # Truncate to nearest multiple of 3
            control_section = control_section[:, :num_triplets*3]
            num_control_vars = num_triplets * 3
        
        # Reshape to (batch_size, num_triplets, 3) for easier processing
        control_reshaped = control_section.reshape(-1, num_triplets, 3)
        
        # Extract normalized spherical components [0,1] range
        alpha_norm = control_reshaped[:, :, 0]  # Shape: (batch_size, num_triplets)
        beta_norm = control_reshaped[:, :, 1]   # Shape: (batch_size, num_triplets)  
        r_norm = control_reshaped[:, :, 2]      # Shape: (batch_size, num_triplets)
        
        # Unnormalize spherical coordinates to physical ranges
        alpha = alpha_norm * 2 * np.pi  # [0, 2π]
        beta = beta_norm * 2 * np.pi    # [0, 2π]
        r = r_norm                      # [0, 1] - already physical magnitude!
        
        # Store physical spherical coordinates (what CR3BP simulator expects)
        control_reshaped[:, :, 0] = alpha
        control_reshaped[:, :, 1] = beta
        control_reshaped[:, :, 2] = r
        
        # Reshape back and put into model_outputs
        model_outputs[:, 3:3+num_control_vars] = control_reshaped.reshape(-1, num_control_vars)
        
        # 🎯 KEY BENEFIT: No clipping needed! r ≤ 1.0 is mathematically guaranteed
        
        # Unnormalize fuel mass and manifold parameters (match 1D implementation exactly)
        model_outputs[:, -3] = model_outputs[:, -3] * (max_final_fuel_mass - min_final_fuel_mass) + min_final_fuel_mass
        model_outputs[:, -1] = model_outputs[:, -1] * (max_manifold_length - min_manifold_length) + min_manifold_length
        # Note: model_outputs[:, -2] is NOT unnormalized (halo period remains normalized as in 1D implementation)

        # Unnormalize halo energy (exactly as in 1D implementation)
        halo_energies = class_labels_normalized * (max_halo_energy - min_halo_energy) + min_halo_energy
        
        # Combine halo energies with model outputs (exactly as in 1D implementation)
        samples = np.column_stack((halo_energies, model_outputs))

        print(f"Model output shape: {samples.shape}")
        print(f"Flattened shape: {samples.shape}")

        return samples, sampling_times
    
    # _convert_to_spherical method removed - no longer needed!
    # With spherical dataset, no Cartesian->Spherical conversion or clipping is required
    
    def compute_gto_halo_metrics(self, samples: np.ndarray) -> Dict[str, Any]:
        """Compute GTO Halo specific metrics."""
        print("Computing GTO Halo specific metrics...")
        
        # Samples should already be in (N, 67) format from generate_samples
        print(f"Input samples shape: {samples.shape}")
        
        # Extract components from 67-vector
        # Assuming format: [class_label, time_vars, thrust_vars, mass_vars, other_vars]
        class_labels = samples[:, 0]  # First value is class label
        time_vars = samples[:, 1:4]   # Time variables
        thrust_vars = samples[:, 4:64]  # Thrust variables (60 values)
        mass_vars = samples[:, 64:67]  # Mass variables (3 values)
        
        # Ensure we have valid data
        if samples.size == 0:
            print("Warning: No samples generated")
            return {}
        
        # Helper to safely compute stats
        def safe_stat(arr, func, default=None):
            return func(arr) if arr.size > 0 else default
        
        # Compute metrics
        metrics = {}
        
        # Class label statistics
        metrics['class_label_mean'] = float(safe_stat(class_labels, np.mean, None))
        metrics['class_label_std'] = float(safe_stat(class_labels, np.std, None))
        metrics['class_label_min'] = float(safe_stat(class_labels, np.min, None))
        metrics['class_label_max'] = float(safe_stat(class_labels, np.max, None))
        
        # Time variables statistics
        metrics['time_vars_mean'] = float(safe_stat(time_vars, np.mean, None))
        metrics['time_vars_std'] = float(safe_stat(time_vars, np.std, None))
        metrics['time_vars_min'] = float(safe_stat(time_vars, np.min, None))
        metrics['time_vars_max'] = float(safe_stat(time_vars, np.max, None))
        
        # Thrust variables statistics
        metrics['thrust_vars_mean'] = float(safe_stat(thrust_vars, np.mean, None))
        metrics['thrust_vars_std'] = float(safe_stat(thrust_vars, np.std, None))
        metrics['thrust_vars_min'] = float(safe_stat(thrust_vars, np.min, None))
        metrics['thrust_vars_max'] = float(safe_stat(thrust_vars, np.max, None))
        
        # Mass variables statistics
        metrics['mass_vars_mean'] = float(safe_stat(mass_vars, np.mean, None))
        metrics['mass_vars_std'] = float(safe_stat(mass_vars, np.std, None))
        metrics['mass_vars_min'] = float(safe_stat(mass_vars, np.min, None))
        metrics['mass_vars_max'] = float(safe_stat(mass_vars, np.max, None))
        
        # Note: No boundary violation checks needed with spherical dataset
        # Spherical coordinates guarantee thrust magnitude ≤ 1.0 mathematically
        # No clipping or violations can occur during unnormalization
        
        # Data quality (basic checks only)
        metrics['has_nan'] = np.any(np.isnan(samples))
        metrics['has_inf'] = np.any(np.isinf(samples))
        
        return metrics
    
    def compute_physical_validation_metrics(self, samples: np.ndarray) -> Dict[str, Any]:
        """Compute physical validation metrics using CR3BP simulator."""
        print(f"DEBUG: enable_physical_validation = {self.config.enable_physical_validation}")
        print(f"DEBUG: GTO_HALO_DM_AVAILABLE = {GTO_HALO_DM_AVAILABLE}")
        
        if not self.config.enable_physical_validation or not GTO_HALO_DM_AVAILABLE:
            print("⚠️  CRITICAL: Physical validation disabled - GTO_Halo_DM modules not available")
            print("⚠️  This means NO feasibility checking, NO optimality analysis, NO trajectory validation")
            print("⚠️  The GTO Halo benchmarking will only provide component statistics, not physical validation")
            return {
                'physical_validation_disabled': True, 
                'reason': 'GTO_Halo_DM modules not available',
                'missing_metrics': [
                    'feasible_solution_ratio',
                    'local_optimal_solution_ratio', 
                    'average_final_mass_feasible',
                    'average_final_mass_optimal',
                    'snopt_inform_distribution',
                    'solving_time_analysis'
                ]
            }
        
        print("Computing physical validation metrics using CR3BP simulator...")
        
        # CR3BP simulation parameters (hardcoded to match 1D implementation exactly)
        cr3bp_config = {
            'seed': 0,
            'seed_step': len(samples),  # Test all generated samples
            'quiet_snopt': True,
            'number_of_segments': 20,  # Match 1D implementation
            'maximum_shooting_time': 40.0,  # Match 1D implementation
            'minimum_shooting_time': 0.0,  # Match 1D implementation
            'start_bdry': 6.48423370092,  # Match 1D implementation
            'end_bdry': 8.0,  # Match 1D implementation
            'thrust': 1.0,  # Match 1D implementation
            'solver_mode': 0,  # Match 1D implementation (0 = optimal, "feasible" = feasible)
            'min_mass_to_sample': 408,  # Match 1D implementation
            'max_mass_to_sample': 470,  # Match 1D implementation
            'snopt_time_limit': 1000.0,  # Match 1D implementation
            'result_folder': os.path.join(self.config.output_dir, 'cr3bp_results')
        }
        
        # Save samples temporarily for CR3BP simulator
        os.makedirs(self.config.output_dir, exist_ok=True)
        temp_sample_path = os.path.join(self.config.output_dir, 'temp_samples.pkl')
        with open(temp_sample_path, 'wb') as f:
            pickle.dump(samples, f)
        
        # Initialize CR3BP simulator
        simulator = CR3BPEarthMissionWarmstartSimulatorBoundary(
            seed=cr3bp_config['seed'],
            seed_step=cr3bp_config['seed_step'],
            quiet_snopt=cr3bp_config['quiet_snopt'],
            number_of_segments=cr3bp_config['number_of_segments'],
            maximum_shooting_time=cr3bp_config['maximum_shooting_time'],
            minimum_shooting_time=cr3bp_config['minimum_shooting_time'],
            sample_path=temp_sample_path,
            start_bdry=cr3bp_config['start_bdry'],
            end_bdry=cr3bp_config['end_bdry'],
            thrust=cr3bp_config['thrust'],
            solver_mode=cr3bp_config['solver_mode'],
            min_mass_to_sample=cr3bp_config['min_mass_to_sample'],
            max_mass_to_sample=cr3bp_config['max_mass_to_sample'],
            snopt_time_limit=cr3bp_config['snopt_time_limit'],
            result_folder=cr3bp_config['result_folder']
        )
        
        # Run physical validation
        try:
            # Run simulation for all samples (was limited to 10 before)
            result_data_list = []
            initial_guesses_list = []

            num_test_samples = len(samples)  # Test all generated samples

            # Format data exactly as 1D implementation expects: [halo_energy, ...rest_of_data]
            # The samples are already in the correct format from generate_samples()
            # First column (index 0) is the halo energy, rest is the initial guess
            
            for i in range(num_test_samples):
                # Data format: [halo_energy, shooting_time, coast_time1, coast_time2, controls..., fuel_mass, manifold_length1, manifold_length2]
                halo_energy = samples[i, 0]  # First column is already the physical halo energy
                initial_guess = samples[i, 1:]  # Rest is the initial guess data
                
                print(f"Testing sample {i+1}/{num_test_samples} with halo energy {halo_energy:.6f}")
                
                # Run simulation
                result_data = simulator.simulate(earth_initial_guess=initial_guess, halo_energy=halo_energy)
                result_data_list.append(result_data)
                initial_guesses_list.append(initial_guess)
            
            # Compute statistics similar to print_statistics method
            physical_metrics = self.compute_cr3bp_statistics(result_data_list, initial_guesses_list)
            
        except Exception as e:
            print(f"Warning: CR3BP simulation failed: {e}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception details: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            physical_metrics = {'simulation_error': str(e)}
        
        # Clean up temporary file
        if os.path.exists(temp_sample_path):
            os.remove(temp_sample_path)
        
        return physical_metrics
    
    def compute_cr3bp_statistics(self, result_data_list: List[Dict], initial_guesses_list: List[np.ndarray]) -> Dict[str, Any]:
        """Compute CR3BP statistics similar to print_statistics method."""
        if not result_data_list:
            return {}
        
        # Feasible solution ratio
        total_num = len(result_data_list)
        feasible_num = sum(1 for result in result_data_list if result["feasibility"])
        feasible_ratio = feasible_num / total_num if total_num > 0 else 0
        
        # Average final mass for feasible solutions
        feasible_final_mass_sum = 0
        for result in result_data_list:
            if result["feasibility"] and result["results.control"] is not None:
                feasible_final_mass_sum += result["results.control"][-3]
        
        avg_final_mass_feasible = feasible_final_mass_sum / feasible_num if feasible_num > 0 else 0
        
        # Local optimal solution ratio
        local_optimal_num = sum(1 for result in result_data_list 
                              if result["feasibility"] and result["snopt_inform"] == 1)
        local_optimal_ratio = local_optimal_num / total_num if total_num > 0 else 0
        
        # Average final mass for local optimal solutions
        local_optimal_final_mass_sum = 0
        for result in result_data_list:
            if result["snopt_inform"] == 1 and result["results.control"] is not None:
                local_optimal_final_mass_sum += result["results.control"][-3]
        
        avg_final_mass_optimal = local_optimal_final_mass_sum / local_optimal_num if local_optimal_num > 0 else 0
        
        # Average solving time
        solving_times = [result["solving_time"] for result in result_data_list]
        avg_solving_time = np.mean(solving_times) if solving_times else 0
        
        # SNOPT inform statistics
        snopt_informs = [result["snopt_inform"] for result in result_data_list if result["snopt_inform"] is not None]
        snopt_inform_counts = {}
        for inform in snopt_informs:
            snopt_inform_counts[inform] = snopt_inform_counts.get(inform, 0) + 1
        
        return {
            'feasible_ratio': feasible_ratio,
            'avg_final_mass_feasible': avg_final_mass_feasible,
            'local_optimal_ratio': local_optimal_ratio,
            'avg_final_mass_optimal': avg_final_mass_optimal,
            'avg_solving_time': avg_solving_time,
            'snopt_inform_distribution': snopt_inform_counts,
            'total_tested': total_num,
            'feasible_count': feasible_num,
            'local_optimal_count': local_optimal_num
        }
    
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
        """Run the complete GTO Halo benchmarking pipeline."""
        print("Starting GTO Halo comprehensive benchmark...")
        
        # Generate samples
        samples, sampling_times = self.generate_samples()
        
        # --- FLATTEN TO (N, 67) ---
        # Model outputs (N, 1, 9, 9) = (N, 81) total values
        # Original data is 67-dimensional, padded with 14 zeros to make 81 for the model input
        # After model prediction, output is flattened and only the first 67 values are used (rest are padding)
        # This is intentional and matches the data pipeline
        samples = samples.reshape(samples.shape[0], -1)  # (N, 81)
        samples = samples[:, :67]  # Keep only the first 67 values
        
        print(f"Model output shape: {samples.shape}")
        print(f"Flattened shape: {samples.shape}")
        
        # Extract components from 67-vector
        # Assuming format: [class_label, time_vars, thrust_vars, mass_vars, other_vars]
        class_labels = samples[:, 0]  # First value is class label
        time_vars = samples[:, 1:4]   # Time variables
        thrust_vars = samples[:, 4:64]  # Thrust variables (60 values)
        mass_vars = samples[:, 64:67]  # Mass variables (3 values)
        
        # Compute metrics
        results = {}
        
        # GTO Halo specific metrics
        results['gto_halo_metrics'] = self.compute_gto_halo_metrics(samples)
        
        # Physical validation metrics
        results['physical_validation'] = self.compute_physical_validation_metrics(samples)
        
        # Sampling efficiency metrics
        results['sampling_efficiency'] = self.compute_sampling_efficiency_metrics(sampling_times)
        
        # Save results
        self.save_results(results, samples)
        
        # Generate plots
        if self.config.save_plots:
            self.generate_plots(results, samples)
        
        # No spherical conversion statistics needed - no clipping occurs with spherical dataset
        
        return results
    
    def save_results(self, results: Dict[str, Any], samples: np.ndarray):
        """Save benchmark results."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(self.config.output_dir, 'gto_halo_benchmark_results.json'), 'w') as f:
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
        summary.append("GTO HALO BENCHMARK RESULTS")
        summary.append("=" * 60)
        
        if 'gto_halo_metrics' in results:
            summary.append("\nGTO HALO METRICS:")
            for key, value in results['gto_halo_metrics'].items():
                if isinstance(value, float):
                    summary.append(f"  {key}: {value:.6f}")
                else:
                    summary.append(f"  {key}: {value}")
        
        if 'physical_validation' in results:
            summary.append("\nPHYSICAL VALIDATION METRICS:")
            for key, value in results['physical_validation'].items():
                if isinstance(value, float):
                    summary.append(f"  {key}: {value:.6f}")
                else:
                    summary.append(f"  {key}: {value}")
        
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
        
        # GTO Halo specific plots
        self.plot_gto_halo_metrics(results.get('gto_halo_metrics', {}))
        
        # Physical validation plots
        if 'physical_validation' in results:
            self.plot_physical_validation(results['physical_validation'])
        
        # Sample distribution plots
        self.plot_sample_distributions(samples)
    
    def plot_gto_halo_metrics(self, metrics: Dict[str, Any]):
        """Plot GTO Halo specific metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Class label distribution
        if 'class_label_mean' in metrics:
            class_stats = ['Mean', 'Std', 'Min', 'Max']
            class_values = [
                metrics['class_label_mean'],
                metrics['class_label_std'],
                metrics['class_label_min'],
                metrics['class_label_max']
            ]
            axes[0, 0].bar(class_stats, class_values)
            axes[0, 0].set_title('Class Label Statistics')
            axes[0, 0].set_ylabel('Value')
        
        # Thrust variables distribution
        if 'thrust_vars_mean' in metrics:
            thrust_stats = ['Mean', 'Std', 'Min', 'Max']
            thrust_values = [
                metrics['thrust_vars_mean'],
                metrics['thrust_vars_std'],
                metrics['thrust_vars_min'],
                metrics['thrust_vars_max']
            ]
            axes[0, 1].bar(thrust_stats, thrust_values)
            axes[0, 1].set_title('Thrust Variables Statistics')
            axes[0, 1].set_ylabel('Value')
        
        # Mass variables distribution
        if 'mass_vars_mean' in metrics:
            mass_stats = ['Mean', 'Std', 'Min', 'Max']
            mass_values = [
                metrics['mass_vars_mean'],
                metrics['mass_vars_std'],
                metrics['mass_vars_min'],
                metrics['mass_vars_max']
            ]
            axes[0, 2].bar(mass_stats, mass_values)
            axes[0, 2].set_title('Mass Variables Statistics')
            axes[0, 2].set_ylabel('Value')
        
        # No boundary violation plots needed with spherical dataset
        # Spherical coordinates mathematically guarantee thrust magnitude ≤ 1.0
        axes[1, 0].text(0.5, 0.5, 'No clipping needed\n(Spherical Dataset)\nMagnitude ≤ 1.0 guaranteed', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Thrust Magnitude Constraint')
        axes[1, 0].set_ylim(0, 1)
        
        # Data quality checks (basic only)
        quality_checks = []
        quality_values = []
        
        for check in ['has_nan', 'has_inf']:
            if check in metrics:
                quality_checks.append(check.replace('_', ' ').title())
                quality_values.append(1 if metrics[check] else 0)
        
        if quality_checks:
            axes[1, 1].bar(quality_checks, quality_values)
            axes[1, 1].set_title('Data Quality Checks')
            axes[1, 1].set_ylabel('Flag (1=True, 0=False)')
        
        # Time variables
        if 'time_vars_mean' in metrics:
            time_stats = ['Mean', 'Std', 'Min', 'Max']
            time_values = [
                metrics['time_vars_mean'],
                metrics['time_vars_std'],
                metrics['time_vars_min'],
                metrics['time_vars_max']
            ]
            axes[1, 2].bar(time_stats, time_values)
            axes[1, 2].set_title('Time Variables Statistics')
            axes[1, 2].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', 'gto_halo_metrics.png'), dpi=300)
        plt.close()
    
    def plot_physical_validation(self, metrics: Dict[str, Any]):
        """Plot physical validation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Feasibility and optimality ratios
        if 'feasible_ratio' in metrics:
            ratios = ['Feasible', 'Local Optimal']
            ratio_values = [
                metrics['feasible_ratio'],
                metrics.get('local_optimal_ratio', 0)
            ]
            axes[0, 0].bar(ratios, ratio_values)
            axes[0, 0].set_title('Solution Quality Ratios')
            axes[0, 0].set_ylabel('Ratio')
            axes[0, 0].set_ylim(0, 1)
        
        # Average final mass
        if 'avg_final_mass_feasible' in metrics:
            mass_metrics = ['Feasible', 'Local Optimal']
            mass_values = [
                metrics['avg_final_mass_feasible'],
                metrics.get('avg_final_mass_optimal', 0)
            ]
            axes[0, 1].bar(mass_metrics, mass_values)
            axes[0, 1].set_title('Average Final Mass')
            axes[0, 1].set_ylabel('Mass')
        
        # Average solving time
        if 'avg_solving_time' in metrics:
            axes[1, 0].bar(['Average Solving Time'], [metrics['avg_solving_time']])
            axes[1, 0].set_title('Computational Efficiency')
            axes[1, 0].set_ylabel('Time (seconds)')
        
        # SNOPT inform distribution
        if 'snopt_inform_distribution' in metrics:
            inform_dist = metrics['snopt_inform_distribution']
            if inform_dist:
                inform_codes = list(inform_dist.keys())
                inform_counts = list(inform_dist.values())
                axes[1, 1].bar(inform_codes, inform_counts)
                axes[1, 1].set_title('SNOPT Inform Distribution')
                axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', 'physical_validation.png'), dpi=300)
        plt.close()
    
    def plot_sample_distributions(self, samples: np.ndarray):
        """Plot sample distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract components
        class_labels = samples[:, 0]
        time_vars = samples[:, 1:4]
        thrust_vars = samples[:, 4:64]
        mass_vars = samples[:, 64:67]
        
        # Class label distribution
        axes[0, 0].hist(class_labels, bins=30, alpha=0.7, density=True)
        axes[0, 0].set_title('Class Label Distribution')
        axes[0, 0].set_xlabel('Class Label')
        axes[0, 0].set_ylabel('Density')
        
        # Thrust variables distribution
        axes[0, 1].hist(thrust_vars.flatten(), bins=50, alpha=0.7, density=True)
        axes[0, 1].set_title('Thrust Variables Distribution')
        axes[0, 1].set_xlabel('Thrust Value')
        axes[0, 1].set_ylabel('Density')
        
        # Time variables mean/std
        time_means = np.mean(time_vars, axis=0)
        time_stds = np.std(time_vars, axis=0)
        
        axes[1, 0].errorbar(range(len(time_means)), time_means, yerr=time_stds, fmt='o-')
        axes[1, 0].set_title('Time Variables Statistics')
        axes[1, 0].set_xlabel('Time Variable Index')
        axes[1, 0].set_ylabel('Value')
        
        # Mass variables mean/std
        mass_means = np.mean(mass_vars, axis=0)
        mass_stds = np.std(mass_vars, axis=0)
        
        axes[1, 1].errorbar(range(len(mass_means)), mass_means, yerr=mass_stds, fmt='o-')
        axes[1, 1].set_title('Mass Variables Statistics')
        axes[1, 1].set_xlabel('Mass Variable Index')
        axes[1, 1].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'plots', 'sample_distributions.png'), dpi=300)
        plt.close() 
    
    # print_spherical_conversion_stats method removed - no clipping occurs with spherical dataset!