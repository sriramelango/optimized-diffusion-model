#!/usr/bin/env python3
"""
Fast Multithreaded GTO Halo Benchmarking Module with Comprehensive Dataset Creation

This version reduces thread initialization delays by:
1. Pre-initializing thread pool
2. Pre-loading libraries in each thread
3. Using thread-local storage for simulators
4. Reducing file I/O overhead

And creates comprehensive datasets that include:
1. Generated samples from diffusion model
2. Complete SNOPT simulation results and statistics
3. Full converged trajectory data (if feasible)
4. Physical orbital trajectory states and manifold arcs
5. Detailed metadata and processing information
6. Organized datasets by convergence status for easy analysis
7. Usage guides and examples for dataset utilization
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from multiprocessing import cpu_count

# Set matplotlib to non-interactive backend to avoid GUI issues in threads
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

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
    CR3BPEarthMissionWarmstartSimulatorBoundary = None
    GTO_HALO_DM_AVAILABLE = False


@dataclass
class GTOHaloBenchmarkConfig:
    """Configuration for fast multithreaded GTO Halo benchmarking."""
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
    
    # Multithreading config
    max_workers: int = None  # Will default to cpu_count() if None
    chunk_size: int = 1  # Number of samples per thread
    pre_warm_threads: bool = True  # Pre-initialize threads
    
    # Output config
    output_dir: str = "gto_halo_results"
    save_samples: bool = True
    save_plots: bool = True
    
    # Device config
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class ThreadLocalStorage:
    """Thread-local storage for pre-initialized simulators."""
    def __init__(self):
        self.local = threading.local()
    
    def get_simulator(self, thread_id: int, output_dir: str):
        """Get or create a simulator for this thread."""
        if not hasattr(self.local, f'simulator_{thread_id}'):
            # Pre-initialize simulator for this thread
            print(f"Thread {thread_id}: Pre-initializing CR3BP simulator...")
            
            # CR3BP simulation parameters (EXACTLY from GTO Halo DM simulator)
            cr3bp_config = {
                'seed': thread_id,
                'seed_step': 1,  # Process single sample
                'quiet_snopt': True,
                'number_of_segments': 20,  # Match GTO Halo DM
                'maximum_shooting_time': 40.0,  # Match GTO Halo DM
                'minimum_shooting_time': 0.0,  # Match GTO Halo DM
                'start_bdry': 6.48423370092,  # Match GTO Halo DM
                'end_bdry': 8.0,  # Match GTO Halo DM
                'thrust': 1.0,  # Match GTO Halo DM
                'solver_mode': "optimal",  # Match GTO Halo DM ("optimal" = optimal, "feasible" = feasible)
                'min_mass_to_sample': 408,  # Match GTO Halo DM
                'max_mass_to_sample': 470,  # Match GTO Halo DM
                'snopt_time_limit': 500.0,  # Match GTO Halo DM (original benchmarking)
                'result_folder': os.path.join(output_dir, 'cr3bp_results')
            }
            
            # Create a dummy simulator for pre-initialization
            dummy_simulator = CR3BPEarthMissionWarmstartSimulatorBoundary(
                seed=cr3bp_config['seed'],
                seed_step=cr3bp_config['seed_step'],
                quiet_snopt=cr3bp_config['quiet_snopt'],
                number_of_segments=cr3bp_config['number_of_segments'],
                maximum_shooting_time=cr3bp_config['maximum_shooting_time'],
                minimum_shooting_time=cr3bp_config['minimum_shooting_time'],
                sample_path=None,  # Will be set later
                start_bdry=cr3bp_config['start_bdry'],
                end_bdry=cr3bp_config['end_bdry'],
                thrust=cr3bp_config['thrust'],
                solver_mode=cr3bp_config['solver_mode'],
                min_mass_to_sample=cr3bp_config['min_mass_to_sample'],
                max_mass_to_sample=cr3bp_config['max_mass_to_sample'],
                snopt_time_limit=cr3bp_config['snopt_time_limit'],
                result_folder=cr3bp_config['result_folder']
            )
            
            setattr(self.local, f'simulator_{thread_id}', dummy_simulator)
            print(f"Thread {thread_id}: CR3BP simulator pre-initialized")
        
        return getattr(self.local, f'simulator_{thread_id}')


# Global thread-local storage
thread_storage = ThreadLocalStorage()


def process_single_sample_fast(args):
    """Process a single sample with pre-initialized thread resources and collect comprehensive data."""
    sample_idx, sample_data, halo_energy, output_dir, thread_id = args
    
    try:
        # Force immediate output with flush
        print(f"Thread {thread_id}: Starting sample {sample_idx} with halo energy {halo_energy:.6f}", flush=True)
        
        # Create directories for outputs
        temp_dir = os.path.join(output_dir, 'temp_samples')
        os.makedirs(temp_dir, exist_ok=True)
        cr3bp_results_dir = os.path.join(output_dir, 'cr3bp_results')
        os.makedirs(cr3bp_results_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f'sample_{sample_idx}.pkl')
        
        # CR3BP simulation parameters (EXACTLY from GTO Halo DM simulator)
        cr3bp_config = {
            'seed': sample_idx,
            'seed_step': 1,  # Process single sample
            'quiet_snopt': True,  # Suppress SNOPT output for cleaner telemetry
            'number_of_segments': 20,  # Match GTO Halo DM
            'maximum_shooting_time': 40.0,  # Match GTO Halo DM
            'minimum_shooting_time': 0.0,  # Match GTO Halo DM
            'start_bdry': 6.48423370092,  # Match GTO Halo DM
            'end_bdry': 8.0,  # Match GTO Halo DM
            'thrust': 1.0,  # Match GTO Halo DM
            'solver_mode': "optimal",  # Match GTO Halo DM ("optimal" = optimal, "feasible" = feasible)
            'min_mass_to_sample': 408,  # Match GTO Halo DM
            'max_mass_to_sample': 470,  # Match GTO Halo DM
            'snopt_time_limit': 500.0,  # Match GTO Halo DM (original benchmarking)
            'result_folder': cr3bp_results_dir
        }
        
        # Save single sample in the format expected by CR3BP simulator
        # Format: [halo_energy, trajectory_params] where trajectory_params is 66-dimensional
        sample_for_cr3bp = np.array([halo_energy] + list(sample_data))
        with open(temp_file, 'wb') as f:
            pickle.dump(sample_for_cr3bp.reshape(1, -1), f)
        
        # Create fresh simulator for this sample
        simulator = CR3BPEarthMissionWarmstartSimulatorBoundary(
            seed=cr3bp_config['seed'],
            seed_step=cr3bp_config['seed_step'],
            quiet_snopt=cr3bp_config['quiet_snopt'],
            number_of_segments=cr3bp_config['number_of_segments'],
            maximum_shooting_time=cr3bp_config['maximum_shooting_time'],
            minimum_shooting_time=cr3bp_config['minimum_shooting_time'],
            sample_path=temp_file,
            start_bdry=cr3bp_config['start_bdry'],
            end_bdry=cr3bp_config['end_bdry'],
            thrust=cr3bp_config['thrust'],
            solver_mode=cr3bp_config['solver_mode'],
            min_mass_to_sample=cr3bp_config['min_mass_to_sample'],
            max_mass_to_sample=cr3bp_config['max_mass_to_sample'],
            snopt_time_limit=cr3bp_config['snopt_time_limit'],
            result_folder=cr3bp_config['result_folder']
        )
        
        # Run simulation
        print(f"📊 TELEMETRY: Thread {thread_id} - Sample {sample_idx} - SNOPT STARTING", flush=True)
        result_data = simulator.simulate(earth_initial_guess=sample_data, halo_energy=halo_energy)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Create comprehensive dataset entry
        dataset_entry = {
            # Sample identification
            'sample_idx': sample_idx,
            'thread_id': thread_id,
            
            # Input data
            'generated_sample': {
                'halo_energy': halo_energy,
                'trajectory_params': sample_data.tolist(),  # Convert to list for JSON serialization
                'full_sample_vector': sample_for_cr3bp.tolist()
            },
            
            # SNOPT simulation parameters
            'simulation_config': cr3bp_config,
            
            # SNOPT results and statistics
            'snopt_results': {},
            
            # Converged trajectory data (if feasible)
            'converged_trajectory': {},
            
            # Physical trajectory states (if feasible)
            'physical_trajectories': {},
            
            # Analysis metadata
            'processing_metadata': {
                'timestamp': time.time(),
                'convergence_status': "UNKNOWN"
            }
        }
        
        # Extract and organize SNOPT results
        if result_data and isinstance(result_data, dict):
            print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Result data keys: {list(result_data.keys())}", flush=True)
            
            # Core SNOPT statistics
            dataset_entry['snopt_results'] = {
                'feasibility': result_data.get('feasibility', False),
                'snopt_inform': result_data.get('snopt_inform', None),
                'solving_time': result_data.get('solving_time', None),
                'thrust': result_data.get('thrust', None),
                'cost_alpha': result_data.get('cost_alpha', None),
                'has_snopt_control_evaluations': result_data.get('snopt_control_evaluations', None) is not None
            }
            
            # If feasible, extract complete trajectory data
            if result_data.get('feasibility', False):
                if 'results.control' in result_data and result_data['results.control'] is not None:
                    control_vector = result_data['results.control']
                    
                    # Store the complete control vector
                    dataset_entry['converged_trajectory']['control_vector'] = control_vector.tolist()
                    
                    # Extract key trajectory parameters
                    dataset_entry['converged_trajectory']['trajectory_parameters'] = {
                        'shooting_time': float(control_vector[0]),
                        'initial_coast_time': float(control_vector[1]),
                        'final_coast_time': float(control_vector[2]),
                        'final_fuel_mass': float(control_vector[-3]),
                        'halo_period': float(control_vector[-2]),
                        'manifold_length': float(control_vector[-1])
                    }
                    
                    # Extract control segments (thrust directions for each segment)
                    control_segments = []
                    for i in range(20):  # 20 segments
                        segment_start = 3 + i * 3
                        control_segments.append({
                            'segment_id': i,
                            'alpha': float(control_vector[segment_start]),      # ux component
                            'beta': float(control_vector[segment_start + 1]),   # uy component  
                            'thrust_magnitude': float(control_vector[segment_start + 2])  # uz component
                        })
                    
                    dataset_entry['converged_trajectory']['control_segments'] = control_segments
                
                # Extract physical trajectory states if available
                if 'results.states' in result_data and result_data['results.states'] is not None:
                    converged_states = result_data['results.states']
                    dataset_entry['physical_trajectories']['converged_states'] = converged_states.tolist()
                    dataset_entry['physical_trajectories']['converged_states_shape'] = list(converged_states.shape)
                    print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Captured converged trajectory states: {converged_states.shape}", flush=True)
                
                if 'results_DM.states' in result_data and result_data['results_DM.states'] is not None:
                    predicted_states = result_data['results_DM.states']
                    dataset_entry['physical_trajectories']['predicted_states'] = predicted_states.tolist()
                    dataset_entry['physical_trajectories']['predicted_states_shape'] = list(predicted_states.shape)
                    print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Captured predicted trajectory states: {predicted_states.shape}", flush=True)
                
                # Extract manifold arcs if available
                if 'manifold_arc' in result_data and result_data['manifold_arc'] is not None:
                    converged_manifold = result_data['manifold_arc']
                    dataset_entry['physical_trajectories']['converged_manifold'] = converged_manifold.tolist()
                    dataset_entry['physical_trajectories']['converged_manifold_shape'] = list(converged_manifold.shape)
                
                if 'manifold_arc_DM' in result_data and result_data['manifold_arc_DM'] is not None:
                    predicted_manifold = result_data['manifold_arc_DM']
                    dataset_entry['physical_trajectories']['predicted_manifold'] = predicted_manifold.tolist()
                    dataset_entry['physical_trajectories']['predicted_manifold_shape'] = list(predicted_manifold.shape)
                
                # Extract GTO spiral trajectory states if available
                if 'gto_states' in result_data and result_data['gto_states'] is not None:
                    gto_states = result_data['gto_states']
                    dataset_entry['physical_trajectories']['gto_states'] = gto_states.tolist()
                    dataset_entry['physical_trajectories']['gto_states_shape'] = list(gto_states.shape)
                    print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Captured GTO spiral states: {gto_states.shape}", flush=True)
                
                # Store SNOPT control evaluations if available
                if result_data.get('snopt_control_evaluations', None) is not None:
                    snopt_evals = result_data['snopt_control_evaluations']
                    dataset_entry['snopt_results']['snopt_control_evaluations'] = {
                        'shape': list(snopt_evals.shape),
                        'data': snopt_evals.tolist(),  # Full evaluation history
                        'num_evaluations': snopt_evals.shape[0] if len(snopt_evals.shape) > 0 else 0
                    }
                
                dataset_entry['processing_metadata']['convergence_status'] = "FEASIBLE"
                
                # Check if solution is locally optimal
                if result_data.get('snopt_inform', 0) == 1:
                    dataset_entry['processing_metadata']['convergence_status'] = "LOCALLY_OPTIMAL"
            else:
                dataset_entry['processing_metadata']['convergence_status'] = "INFEASIBLE"
                
                # Even for infeasible solutions, store what we can
                if 'error' in result_data:
                    dataset_entry['snopt_results']['error_message'] = str(result_data['error'])
                
                # Try to capture predicted trajectory states even if SNOPT failed
                if 'results_DM.states' in result_data and result_data['results_DM.states'] is not None:
                    predicted_states = result_data['results_DM.states']
                    dataset_entry['physical_trajectories']['predicted_states'] = predicted_states.tolist()
                    dataset_entry['physical_trajectories']['predicted_states_shape'] = list(predicted_states.shape)
                    print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Captured predicted trajectory states (infeasible case): {predicted_states.shape}", flush=True)
                
                if 'manifold_arc_DM' in result_data and result_data['manifold_arc_DM'] is not None:
                    predicted_manifold = result_data['manifold_arc_DM']
                    dataset_entry['physical_trajectories']['predicted_manifold'] = predicted_manifold.tolist()
                    dataset_entry['physical_trajectories']['predicted_manifold_shape'] = list(predicted_manifold.shape)
                
                # Extract GTO spiral trajectory states if available (even for infeasible cases)  
                if 'gto_states' in result_data and result_data['gto_states'] is not None:
                    gto_states = result_data['gto_states']
                    dataset_entry['physical_trajectories']['gto_states'] = gto_states.tolist()
                    dataset_entry['physical_trajectories']['gto_states_shape'] = list(gto_states.shape)
                    print(f"DEBUG: Thread {thread_id} - Sample {sample_idx} - Captured GTO spiral states (infeasible case): {gto_states.shape}", flush=True)
        else:
            dataset_entry['processing_metadata']['convergence_status'] = "ERROR"
            if result_data:
                dataset_entry['snopt_results']['raw_result'] = str(result_data)
        
        print(f"📊 TELEMETRY: Thread {thread_id} - Sample {sample_idx} - SNOPT COMPLETED - Result: {dataset_entry['processing_metadata']['convergence_status']}", flush=True)
        
        return dataset_entry
        
    except Exception as e:
        print(f"Thread {thread_id}: Error processing sample {sample_idx}: {e}", flush=True)
        
        # Return error entry for dataset
        return {
            'sample_idx': sample_idx,
            'thread_id': thread_id,
            'generated_sample': {
                'halo_energy': halo_energy,
                'trajectory_params': sample_data.tolist() if hasattr(sample_data, 'tolist') else list(sample_data),
                'full_sample_vector': None
            },
            'simulation_config': {},
            'snopt_results': {
                'feasibility': False,
                'error_message': str(e)
            },
            'converged_trajectory': {},
            'processing_metadata': {
                'timestamp': time.time(),
                'convergence_status': "PROCESSING_ERROR"
            }
        }


class GTOHaloBenchmarker:
    """Fast multithreaded GTO Halo specific benchmarking for diffusion models."""
    
    def __init__(self, config: GTOHaloBenchmarkConfig):
        """Initialize the benchmarker."""
        self.config = config
        
        # Set up multithreading
        if self.config.max_workers is None:
            self.config.max_workers = cpu_count()
        
        print(f"✓ Using {self.config.max_workers} CPU cores for parallel processing")
        
        # Initialize model and data
        self.model = None
        self.sde = None
        self.score_fn = None
        self.inverse_scaler = None
        self.reference_data = None
        
        # Load model and data
        self.load_model()
        self.load_reference_data()
        
        # Pre-warm threads if enabled
        if self.config.pre_warm_threads:
            self.pre_warm_threads()
    
    def pre_warm_threads(self):
        """Pre-initialize threads to reduce startup delays."""
        print("Pre-warming threads to reduce initialization delays...")
        
        # Create a dummy task to warm up each thread
        def warm_up_thread(thread_id):
            print(f"Warming up thread {thread_id}...")
            # Just import the library to warm up the thread
            import sys
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../GTO_Halo_DM/data_generation_scripts')))
            from cr3bp_earth_mission_simulator_boundary_diffusion_warmstart import CR3BPEarthMissionWarmstartSimulatorBoundary
            return f"Thread {thread_id} warmed up"
        
        # Warm up all threads
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(warm_up_thread, i) for i in range(self.config.max_workers)]
            for future in as_completed(futures):
                result = future.result()
                print(f"✓ {result}")
        
        print("✓ All threads pre-warmed")
    
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
            self.score_model = mutils.create_model(self.cfg).to(self.config.device)
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
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
            
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
            self.cfg, self.sde, sampling_shape, 1e-5, self.config.device
        )
    
    def load_reference_data(self):
        """Load reference data for comparison."""
        if os.path.exists(self.config.data_path):
            with open(self.config.data_path, 'rb') as f:
                self.reference_data = pickle.load(f)
            print(f"✓ Reference data loaded: {self.reference_data.shape}")
        else:
            print(f"⚠️  Reference data not found: {self.config.data_path}")
            self.reference_data = None
    
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
            class_labels = torch.rand(batch_size, 1, device=self.config.device)
            
            # Store input class labels for later use (like original GTO Halo implementation)
            input_class_labels.append(class_labels.cpu().numpy())
            
            # Print input class labels for verification
            print(f"Batch {i+1}: Input class labels: {class_labels.cpu().numpy().flatten()}")
            
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
        
        # Store input class labels as instance variable for later use
        self.input_class_labels = input_class_labels

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
        
        # Print unnormalized halo energies for verification
        print(f"Unnormalized halo energies: {halo_energies}")
        
        # Combine halo energies with model outputs (exactly as in 1D implementation)
        samples = np.column_stack((halo_energies, model_outputs))

        print(f"Model output shape: {samples.shape}")
        print(f"Flattened shape: {samples.shape}")

        return samples, sampling_times
    
    def compute_physical_validation_metrics(self, samples: np.ndarray) -> Dict[str, Any]:
        """Compute physical validation metrics using CR3BP simulator with fast multithreading."""
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
        
        print("Computing physical validation metrics using CR3BP simulator with fast multithreading...")
        
        # Prepare samples for parallel processing
        num_samples = len(samples)
        sample_args = []
        
        # Create arguments for each sample with thread ID
        for i in range(num_samples):
            halo_energy = samples[i, 0]  # First column is the physical halo energy (unnormalized input class label)
            initial_guess = samples[i, 1:]  # Rest is the initial guess data
            thread_id = i % self.config.max_workers  # Distribute across threads
            sample_args.append((i, initial_guess, halo_energy, self.config.output_dir, thread_id))
        
        print(f"Processing {num_samples} samples using {self.config.max_workers} pre-warmed threads...")
        
        # Process samples in parallel with pre-warmed threads
        all_results = []
        sample_status = {}  # Track status of each sample
        active_samples = set()  # Track currently running samples
        
        print(f"Starting parallel processing of {num_samples} samples with {self.config.max_workers} threads...", flush=True)
        print(f"📊 TELEMETRY: Initializing {num_samples} samples...", flush=True)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all samples
            future_to_sample = {executor.submit(process_single_sample_fast, args): args[0] for args in sample_args}
            
            # Initialize status for all samples
            for sample_idx in range(num_samples):
                sample_status[sample_idx] = "QUEUED"
            
            # Track active samples
            active_samples = set(range(num_samples))
            
            # Print initial status
            print(f"📊 TELEMETRY: All {num_samples} samples submitted to thread pool", flush=True)
            print(f"📊 TELEMETRY: Active samples: {sorted(active_samples)}", flush=True)
            
            # Collect results as they complete
            start_time = time.time()
            last_status_update = start_time
            
            for future in as_completed(future_to_sample):
                sample_idx = future_to_sample[future]
                result = future.result()
                all_results.append(result)
                
                # Periodic status update every 10 seconds
                current_time = time.time()
                if current_time - last_status_update > 10:
                    print(f"📊 TELEMETRY: STATUS UPDATE at {current_time - start_time:.1f}s", flush=True)
                    print(f"📊 TELEMETRY: Active samples: {sorted(active_samples)}", flush=True)
                    print(f"📊 TELEMETRY: Completed: {len(all_results)}/{num_samples}", flush=True)
                    last_status_update = current_time
                
                # Update status
                sample_status[sample_idx] = "COMPLETED"
                active_samples.discard(sample_idx)
                
                # Extract convergence info
                convergence_status = "UNKNOWN"
                if 'result_data' in result and result['result_data']:
                    if isinstance(result['result_data'], dict):
                        if 'feasibility' in result['result_data']:
                            convergence_status = "FEASIBLE" if result['result_data']['feasibility'] else "INFEASIBLE"
                        elif 'error' in result['result_data']:
                            convergence_status = f"ERROR: {result['result_data']['error']}"
                    else:
                        convergence_status = "PROCESSED"
                
                # Print detailed progress
                completed = len(all_results)
                print(f"📊 TELEMETRY: Sample {sample_idx} COMPLETED - Status: {convergence_status}", flush=True)
                print(f"📊 TELEMETRY: Progress: {completed}/{num_samples} samples ({completed/num_samples*100:.1f}%)", flush=True)
                print(f"📊 TELEMETRY: Remaining active samples: {sorted(active_samples)}", flush=True)
                
                # Print summary of all samples
                print(f"📊 TELEMETRY: Sample Status Summary:", flush=True)
                for idx in range(num_samples):
                    status = sample_status.get(idx, "UNKNOWN")
                    if idx in active_samples:
                        status = "RUNNING"
                    print(f"   Sample {idx}: {status}", flush=True)
                print("-" * 50, flush=True)
        
        # Sort results by sample index to maintain order
        all_results.sort(key=lambda x: x['sample_idx'])
        
        # Save the comprehensive dataset
        self.save_comprehensive_dataset(all_results)
        
        # Extract result data and initial guesses for legacy statistics computation
        result_data_list = []
        initial_guesses_list = []
        
        for result in all_results:
            # Extract legacy format for compatibility with existing statistics
            if 'snopt_results' in result:
                legacy_result = {
                    'feasibility': result['snopt_results'].get('feasibility', False),
                    'snopt_inform': result['snopt_results'].get('snopt_inform', None),
                    'solving_time': result['snopt_results'].get('solving_time', None),
                    'results.control': None
                }
                
                # Extract control vector if available
                if 'converged_trajectory' in result and 'control_vector' in result['converged_trajectory']:
                    legacy_result['results.control'] = np.array(result['converged_trajectory']['control_vector'])
                
                result_data_list.append(legacy_result)
            else:
                # Handle old format (backward compatibility)
                result_data_list.append(result.get('result_data', {}))
            
            # Extract initial guess
            if 'generated_sample' in result:
                initial_guesses_list.append(np.array(result['generated_sample']['trajectory_params']))
            else:
                # Handle old format (backward compatibility)
                initial_guesses_list.append(result.get('initial_guess', []))
        
        # Compute statistics
        physical_metrics = self.compute_cr3bp_statistics(result_data_list, initial_guesses_list)
        
        # Add dataset information to metrics
        physical_metrics['comprehensive_dataset_saved'] = True
        physical_metrics['dataset_entries'] = len(all_results)
        
        return physical_metrics
    
    def save_comprehensive_dataset(self, all_results: List[Dict]):
        """Save the comprehensive dataset with all sample data, SNOPT results, and trajectory information."""
        print("Saving comprehensive dataset...")
        
        # Create dataset directory
        dataset_dir = os.path.join(self.config.output_dir, 'comprehensive_dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Organize data by convergence status for easier analysis
        dataset_by_status = {
            'LOCALLY_OPTIMAL': [],
            'FEASIBLE': [],
            'INFEASIBLE': [],
            'PROCESSING_ERROR': [],
            'UNKNOWN': []
        }
        
        for result in all_results:
            status = result.get('processing_metadata', {}).get('convergence_status', 'UNKNOWN')
            dataset_by_status[status].append(result)
        
        # Save complete dataset as pickle for easy loading in Python
        complete_dataset_path = os.path.join(dataset_dir, 'complete_dataset.pkl')
        with open(complete_dataset_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"✓ Complete dataset saved to {complete_dataset_path}")
        
        # Save complete dataset as JSON for readability
        complete_dataset_json_path = os.path.join(dataset_dir, 'complete_dataset.json')
        with open(complete_dataset_json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"✓ Complete dataset (JSON) saved to {complete_dataset_json_path}")
        
        # Save datasets by convergence status
        for status, data in dataset_by_status.items():
            if data:  # Only save if there's data for this status
                status_path = os.path.join(dataset_dir, f'dataset_{status.lower()}.pkl')
                with open(status_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"✓ {status} dataset saved to {status_path} ({len(data)} samples)")
        
        # Save feasible trajectories separately for quick access
        feasible_trajectories = []
        for result in all_results:
            if (result.get('processing_metadata', {}).get('convergence_status') in ['FEASIBLE', 'LOCALLY_OPTIMAL'] 
                and 'converged_trajectory' in result 
                and result['converged_trajectory']):
                feasible_trajectories.append({
                    'sample_idx': result['sample_idx'],
                    'halo_energy': result['generated_sample']['halo_energy'],
                    'convergence_status': result['processing_metadata']['convergence_status'],
                    'snopt_inform': result['snopt_results']['snopt_inform'],
                    'solving_time': result['snopt_results']['solving_time'],
                    'trajectory_parameters': result['converged_trajectory']['trajectory_parameters'],
                    'control_vector': result['converged_trajectory']['control_vector'],
                    'control_segments': result['converged_trajectory']['control_segments']
                })
        
        if feasible_trajectories:
            feasible_traj_path = os.path.join(dataset_dir, 'feasible_trajectories.pkl')
            with open(feasible_traj_path, 'wb') as f:
                pickle.dump(feasible_trajectories, f)
            print(f"✓ Feasible trajectories saved to {feasible_traj_path} ({len(feasible_trajectories)} trajectories)")
        
        # Create dataset summary
        summary = {
            'total_samples': len(all_results),
            'by_status': {status: len(data) for status, data in dataset_by_status.items()},
            'feasible_trajectories': len(feasible_trajectories),
            'dataset_structure': {
                'sample_identification': ['sample_idx', 'thread_id'],
                'input_data': ['generated_sample'],
                'simulation_config': ['simulation_config'],
                'snopt_results': ['snopt_results'],
                'converged_trajectory': ['converged_trajectory'],
                'physical_trajectories': ['physical_trajectories'],
                'processing_metadata': ['processing_metadata']
            },
            'files_saved': [
                'complete_dataset.pkl',
                'complete_dataset.json',
                'feasible_trajectories.pkl'
            ] + [f'dataset_{status.lower()}.pkl' for status, data in dataset_by_status.items() if data]
        }
        
        # Save dataset summary
        summary_path = os.path.join(dataset_dir, 'dataset_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Dataset summary saved to {summary_path}")
        
        # Create usage guide
        usage_guide = """
# GTO Halo Comprehensive Dataset Usage Guide

This directory contains a comprehensive dataset with all generated samples, 
SNOPT simulation results, converged trajectory data, and physical orbital trajectories.

## Files:

1. **complete_dataset.pkl**: Full dataset in Python pickle format
   - Load with: `data = pickle.load(open('complete_dataset.pkl', 'rb'))`
   
2. **complete_dataset.json**: Full dataset in JSON format (human-readable)

3. **feasible_trajectories.pkl**: Only the feasible/optimal trajectory data
   - Quick access to successful convergence results
   
4. **dataset_[status].pkl**: Data organized by convergence status
   - dataset_locally_optimal.pkl: SNOPT inform = 1 solutions
   - dataset_feasible.pkl: Feasible but not necessarily optimal
   - dataset_infeasible.pkl: Failed convergence
   - dataset_processing_error.pkl: Processing errors

5. **dataset_summary.json**: Overview of dataset contents

## Data Structure:

Each entry contains:
- **sample_idx**: Sample identifier
- **generated_sample**: Original diffusion model output
  - halo_energy: Physical halo energy parameter
  - trajectory_params: 66-dimensional trajectory parameters  
  - full_sample_vector: Complete input vector for SNOPT
- **simulation_config**: SNOPT simulation parameters used
- **snopt_results**: SNOPT convergence statistics
  - feasibility: Boolean success flag
  - snopt_inform: SNOPT convergence code
  - solving_time: Computation time in seconds
  - snopt_control_evaluations: Full SNOPT evaluation history (if available)
- **converged_trajectory**: Complete trajectory data (if feasible)
  - control_vector: Full SNOPT control solution
  - trajectory_parameters: Key trajectory parameters
  - control_segments: Thrust control for each segment
- **physical_trajectories**: Physical orbital state trajectories
  - converged_states: SNOPT converged trajectory states [x,y,z,vx,vy,vz,...]
  - predicted_states: Predicted trajectory states from initial guess
  - converged_manifold: Converged manifold arc states
  - predicted_manifold: Predicted manifold arc states
  - gto_states: GTO spiral trajectory states
- **processing_metadata**: Timestamps and convergence status

## Example Usage:

```python
import pickle
import numpy as np

# Load complete dataset
with open('complete_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Filter for locally optimal solutions
optimal_solutions = [entry for entry in data 
                    if entry['processing_metadata']['convergence_status'] == 'LOCALLY_OPTIMAL']

# Extract feasible trajectories
feasible_controls = [entry['converged_trajectory']['control_vector'] 
                    for entry in data 
                    if entry['snopt_results']['feasibility']]

# Analyze solving times
solving_times = [entry['snopt_results']['solving_time'] 
                for entry in data 
                if entry['snopt_results']['solving_time'] is not None]

# Extract physical trajectory states for orbital path analysis
predicted_trajectories = [np.array(entry['physical_trajectories']['predicted_states'])
                         for entry in data 
                         if 'physical_trajectories' in entry and 
                            entry['physical_trajectories'].get('predicted_states') is not None]

converged_trajectories = [np.array(entry['physical_trajectories']['converged_states'])
                        for entry in data 
                        if 'physical_trajectories' in entry and 
                           entry['physical_trajectories'].get('converged_states') is not None]

# Extract GTO spiral trajectory states
gto_trajectories = [np.array(entry['physical_trajectories']['gto_states'])
                   for entry in data 
                   if 'physical_trajectories' in entry and 
                      entry['physical_trajectories'].get('gto_states') is not None]

# Extract manifold arcs for trajectory analysis  
converged_manifolds = [np.array(entry['physical_trajectories']['converged_manifold'])
                      for entry in data 
                      if 'physical_trajectories' in entry and 
                         entry['physical_trajectories'].get('converged_manifold') is not None]
```
"""
        
        usage_guide_path = os.path.join(dataset_dir, 'README.md')
        with open(usage_guide_path, 'w') as f:
            f.write(usage_guide)
        print(f"✓ Usage guide saved to {usage_guide_path}")
        
        print(f"✓ Comprehensive dataset completed:")
        print(f"   - Total samples: {len(all_results)}")
        print(f"   - Locally optimal: {len(dataset_by_status['LOCALLY_OPTIMAL'])}")
        print(f"   - Feasible: {len(dataset_by_status['FEASIBLE'])}")
        print(f"   - Infeasible: {len(dataset_by_status['INFEASIBLE'])}")
        print(f"   - Processing errors: {len(dataset_by_status['PROCESSING_ERROR'])}")
        print(f"   - Feasible trajectories: {len(feasible_trajectories)}")
    
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
        
        # Data quality checks
        metrics['has_nan'] = bool(np.any(np.isnan(samples)))
        metrics['has_inf'] = bool(np.any(np.isinf(samples)))
        
        # Spherical coordinates guarantee thrust magnitude ≤ 1.0 mathematically
        # No clipping or violations can occur during unnormalization
        
        return metrics
    
    def compute_cr3bp_statistics(self, result_data_list: List[Dict], initial_guesses_list: List[np.ndarray]) -> Dict[str, Any]:
        """Compute CR3BP simulation statistics."""
        if not result_data_list:
            return {'error': 'No CR3BP results available'}
        
        # Extract metrics
        feasible_count = sum(1 for result in result_data_list if result.get('feasibility', False))
        total_count = len(result_data_list)
        feasible_ratio = feasible_count / total_count if total_count > 0 else 0
        
        # Local optimal solutions (SNOPT inform = 1)
        local_optimal_count = sum(1 for result in result_data_list 
                                if result.get('feasibility', False) and result.get('snopt_inform', 0) == 1)
        local_optimal_ratio = local_optimal_count / total_count if total_count > 0 else 0
        
        # Final mass analysis
        final_masses_feasible = [result.get('final_mass', 0) for result in result_data_list 
                               if result.get('feasibility', False)]
        final_masses_optimal = [result.get('final_mass', 0) for result in result_data_list 
                              if result.get('feasibility', False) and result.get('snopt_inform', 0) == 1]
        
        avg_final_mass_feasible = np.mean(final_masses_feasible) if final_masses_feasible else 0
        avg_final_mass_optimal = np.mean(final_masses_optimal) if final_masses_optimal else 0
        
        # Solving time analysis
        solving_times = [result.get('solving_time', 0) for result in result_data_list 
                        if result.get('feasibility', False)]
        avg_solving_time = np.mean(solving_times) if solving_times else 0
        
        # SNOPT inform distribution
        snopt_informs = [result.get('snopt_inform', 0) for result in result_data_list]
        snopt_inform_distribution = {}
        for inform in snopt_informs:
            snopt_inform_distribution[inform] = snopt_inform_distribution.get(inform, 0) + 1
        
        return {
            'feasible_ratio': feasible_ratio,
            'avg_final_mass_feasible': avg_final_mass_feasible,
            'local_optimal_ratio': local_optimal_ratio,
            'avg_final_mass_optimal': avg_final_mass_optimal,
            'avg_solving_time': avg_solving_time,
            'snopt_inform_distribution': snopt_inform_distribution,
            'total_tested': total_count,
            'feasible_count': feasible_count,
            'local_optimal_count': local_optimal_count
        }
    
    def compute_sampling_efficiency_metrics(self, sampling_times: List[float]) -> Dict[str, float]:
        """Compute sampling efficiency metrics."""
        if not sampling_times:
            return {}
        
        total_sampling_time = sum(sampling_times)
        avg_sampling_time = np.mean(sampling_times)
        sampling_time_std = np.std(sampling_times)
        samples_per_second = self.config.num_samples / total_sampling_time if total_sampling_time > 0 else 0
        
        return {
            'total_sampling_time': total_sampling_time,
            'average_sampling_time_per_sample': avg_sampling_time,
            'sampling_time_std': sampling_time_std,
            'samples_per_second': samples_per_second,
            'min_sampling_time': min(sampling_times),
            'max_sampling_time': max(sampling_times)
        }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark."""
        print("=" * 60)
        print("RUNNING GTO HALO BENCHMARK (FAST MULTITHREADED)")
        print("=" * 60)
        
        # Generate samples
        samples, sampling_times = self.generate_samples()
        
        # Compute metrics
        gto_halo_metrics = self.compute_gto_halo_metrics(samples)
        physical_validation_metrics = self.compute_physical_validation_metrics(samples)
        sampling_efficiency_metrics = self.compute_sampling_efficiency_metrics(sampling_times)
        
        # Combine results
        results = {
            'gto_halo_metrics': gto_halo_metrics,
            'physical_validation_metrics': physical_validation_metrics,
            'sampling_efficiency_metrics': sampling_efficiency_metrics,
            'num_samples': len(samples),
            'multithreading_config': {
                'max_workers': self.config.max_workers,
                'chunk_size': self.config.chunk_size,
                'pre_warm_threads': self.config.pre_warm_threads
            }
        }
        
        # Save results
        self.save_results(results, samples)
        
        return results
    
    def save_results(self, results: Dict[str, Any], samples: np.ndarray):
        """Save benchmark results."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save samples
        if self.config.save_samples:
            samples_path = os.path.join(self.config.output_dir, 'samples.npy')
            np.save(samples_path, samples)
            print(f"✓ Samples saved to {samples_path}")
        
        # Save results as JSON
        results_path = os.path.join(self.config.output_dir, 'gto_halo_benchmark_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results saved to {results_path}")
        
        # Save summary
        self.save_summary(results)
        
        # Generate plots
        if self.config.save_plots:
            self.generate_plots(results, samples)
    
    def save_summary(self, results: Dict[str, Any]):
        """Save human-readable summary."""
        summary_path = os.path.join(self.config.output_dir, 'summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("GTO HALO BENCHMARK RESULTS (FAST MULTITHREADED)\n")
            f.write("=" * 60 + "\n\n")
            
            # GTO Halo metrics
            gto_metrics = results.get('gto_halo_metrics', {})
            if gto_metrics:
                f.write("GTO HALO METRICS:\n")
                for key, value in gto_metrics.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Physical validation metrics
            physical_metrics = results.get('physical_validation_metrics', {})
            if physical_metrics:
                f.write("PHYSICAL VALIDATION METRICS:\n")
                for key, value in physical_metrics.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                # Add dataset information if available
                if physical_metrics.get('comprehensive_dataset_saved', False):
                    f.write("COMPREHENSIVE DATASET:\n")
                    f.write(f"  dataset_saved: {physical_metrics.get('comprehensive_dataset_saved', False)}\n")
                    f.write(f"  total_entries: {physical_metrics.get('dataset_entries', 0)}\n")
                    f.write(f"  location: {self.config.output_dir}/comprehensive_dataset/\n")
                    f.write("  files: complete_dataset.pkl, feasible_trajectories.pkl, dataset_summary.json\n")
                    f.write("  usage_guide: comprehensive_dataset/README.md\n")
                    f.write("\n")
            
            # Sampling efficiency metrics
            sampling_metrics = results.get('sampling_efficiency_metrics', {})
            if sampling_metrics:
                f.write("SAMPLING EFFICIENCY:\n")
                for key, value in sampling_metrics.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Multithreading info
            multithreading_config = results.get('multithreading_config', {})
            if multithreading_config:
                f.write("MULTITHREADING CONFIG:\n")
                for key, value in multithreading_config.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Add important file locations
            f.write("OUTPUT FILES:\n")
            f.write(f"  benchmark_results: gto_halo_benchmark_results.json\n")
            f.write(f"  generated_samples: samples.npy\n")
            f.write(f"  comprehensive_dataset: comprehensive_dataset/\n")
            f.write(f"  cr3bp_simulation_results: cr3bp_results/\n")
            f.write(f"  summary: summary.txt\n")
        
        print(f"✓ Summary saved to {summary_path}")
    
    def generate_plots(self, results: Dict[str, Any], samples: np.ndarray):
        """Generate visualization plots."""
        # This can be implemented similar to the original benchmarking script
        # For now, we'll skip plotting to focus on the multithreading functionality
        print("✓ Plot generation skipped (focusing on multithreading)")


def main():
    """Main function to run the fast multithreaded benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast Multithreaded GTO Halo Benchmarking')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--data_path', type=str, required=True, help='Path to reference data')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for sampling')
    parser.add_argument('--max_workers', type=int, default=None, help='Number of worker threads (default: CPU count)')
    parser.add_argument('--chunk_size', type=int, default=1, help='Number of samples per thread')
    parser.add_argument('--pre_warm_threads', action='store_true', default=True, help='Pre-warm threads to reduce initialization delays')
    parser.add_argument('--output_dir', type=str, default='gto_halo_fast_results', help='Output directory')
    parser.add_argument('--enable_physical_validation', action='store_true', default=True, help='Enable physical validation')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create config
    config = GTOHaloBenchmarkConfig(
        model_path=args.model_path,
        config_path=args.model_path,  # Use model_path as config_path
        data_path=args.data_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        pre_warm_threads=args.pre_warm_threads,
        output_dir=args.output_dir,
        enable_physical_validation=args.enable_physical_validation
    )
    
    # Run benchmark
    benchmarker = GTOHaloBenchmarker(config)
    results = benchmarker.run_benchmark()
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total sampling time: {results['sampling_efficiency_metrics'].get('total_sampling_time', 0):.2f}s")
    print(f"Samples per second: {results['sampling_efficiency_metrics'].get('samples_per_second', 0):.3f}")
    print(f"Class label mean: {results['gto_halo_metrics'].get('class_label_mean', 0):.6f}")
    print(f"Feasible ratio: {results['physical_validation_metrics'].get('feasible_ratio', 0):.3f}")
    print(f"Local optimal ratio: {results['physical_validation_metrics'].get('local_optimal_ratio', 0):.3f}")
    print(f"Results saved to: {args.output_dir}")
    
    # Comprehensive dataset information
    if results['physical_validation_metrics'].get('comprehensive_dataset_saved', False):
        print(f"Comprehensive dataset: {args.output_dir}/comprehensive_dataset/")
        print(f"Dataset entries: {results['physical_validation_metrics'].get('dataset_entries', 0)}")
        print("Dataset includes: complete_dataset.pkl, feasible_trajectories.pkl, and usage guide")
    
    print("Check the summary.txt and comprehensive_dataset/README.md for detailed results.")


if __name__ == "__main__":
    main() 