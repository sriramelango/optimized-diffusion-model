
#!/usr/bin/env python3
"""
Fast Multithreaded GTO Halo Benchmarking Module

This version reduces thread initialization delays by:
1. Pre-initializing thread pool
2. Pre-loading libraries in each thread
3. Using thread-local storage for simulators
4. Reducing file I/O overhead
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

# Add thread lock for matplotlib operations to prevent PDF corruption
matplotlib_lock = threading.Lock()

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
    output_dir: str = "benchmark_results/gto_halo_fast_results"
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
            
            # CR3BP simulation parameters (hardcoded to match 1D implementation exactly)
            cr3bp_config = {
                'seed': thread_id,
                'seed_step': 1,  # Process single sample
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
    """Process a single sample with pre-initialized thread resources."""
    sample_idx, sample_data, halo_energy, output_dir, thread_id = args
    
    # Set up thread-specific matplotlib backend to avoid conflicts
    import matplotlib
    matplotlib.use('Agg')  # Ensure non-interactive backend
    
    try:
        # Force immediate output with flush
        print(f"Thread {thread_id}: Starting sample {sample_idx} with halo energy {halo_energy:.6f}", flush=True)
        
        # Create a fresh simulator for each sample to avoid output conflicts
        # CR3BP simulation parameters (hardcoded to match 1D implementation exactly)
        cr3bp_config = {
            'seed': sample_idx,
            'seed_step': 1,  # Process single sample
            'quiet_snopt': True,  # Suppress SNOPT output for cleaner telemetry
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
            'result_folder': os.path.join(output_dir, 'cr3bp_results')
        }
        
        # Create a temporary file for this sample
        temp_dir = os.path.join(output_dir, 'temp_samples')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f'sample_{sample_idx}.pkl')
        
        # Save single sample
        with open(temp_file, 'wb') as f:
            pickle.dump(sample_data.reshape(1, -1), f)
        
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
        
        # Use thread lock to prevent matplotlib PDF corruption
        with matplotlib_lock:
            result_data = simulator.simulate(earth_initial_guess=sample_data, halo_energy=halo_energy)
        
        # Check if PDF was generated successfully
        expected_pdf = os.path.join(output_dir, 'cr3bp_results', f'earth_mission_alpha_{halo_energy}_seed_{sample_idx}_DM.pdf')
        if os.path.exists(expected_pdf):
            pdf_size = os.path.getsize(expected_pdf)
            if pdf_size == 0:
                print(f"⚠️  WARNING: Thread {thread_id} - Sample {sample_idx} - PDF is empty (0 bytes)", flush=True)
            else:
                print(f"✓ Thread {thread_id} - Sample {sample_idx} - PDF generated successfully ({pdf_size} bytes)", flush=True)
        else:
            print(f"⚠️  WARNING: Thread {thread_id} - Sample {sample_idx} - PDF not found", flush=True)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Analyze result
        convergence_info = "UNKNOWN"
        if result_data:
            if isinstance(result_data, dict):
                if 'feasibility' in result_data:
                    convergence_info = "FEASIBLE" if result_data['feasibility'] else "INFEASIBLE"
                elif 'error' in result_data:
                    convergence_info = f"ERROR: {result_data['error']}"
            else:
                convergence_info = "PROCESSED"
        
        print(f"📊 TELEMETRY: Thread {thread_id} - Sample {sample_idx} - SNOPT COMPLETED - Result: {convergence_info}", flush=True)
        
        return {
            'sample_idx': sample_idx,
            'halo_energy': halo_energy,
            'result_data': result_data,
            'initial_guess': sample_data
        }
        
    except Exception as e:
        print(f"Thread {thread_id}: Error processing sample {sample_idx}: {e}", flush=True)
        return {
            'sample_idx': sample_idx,
            'halo_energy': halo_energy,
            'result_data': {'feasibility': False, 'error': str(e)},
            'initial_guess': sample_data
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
        
        # Initialize counters for tracking boundary violations (original logic)
        self.total_spherical_clips = 0
        self.total_spherical_elements = 0
        
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
        
        num_batches = (self.config.num_samples + self.config.batch_size - 1) // self.config.batch_size
        
        for i in range(num_batches):
            batch_size = min(self.config.batch_size, self.config.num_samples - i * self.config.batch_size)
            
            # For GTO Halo data, use uniform sampling of class labels in [0, 1]
            class_labels = torch.rand(batch_size, 1, device=self.config.device)
            
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

        # --- FLATTEN TO (N, 67) ---
        samples = all_samples.reshape(all_samples.shape[0], -1)  # (N, 81)
        samples = samples[:, :67]  # Keep only the first 67 values

        # Extract class labels (normalized halo energies) and model outputs separately
        # This matches the 1D implementation approach
        class_labels_normalized = samples[:, 0]  # First column is normalized class labels
        model_outputs = samples[:, 1:]  # Rest is the model output (66 values)

        # Skip mean/std unnormalization - model outputs are already in [0,1] range
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
        
        # Convert cartesian control back to correct range, NO CONVERSION TO POLAR
        model_outputs[:, 3:-3] = model_outputs[:, 3:-3] * 2 * thrust - thrust
        
        # Convert to spherical coordinates (exactly as in 1D script)
        # Need to be careful with slicing - the control variables should be in groups of 3
        control_section = model_outputs[:, 3:-3]  # Extract control section
        
        # Reshape to ensure we have groups of 3 (ux, uy, uz)
        num_control_vars = control_section.shape[1]
        num_triplets = num_control_vars // 3
        
        if num_control_vars % 3 != 0:
            # Truncate to nearest multiple of 3
            control_section = control_section[:, :num_triplets*3]
            num_control_vars = num_triplets * 3
        
        # Reshape to (batch_size, num_triplets, 3) for easier processing
        control_reshaped = control_section.reshape(-1, num_triplets, 3)
        ux = control_reshaped[:, :, 0]  # Shape: (batch_size, num_triplets)
        uy = control_reshaped[:, :, 1]  # Shape: (batch_size, num_triplets)  
        uz = control_reshaped[:, :, 2]  # Shape: (batch_size, num_triplets)
        
        alpha, beta, r = self._convert_to_spherical(ux, uy, uz)
        
        # Put the spherical coordinates back
        control_reshaped[:, :, 0] = alpha
        control_reshaped[:, :, 1] = beta
        control_reshaped[:, :, 2] = r
        
        # Reshape back and put into model_outputs
        model_outputs[:, 3:3+num_control_vars] = control_reshaped.reshape(-1, num_control_vars)
        
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
    
    def _convert_to_spherical(self, ux, uy, uz):
        """Convert cartesian coordinates to spherical coordinates (exactly as in 1D script)."""
        u = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
        theta = np.zeros_like(u)
        mask_non_zero = u != 0
        theta[mask_non_zero] = np.arcsin(uz[mask_non_zero] / u[mask_non_zero])
        alpha = np.arctan2(uy, ux)
        alpha = np.where(alpha >= 0, alpha, 2 * np.pi + alpha)

        # Make sure theta is in [0, 2*pi]
        theta = np.where(theta >= 0, theta, 2 * np.pi + theta)
        
        # Track how many times we need to clip u > 1
        u_exceeds_one = u > 1
        num_clips = np.sum(u_exceeds_one)
        total_elements = u.size
        
        # Accumulate global statistics
        self.total_spherical_clips += num_clips
        self.total_spherical_elements += total_elements
        
        if num_clips > 0:
            print(f"⚠️  SPHERICAL CONVERSION CLIPPING: {num_clips}/{total_elements} values ({100*num_clips/total_elements:.2f}%) exceeded magnitude 1")
            print(f"   Max magnitude before clipping: {np.max(u):.6f}")
            print(f"   Min magnitude before clipping: {np.min(u):.6f}")
        
        # Make sure u is not larger than 1
        u[u>1] = 1
        return alpha, theta, u
    
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
            halo_energy = samples[i, 0]  # First column is already the physical halo energy
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
        
        # Extract result data and initial guesses
        result_data_list = [result['result_data'] for result in all_results]
        initial_guesses_list = [result['initial_guess'] for result in all_results]
        
        # Compute statistics
        physical_metrics = self.compute_cr3bp_statistics(result_data_list, initial_guesses_list)
        
        return physical_metrics
    
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
        
        # Note: Boundary violation checks are handled in _convert_to_spherical method
        # which tracks clipping statistics for thrust magnitudes > 1.0
        
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
        print("RUNNING GTO HALO BENCHMARK (MULTITHREADED - ORIGINAL UNNORMALIZATION)")
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
        
        # Print final spherical conversion statistics
        self.print_spherical_conversion_stats()
        
        return results
    
    def save_results(self, results: Dict[str, Any], samples: np.ndarray):
        """Save benchmark results."""
        # Ensure benchmark_results directory exists
        benchmark_results_dir = "benchmark_results"
        os.makedirs(benchmark_results_dir, exist_ok=True)
        
        # Create the specific output directory
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
        
        print(f"✓ Summary saved to {summary_path}")
    
    def generate_plots(self, results: Dict[str, Any], samples: np.ndarray):
        """Generate visualization plots."""
        # This can be implemented similar to the original benchmarking script
        # For now, we'll skip plotting to focus on the multithreading functionality
        print("✓ Plot generation skipped (focusing on multithreading)")
    
    def print_spherical_conversion_stats(self):
        """Print final statistics about spherical coordinate conversion clipping."""
        if self.total_spherical_elements > 0:
            print("\n" + "="*60)
            print("SPHERICAL CONVERSION CLIPPING STATISTICS")
            print("="*60)
            print(f"Total elements processed: {self.total_spherical_elements}")
            print(f"Total elements clipped (u > 1): {self.total_spherical_clips}")
            print(f"Overall clipping rate: {100*self.total_spherical_clips/self.total_spherical_elements:.4f}%")
            print("="*60)
            
            # Add to results summary
            if hasattr(self, 'config') and hasattr(self.config, 'output_dir'):
                # Ensure the output directory exists
                os.makedirs(self.config.output_dir, exist_ok=True)
                summary_file = os.path.join(self.config.output_dir, 'spherical_clipping_stats.txt')
                with open(summary_file, 'w') as f:
                    f.write("SPHERICAL CONVERSION CLIPPING STATISTICS\n")
                    f.write("="*50 + "\n")
                    f.write(f"Total elements processed: {self.total_spherical_elements}\n")
                    f.write(f"Total elements clipped (u > 1): {self.total_spherical_clips}\n")
                    f.write(f"Overall clipping rate: {100*self.total_spherical_clips/self.total_spherical_elements:.4f}%\n")
                    f.write("="*50 + "\n")
        else:
            print("\nNo spherical coordinate conversion performed.")


def main():
    """Main function to run the fast multithreaded benchmark."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Fast Multithreaded GTO Halo Benchmarking')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--data_path', type=str, required=True, help='Path to reference data')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for sampling')
    parser.add_argument('--max_workers', type=int, default=None, help='Number of worker threads (default: CPU count)')
    parser.add_argument('--chunk_size', type=int, default=1, help='Number of samples per thread')
    parser.add_argument('--pre_warm_threads', action='store_true', default=True, help='Pre-warm threads to reduce initialization delays')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: benchmark_results/gto_halo_fast_results_YYYYMMDD_HHMMSS)')
    parser.add_argument('--enable_physical_validation', action='store_true', default=True, help='Enable physical validation')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create timestamped output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"benchmark_results/gto_halo_fast_results_{timestamp}"
    
    # Ensure the output directory is within benchmark_results
    if not args.output_dir.startswith("benchmark_results/"):
        args.output_dir = f"benchmark_results/{args.output_dir}"
    
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
    print("Check the summary.txt files in each subdirectory for detailed results.")


if __name__ == "__main__":
    main() 
