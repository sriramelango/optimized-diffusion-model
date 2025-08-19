import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import random
import os

def load_dataset(dataset_path):
    """Load the comprehensive dataset."""
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")
    return data

def extract_trajectory_data(sample):
    """Extract trajectory data from a sample."""
    if 'physical_trajectories' not in sample:
        return None, None
    
    trajectory_data = sample['physical_trajectories']
    
    if 'converged_states' not in trajectory_data or 'predicted_states' not in trajectory_data:
        return None, None
    
    converged_states = np.array(trajectory_data['converged_states'])
    predicted_states = np.array(trajectory_data['predicted_states'])
    
    # Ensure both trajectories have the same length
    min_length = min(len(converged_states), len(predicted_states))
    converged_states = converged_states[:min_length]
    predicted_states = predicted_states[:min_length]
    
    return converged_states, predicted_states

def plot_3d_trajectory_comparison(converged_states, predicted_states, sample_idx, output_dir):
    """Create a 3D plot comparing converged vs predicted trajectories."""
    
    # Extract position data (first 3 columns)
    converged_pos = converged_states[:, :3]
    predicted_pos = predicted_states[:, :3]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(converged_pos[:, 0], converged_pos[:, 1], converged_pos[:, 2], 
            'b-', linewidth=2, label='Converged Trajectory', alpha=0.8)
    ax.plot(predicted_pos[:, 0], predicted_pos[:, 1], predicted_pos[:, 2], 
            'r--', linewidth=2, label='Predicted Trajectory', alpha=0.8)
    
    # Mark start and end points
    ax.scatter(converged_pos[0, 0], converged_pos[0, 1], converged_pos[0, 2], 
               c='green', s=100, marker='o', label='Start Point')
    ax.scatter(converged_pos[-1, 0], converged_pos[-1, 1], converged_pos[-1, 2], 
               c='blue', s=100, marker='s', label='Converged End')
    ax.scatter(predicted_pos[-1, 0], predicted_pos[-1, 1], predicted_pos[-1, 2], 
               c='red', s=100, marker='^', label='Predicted End')
    
    # Calculate deviation metrics
    pos_deviation = np.linalg.norm(converged_pos - predicted_pos, axis=1)
    max_deviation = np.max(pos_deviation)
    mean_deviation = np.mean(pos_deviation)
    final_deviation = pos_deviation[-1]
    
    # Set labels and title
    ax.set_xlabel('X (DU)')
    ax.set_ylabel('Y (DU)')
    ax.set_zlabel('Z (DU)')
    ax.set_title(f'Trajectory Comparison - Sample {sample_idx}\n'
                f'Max Dev: {max_deviation:.4f} DU, Mean Dev: {mean_deviation:.4f} DU, Final Dev: {final_deviation:.4f} DU')
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save plot
    output_path = os.path.join(output_dir, f'trajectory_comparison_sample_{sample_idx}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved trajectory comparison plot: {output_path}")
    print(f"  - Max deviation: {max_deviation:.4f} DU")
    print(f"  - Mean deviation: {mean_deviation:.4f} DU")
    print(f"  - Final deviation: {final_deviation:.4f} DU")
    
    return {
        'sample_idx': sample_idx,
        'max_deviation': max_deviation,
        'mean_deviation': mean_deviation,
        'final_deviation': final_deviation,
        'trajectory_length': len(converged_pos)
    }

def main():
    # Define dataset path for 30-sample benchmark
    dataset_path = "test_30_samples_benchmark/comprehensive_dataset/complete_dataset.pkl"
    
    # Create output directory
    output_dir = "trajectory_3d_plots_30_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("30-SAMPLE BENCHMARK TRAJECTORY COMPARISON")
    print("="*60)
    
    # Load dataset
    data = load_dataset(dataset_path)
    
    # Find feasible samples with trajectory data
    feasible_samples = []
    for i, sample in enumerate(data):
        if sample.get('snopt_results', {}).get('feasibility', False):
            converged_states, predicted_states = extract_trajectory_data(sample)
            if converged_states is not None and predicted_states is not None:
                feasible_samples.append((i, sample, converged_states, predicted_states))
    
    print(f"Found {len(feasible_samples)} feasible samples with trajectory data")
    
    if len(feasible_samples) < 3:
        print(f"Not enough feasible samples with trajectory data. Found {len(feasible_samples)} samples.")
        return
    
    # Select 3 random samples
    selected_samples = random.sample(feasible_samples, 3)
    
    # Generate plots for each selected sample
    results = []
    for sample_idx, sample, converged_states, predicted_states in selected_samples:
        print(f"\nProcessing sample {sample_idx}...")
        result = plot_3d_trajectory_comparison(converged_states, predicted_states, sample_idx, output_dir)
        results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("30-SAMPLE BENCHMARK TRAJECTORY COMPARISON SUMMARY")
    print(f"{'='*60}")
    for result in results:
        print(f"Sample {result['sample_idx']}:")
        print(f"  - Trajectory length: {result['trajectory_length']} time steps")
        print(f"  - Max deviation: {result['max_deviation']:.4f} DU")
        print(f"  - Mean deviation: {result['mean_deviation']:.4f} DU")
        print(f"  - Final deviation: {result['final_deviation']:.4f} DU")
        print()
    
    # Calculate overall statistics
    max_deviations = [r['max_deviation'] for r in results]
    mean_deviations = [r['mean_deviation'] for r in results]
    final_deviations = [r['final_deviation'] for r in results]
    
    print(f"30-SAMPLE BENCHMARK OVERALL STATISTICS (3 samples):")
    print(f"  - Average max deviation: {np.mean(max_deviations):.4f} ± {np.std(max_deviations):.4f} DU")
    print(f"  - Average mean deviation: {np.mean(mean_deviations):.4f} ± {np.std(mean_deviations):.4f} DU")
    print(f"  - Average final deviation: {np.mean(final_deviations):.4f} ± {np.std(final_deviations):.4f} DU")
    
    # Analyze all feasible samples for comprehensive statistics
    print(f"\n{'='*60}")
    print("COMPREHENSIVE ANALYSIS OF ALL FEASIBLE SAMPLES")
    print(f"{'='*60}")
    
    all_results = []
    for sample_idx, sample, converged_states, predicted_states in feasible_samples:
        # Calculate deviation metrics for all samples
        converged_pos = converged_states[:, :3]
        predicted_pos = predicted_states[:, :3]
        
        pos_deviation = np.linalg.norm(converged_pos - predicted_pos, axis=1)
        max_deviation = np.max(pos_deviation)
        mean_deviation = np.mean(pos_deviation)
        final_deviation = pos_deviation[-1]
        
        all_results.append({
            'sample_idx': sample_idx,
            'max_deviation': max_deviation,
            'mean_deviation': mean_deviation,
            'final_deviation': final_deviation,
            'trajectory_length': len(converged_pos)
        })
    
    # Calculate comprehensive statistics
    all_max_deviations = [r['max_deviation'] for r in all_results]
    all_mean_deviations = [r['mean_deviation'] for r in all_results]
    all_final_deviations = [r['final_deviation'] for r in all_results]
    
    print(f"COMPREHENSIVE STATISTICS ({len(all_results)} feasible samples):")
    print(f"  - Average max deviation: {np.mean(all_max_deviations):.4f} ± {np.std(all_max_deviations):.4f} DU")
    print(f"  - Average mean deviation: {np.mean(all_mean_deviations):.4f} ± {np.std(all_mean_deviations):.4f} DU")
    print(f"  - Average final deviation: {np.mean(all_final_deviations):.4f} ± {np.std(all_final_deviations):.4f} DU")
    print(f"  - Min max deviation: {np.min(all_max_deviations):.4f} DU")
    print(f"  - Max max deviation: {np.max(all_max_deviations):.4f} DU")
    print(f"  - Min mean deviation: {np.min(all_mean_deviations):.4f} DU")
    print(f"  - Max mean deviation: {np.max(all_mean_deviations):.4f} DU")
    
    print(f"\nPlots saved to: {output_dir}")
    print(f"Total feasible samples analyzed: {len(all_results)}")

if __name__ == "__main__":
    main() 