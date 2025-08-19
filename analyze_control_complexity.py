#!/usr/bin/env python3
"""
Control Vector Complexity and Diversity Analysis

This script analyzes the complexity, diversity, and uniqueness of control vectors
in the GTO Halo training dataset. It provides comprehensive metrics and visualizations
to understand how different and complex the control trajectories are.

Features:
1. Control vector diversity analysis
2. Complexity metrics (entropy, variance, uniqueness)
3. Trajectory clustering analysis
4. Parameter-by-parameter complexity
5. Visualizations of control vector distributions
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ControlVectorAnalyzer:
    """Analyze complexity and diversity of control vectors in training dataset."""
    
    def __init__(self, dataset_path: str, output_dir: str = None):
        """Initialize the analyzer with dataset path."""
        self.dataset_path = dataset_path
        self.output_dir = output_dir or "control_analysis_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the training dataset
        self.load_dataset()
        
        # Parameter names for analysis
        self.param_names = self._get_parameter_names()
        
    def load_dataset(self):
        """Load the training dataset."""
        print(f"Loading dataset from {self.dataset_path}")
        
        with open(self.dataset_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Dataset loaded: {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
        
        # Extract control vectors (indices 1:64 - time and thrust variables)
        self.control_vectors = self.data[:, 1:64]  # 63-dimensional control vectors
        self.halo_energies = self.data[:, 0]  # Class labels
        
        print(f"Control vectors shape: {self.control_vectors.shape}")
        print(f"Halo energies range: [{self.halo_energies.min():.6f}, {self.halo_energies.max():.6f}]")
        
    def _get_parameter_names(self):
        """Get parameter names for the 63-dimensional control vector."""
        names = []
        
        # Time parameters (3)
        names.extend(['Shooting Time', 'Initial Coast', 'Final Coast'])
        
        # Thrust segments (60 parameters = 20 segments × 3 spherical coordinates)
        for i in range(20):
            names.extend([f'Seg{i+1}_Alpha', f'Seg{i+1}_Beta', f'Seg{i+1}_R'])
        
        return names
    
    def compute_diversity_metrics(self):
        """Compute diversity and complexity metrics."""
        print("Computing diversity metrics...")
        
        # 1. Variance analysis
        variances = np.var(self.control_vectors, axis=0)
        
        # 2. Entropy analysis (discretize for entropy calculation)
        entropies = []
        for i in range(self.control_vectors.shape[1]):
            # Discretize into 20 bins for entropy calculation
            hist, _ = np.histogram(self.control_vectors[:, i], bins=20, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropies.append(entropy(hist))
        
        # 3. Pairwise distances (sample of 1000 for efficiency)
        n_samples = min(1000, len(self.control_vectors))
        sample_indices = np.random.choice(len(self.control_vectors), n_samples, replace=False)
        sample_vectors = self.control_vectors[sample_indices]
        
        # Compute pairwise distances
        distances = pdist(sample_vectors, metric='euclidean')
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # 4. Uniqueness analysis
        uniqueness_scores = []
        for i in range(self.control_vectors.shape[1]):
            # Count how many unique values (normalized by total samples)
            unique_ratio = len(np.unique(self.control_vectors[:, i])) / len(self.control_vectors)
            uniqueness_scores.append(unique_ratio)
        
        # 5. Correlation analysis
        correlation_matrix = np.corrcoef(self.control_vectors.T)
        
        self.diversity_metrics = {
            'variances': variances,
            'entropies': entropies,
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'uniqueness_scores': uniqueness_scores,
            'correlation_matrix': correlation_matrix
        }
        
        print(f"Mean pairwise distance: {mean_distance:.4f} ± {std_distance:.4f}")
        print(f"Mean uniqueness score: {np.mean(uniqueness_scores):.4f}")
        print(f"Mean entropy: {np.mean(entropies):.4f}")
    
    def perform_pca_analysis(self):
        """Perform PCA analysis to understand dimensionality and clustering."""
        print("Performing PCA analysis...")
        
        # Standardize the data
        control_vectors_std = (self.control_vectors - np.mean(self.control_vectors, axis=0)) / np.std(self.control_vectors, axis=0)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(control_vectors_std)
        
        # Analyze explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        self.pca_results = {
            'pca_result': pca_result,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'n_components_95': n_components_95,
            'components': pca.components_
        }
        
        print(f"95% variance explained by {n_components_95} components")
        print(f"First 5 components explain: {np.sum(explained_variance_ratio[:5]):.4f} of variance")
    
    def perform_clustering_analysis(self):
        """Perform clustering analysis to understand trajectory groups."""
        print("Performing clustering analysis...")
        
        # Use PCA-reduced data for clustering
        n_clusters = 5  # Start with 5 clusters
        pca_reduced = self.pca_results['pca_result'][:, :10]  # Use first 10 components
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_reduced)
        
        # Compute silhouette score
        silhouette_avg = silhouette_score(pca_reduced, cluster_labels)
        
        # Analyze cluster characteristics
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_energies = self.halo_energies[cluster_mask]
            cluster_size = np.sum(cluster_mask)
            
            cluster_analysis[i] = {
                'size': cluster_size,
                'percentage': cluster_size / len(self.control_vectors) * 100,
                'mean_energy': np.mean(cluster_energies),
                'std_energy': np.std(cluster_energies),
                'energy_range': [np.min(cluster_energies), np.max(cluster_energies)]
            }
        
        self.clustering_results = {
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'cluster_analysis': cluster_analysis,
            'n_clusters': n_clusters
        }
        
        print(f"Silhouette score: {silhouette_avg:.4f}")
        for i in range(n_clusters):
            analysis = cluster_analysis[i]
            print(f"Cluster {i}: {analysis['size']} samples ({analysis['percentage']:.1f}%), "
                  f"mean energy: {analysis['mean_energy']:.6f}")
    
    def plot_diversity_analysis(self):
        """Create comprehensive diversity analysis plots."""
        print("Creating diversity analysis plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Control Vector Diversity and Complexity Analysis', fontsize=16)
        
        # 1. Variance by parameter
        axes[0, 0].bar(range(len(self.param_names)), self.diversity_metrics['variances'])
        axes[0, 0].set_title('Parameter Variance')
        axes[0, 0].set_xlabel('Parameter Index')
        axes[0, 0].set_ylabel('Variance')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Entropy by parameter
        axes[0, 1].bar(range(len(self.param_names)), self.diversity_metrics['entropies'])
        axes[0, 1].set_title('Parameter Entropy')
        axes[0, 1].set_xlabel('Parameter Index')
        axes[0, 1].set_ylabel('Entropy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Uniqueness by parameter
        axes[0, 2].bar(range(len(self.param_names)), self.diversity_metrics['uniqueness_scores'])
        axes[0, 2].set_title('Parameter Uniqueness')
        axes[0, 2].set_xlabel('Parameter Index')
        axes[0, 2].set_ylabel('Uniqueness Score')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Summary statistics text box
        summary_text = f"""SUMMARY STATISTICS
Silhouette Score: {self.clustering_results['silhouette_score']:.4f}
Mean Pairwise Distance: {self.diversity_metrics['mean_distance']:.4f} ± {self.diversity_metrics['std_distance']:.4f}
Components for 95% Variance: {self.pca_results['n_components_95']}
Mean Variance: {np.mean(self.diversity_metrics['variances']):.4f}
Mean Entropy: {np.mean(self.diversity_metrics['entropies']):.4f}
Mean Uniqueness: {np.mean(self.diversity_metrics['uniqueness_scores']):.4f}"""
        
        axes[1, 0].text(0.1, 0.5, summary_text, transform=axes[1, 0].transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        axes[1, 0].set_title('Summary Statistics')
        axes[1, 0].axis('off')
        
        # 5. Clustering results
        cluster_sizes = [self.clustering_results['cluster_analysis'][i]['size'] 
                        for i in range(self.clustering_results['n_clusters'])]
        cluster_labels = [f'Cluster {i}' for i in range(self.clustering_results['n_clusters'])]
        axes[1, 1].pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%')
        axes[1, 1].set_title('Trajectory Clusters')
        
        # 6. Energy distribution by cluster
        for i in range(self.clustering_results['n_clusters']):
            cluster_mask = self.clustering_results['cluster_labels'] == i
            cluster_energies = self.halo_energies[cluster_mask]
            axes[1, 2].hist(cluster_energies, alpha=0.7, label=f'Cluster {i}', bins=20)
        axes[1, 2].set_title('Halo Energy Distribution by Cluster')
        axes[1, 2].set_xlabel('Halo Energy')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'diversity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Diversity analysis plot saved to {os.path.join(self.output_dir, 'diversity_analysis.png')}")
    
    def plot_parameter_complexity(self):
        """Create detailed parameter-by-parameter complexity analysis."""
        print("Creating parameter complexity plots...")
        
        # Group parameters by type
        time_params = self.param_names[:3]
        thrust_params = self.param_names[3:]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Parameter-by-Parameter Complexity Analysis', fontsize=16)
        
        # 1. Time parameters analysis
        time_variances = self.diversity_metrics['variances'][:3]
        time_entropies = self.diversity_metrics['entropies'][:3]
        
        x = np.arange(len(time_params))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, time_variances, width, label='Variance', alpha=0.7)
        axes[0, 0].bar(x + width/2, time_entropies, width, label='Entropy', alpha=0.7)
        axes[0, 0].set_title('Time Parameters Complexity')
        axes[0, 0].set_xlabel('Parameter')
        axes[0, 0].set_ylabel('Complexity Metric')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(time_params, rotation=45)
        axes[0, 0].legend()
        
        # 2. Thrust parameters analysis (first 20 segments)
        thrust_variances = self.diversity_metrics['variances'][3:63]
        thrust_entropies = self.diversity_metrics['entropies'][3:63]
        
        # Reshape to 20 segments × 3 parameters
        thrust_variances_reshaped = np.array(thrust_variances).reshape(20, 3)
        thrust_entropies_reshaped = np.array(thrust_entropies).reshape(20, 3)
        
        x = np.arange(20)
        width = 0.25
        
        for i in range(3):
            axes[0, 1].bar(x + i*width, thrust_variances_reshaped[:, i], width, 
                           label=f'{"Alpha" if i==0 else "Beta" if i==1 else "R"}', alpha=0.7)
        
        axes[0, 1].set_title('Thrust Parameters Variance (First 20 Segments)')
        axes[0, 1].set_xlabel('Segment')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].legend()
        
        # 3. Correlation heatmap (first 20 parameters for visibility)
        correlation_subset = self.diversity_metrics['correlation_matrix'][:20, :20]
        im = axes[1, 0].imshow(correlation_subset, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title('Parameter Correlation Matrix (First 20)')
        axes[1, 0].set_xlabel('Parameter Index')
        axes[1, 0].set_ylabel('Parameter Index')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Complexity ranking
        complexity_scores = (self.diversity_metrics['variances'] + 
                           self.diversity_metrics['entropies'] + 
                           self.diversity_metrics['uniqueness_scores']) / 3
        
        top_complex_params = np.argsort(complexity_scores)[-10:]  # Top 10 most complex
        axes[1, 1].barh(range(10), complexity_scores[top_complex_params])
        axes[1, 1].set_yticks(range(10))
        axes[1, 1].set_yticklabels([self.param_names[i] for i in top_complex_params])
        axes[1, 1].set_title('Top 10 Most Complex Parameters')
        axes[1, 1].set_xlabel('Complexity Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_complexity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter complexity plot saved to {os.path.join(self.output_dir, 'parameter_complexity.png')}")
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        print("Creating summary report...")
        
        # Calculate overall metrics
        overall_variance = np.mean(self.diversity_metrics['variances'])
        overall_entropy = np.mean(self.diversity_metrics['entropies'])
        overall_uniqueness = np.mean(self.diversity_metrics['uniqueness_scores'])
        
        # Find most and least complex parameters
        complexity_scores = (self.diversity_metrics['variances'] + 
                           self.diversity_metrics['entropies'] + 
                           self.diversity_metrics['uniqueness_scores']) / 3
        
        most_complex_idx = np.argmax(complexity_scores)
        least_complex_idx = np.argmin(complexity_scores)
        
        # Create summary
        summary = {
            'dataset_info': {
                'total_samples': len(self.control_vectors),
                'control_vector_dimension': self.control_vectors.shape[1],
                'halo_energy_range': [float(self.halo_energies.min()), float(self.halo_energies.max())]
            },
            'diversity_metrics': {
                'mean_variance': float(overall_variance),
                'mean_entropy': float(overall_entropy),
                'mean_uniqueness': float(overall_uniqueness),
                'mean_pairwise_distance': float(self.diversity_metrics['mean_distance']),
                'std_pairwise_distance': float(self.diversity_metrics['std_distance'])
            },
            'pca_analysis': {
                'n_components_95_variance': int(self.pca_results['n_components_95']),
                'first_5_components_variance': float(np.sum(self.pca_results['explained_variance_ratio'][:5]))
            },
            'clustering_analysis': {
                'n_clusters': self.clustering_results['n_clusters'],
                'silhouette_score': float(self.clustering_results['silhouette_score']),
                'cluster_distribution': {f'cluster_{i}': self.clustering_results['cluster_analysis'][i] 
                                       for i in range(self.clustering_results['n_clusters'])}
            },
            'parameter_analysis': {
                'most_complex_parameter': {
                    'name': self.param_names[most_complex_idx],
                    'index': int(most_complex_idx),
                    'score': float(complexity_scores[most_complex_idx])
                },
                'least_complex_parameter': {
                    'name': self.param_names[least_complex_idx],
                    'index': int(least_complex_idx),
                    'score': float(complexity_scores[least_complex_idx])
                }
            }
        }
        
        # Save JSON report
        report_path = os.path.join(self.output_dir, 'complexity_analysis_report.json')
        with open(report_path, 'w') as f:
            import json
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Recursively convert numpy types
            def convert_dict(d):
                if isinstance(d, dict):
                    return {k: convert_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [convert_dict(v) for v in d]
                else:
                    return convert_numpy(d)
            
            summary_converted = convert_dict(summary)
            json.dump(summary_converted, f, indent=2)
        
        # Create human-readable summary
        summary_path = os.path.join(self.output_dir, 'analysis_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("CONTROL VECTOR COMPLEXITY AND DIVERSITY ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write(f"  Total samples: {summary['dataset_info']['total_samples']}\n")
            f.write(f"  Control vector dimension: {summary['dataset_info']['control_vector_dimension']}\n")
            f.write(f"  Halo energy range: [{summary['dataset_info']['halo_energy_range'][0]:.6f}, {summary['dataset_info']['halo_energy_range'][1]:.6f}]\n\n")
            
            f.write("DIVERSITY METRICS:\n")
            f.write(f"  Mean variance: {summary['diversity_metrics']['mean_variance']:.6f}\n")
            f.write(f"  Mean entropy: {summary['diversity_metrics']['mean_entropy']:.6f}\n")
            f.write(f"  Mean uniqueness: {summary['diversity_metrics']['mean_uniqueness']:.6f}\n")
            f.write(f"  Mean pairwise distance: {summary['diversity_metrics']['mean_pairwise_distance']:.6f} ± {summary['diversity_metrics']['std_pairwise_distance']:.6f}\n\n")
            
            f.write("DIMENSIONALITY ANALYSIS:\n")
            f.write(f"  Components for 95% variance: {summary['pca_analysis']['n_components_95_variance']}\n")
            f.write(f"  First 5 components explain: {summary['pca_analysis']['first_5_components_variance']:.4f} of variance\n\n")
            
            f.write("CLUSTERING ANALYSIS:\n")
            f.write(f"  Number of clusters: {summary['clustering_analysis']['n_clusters']}\n")
            f.write(f"  Silhouette score: {summary['clustering_analysis']['silhouette_score']:.4f}\n")
            f.write("  Cluster distribution:\n")
            for i in range(summary['clustering_analysis']['n_clusters']):
                cluster_info = summary['clustering_analysis']['cluster_distribution'][f'cluster_{i}']
                f.write(f"    Cluster {i}: {cluster_info['size']} samples ({cluster_info['percentage']:.1f}%), "
                       f"mean energy: {cluster_info['mean_energy']:.6f}\n")
            f.write("\n")
            
            f.write("PARAMETER COMPLEXITY:\n")
            f.write(f"  Most complex parameter: {summary['parameter_analysis']['most_complex_parameter']['name']} "
                   f"(score: {summary['parameter_analysis']['most_complex_parameter']['score']:.6f})\n")
            f.write(f"  Least complex parameter: {summary['parameter_analysis']['least_complex_parameter']['name']} "
                   f"(score: {summary['parameter_analysis']['least_complex_parameter']['score']:.6f})\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write("  - High variance indicates parameters with wide value ranges\n")
            f.write("  - High entropy indicates parameters with diverse distributions\n")
            f.write("  - High uniqueness indicates parameters with many distinct values\n")
            f.write("  - Low PCA components needed suggests some redundancy in parameters\n")
            f.write("  - High silhouette score indicates well-separated trajectory clusters\n")
        
        print(f"Summary report saved to {summary_path}")
        print(f"Detailed JSON report saved to {report_path}")
    
    def run_complete_analysis(self):
        """Run the complete control vector complexity analysis."""
        print("Starting control vector complexity and diversity analysis...")
        
        # Run all analyses
        self.compute_diversity_metrics()
        self.perform_pca_analysis()
        self.perform_clustering_analysis()
        
        # Create visualizations
        self.plot_diversity_analysis()
        self.plot_parameter_complexity()
        
        # Create summary report
        self.create_summary_report()
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        print(f"Key files created:")
        print(f"  - diversity_analysis.png: Overall diversity metrics")
        print(f"  - parameter_complexity.png: Parameter-by-parameter analysis")
        print(f"  - analysis_summary.txt: Human-readable summary")
        print(f"  - complexity_analysis_report.json: Detailed JSON report")


def main():
    """Main function to run control vector complexity analysis."""
    parser = argparse.ArgumentParser(description='Analyze control vector complexity and diversity')
    parser.add_argument('dataset_path', type=str, help='Path to training dataset pickle file')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for plots and analysis (default: auto-generated)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found: {args.dataset_path}")
        return
    
    # Run analysis
    analyzer = ControlVectorAnalyzer(args.dataset_path, args.output_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main() 