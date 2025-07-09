import matplotlib.pyplot as plt
import numpy as np
import re
import time
import os
import glob
from datetime import datetime

def find_latest_log():
    """Find the most recent log file in the runs directory."""
    log_pattern = "runs/*/*/*/logs"
    log_files = glob.glob(log_pattern)
    if not log_files:
        return None
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return log_files[0]

def parse_log_file(log_file):
    """Parse the log file and extract step, training loss, and evaluation loss data."""
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for training loss lines: "step: 100, training_loss: 1.80509e+01"
                train_match = re.search(r'step: (\d+), training_loss: ([\d.e+-]+)', line)
                if train_match:
                    step = int(train_match.group(1))
                    loss = float(train_match.group(2))
                    train_steps.append(step)
                    train_losses.append(loss)
                
                # Look for evaluation loss lines: "step: 100, evaluation_loss: 1.80509e+01"
                eval_match = re.search(r'step: (\d+), evaluation_loss: ([\d.e+-]+)', line)
                if eval_match:
                    step = int(eval_match.group(1))
                    loss = float(eval_match.group(2))
                    eval_steps.append(step)
                    eval_losses.append(loss)
    except Exception as e:
        print(f"Error reading log file: {e}")
    
    return train_steps, train_losses, eval_steps, eval_losses

def plot_losses_realtime():
    """Plot training and evaluation losses in real-time with moving averages."""
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 8))
    
    last_plot_time = 0
    plot_interval = 2  # Update plot every 2 seconds
    
    while True:
        log_file = find_latest_log()
        if log_file is None:
            print("No log files found. Waiting...")
            time.sleep(5)
            continue
        
        current_time = time.time()
        if current_time - last_plot_time < plot_interval:
            time.sleep(0.5)
            continue
        
        train_steps, train_losses, eval_steps, eval_losses = parse_log_file(log_file)
        
        if not train_steps:
            print("No loss data found. Waiting...")
            time.sleep(5)
            continue
        
        # Clear the plot
        ax.clear()
        
        # Plot raw training loss data
        ax.plot(train_steps, train_losses, 'b-', alpha=0.6, label='Training Loss', linewidth=1)
        ax.scatter(train_steps, train_losses, color='blue', s=30, alpha=0.6)
        
        # Plot raw evaluation loss data
        if eval_steps:
            ax.plot(eval_steps, eval_losses, 'r-', alpha=0.6, label='Evaluation Loss', linewidth=1)
            ax.scatter(eval_steps, eval_losses, color='red', s=30, alpha=0.6)
        
        # Calculate and plot moving averages
        if len(train_losses) > 10:
            window_size = min(20, len(train_losses) // 5)  # Adaptive window size
            train_moving_avg = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
            train_moving_avg_steps = train_steps[window_size-1:]
            ax.plot(train_moving_avg_steps, train_moving_avg, 'b-', linewidth=3, 
                   label=f'Training Loss (MA, window={window_size})', alpha=0.8)
        
        if len(eval_losses) > 10:
            window_size = min(20, len(eval_losses) // 5)  # Adaptive window size
            eval_moving_avg = np.convolve(eval_losses, np.ones(window_size)/window_size, mode='valid')
            eval_moving_avg_steps = eval_steps[window_size-1:]
            ax.plot(eval_moving_avg_steps, eval_moving_avg, 'r-', linewidth=3, 
                   label=f'Evaluation Loss (MA, window={window_size})', alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('Training Step', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title('Real-Time Training and Evaluation Loss\nGTO Halo Diffusion Model', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Set y-axis limits to focus on recent loss range
        if len(train_losses) > 10:
            recent_train = train_losses[-50:]  # Last 50 points
            recent_eval = eval_losses[-50:] if eval_losses else []
            all_recent = recent_train + recent_eval
            if all_recent:
                y_min = max(0, min(all_recent) * 0.9)
                y_max = max(all_recent) * 1.1
                ax.set_ylim(y_min, y_max)
        
        # Add statistics
        if train_losses:
            min_train = min(train_losses)
            max_train = max(train_losses)
            current_train = train_losses[-1]
            ax.text(0.02, 0.98, f'Training Loss Range: {min_train:.2f} - {max_train:.2f}\nCurrent: {current_train:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        if eval_losses:
            min_eval = min(eval_losses)
            max_eval = max(eval_losses)
            current_eval = eval_losses[-1]
            ax.text(0.02, 0.90, f'Eval Loss Range: {min_eval:.2f} - {max_eval:.2f}\nCurrent: {current_eval:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Add progress indicator
        current_step = max(train_steps) if train_steps else 0
        total_steps = 1500
        progress = (current_step / total_steps) * 100
        ax.text(0.02, 0.82, f'Progress: {current_step:,} / {total_steps:,} steps ({progress:.2f}%)', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        last_plot_time = current_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Updated plot with {len(train_steps)} training points, {len(eval_steps)} eval points")

if __name__ == "__main__":
    print("[INFO] Starting real-time loss plotting...")
    log_file = find_latest_log()
    if log_file:
        print(f"[INFO] Monitoring log file: {log_file}")
    else:
        print("[WARNING] No log files found. Will wait for training to start.")
    
    try:
        plot_losses_realtime()
    except KeyboardInterrupt:
        print("\n[INFO] Plotting stopped by user.")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}") 