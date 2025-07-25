import matplotlib.pyplot as plt
import numpy as np
import re
import os
import glob

def find_latest_log():
    """Find the most recent log file in the Training Runs directory."""
    log_pattern = "Training Runs/*/logs"
    log_files = glob.glob(log_pattern)
    if not log_files:
        return None
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return log_files[0]

def parse_log_file(log_file):
    """Parse the log file and extract step, training loss, and evaluation loss data."""
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Training loss: "step: 100, training_loss: 1.80509e+01"
                train_match = re.search(r'step: (\d+), training_loss: ([\d.e+-]+)', line)
                if train_match:
                    step = int(train_match.group(1))
                    loss = float(train_match.group(2))
                    train_steps.append(step)
                    train_losses.append(loss)
                # Evaluation loss: "step: 100, evaluation_loss: 1.80509e+01"
                eval_match = re.search(r'step: (\d+), evaluation_loss: ([\d.e+-]+)', line)
                if eval_match:
                    step = int(eval_match.group(1))
                    loss = float(eval_match.group(2))
                    eval_steps.append(step)
                    eval_losses.append(loss)
    except Exception as e:
        print(f"Error reading log file: {e}")
    return train_steps, train_losses, eval_steps, eval_losses

if __name__ == "__main__":
    log_file = find_latest_log()
    if not log_file:
        print("No log files found.")
        exit(1)
    print(f"Using log file: {log_file}")

    train_steps, train_losses, eval_steps, eval_losses = parse_log_file(log_file)

    if not train_steps:
        print("No training loss data found in the log file.")
        exit(1)

    # --- Filter range ---
    min_step = 4800
    max_step = 8954
    print(f"Plotting steps from {min_step} to {max_step}.")
    # Filter training data
    filtered_train = [(s, l) for s, l in zip(train_steps, train_losses) if min_step <= s <= max_step]
    if filtered_train:
        f_train_steps, f_train_losses = zip(*filtered_train)
    else:
        f_train_steps, f_train_losses = [], []
    # Filter evaluation data
    filtered_eval = [(s, l) for s, l in zip(eval_steps, eval_losses) if min_step <= s <= max_step]
    if filtered_eval:
        f_eval_steps, f_eval_losses = zip(*filtered_eval)
    else:
        f_eval_steps, f_eval_losses = [], []

    plt.figure(figsize=(12, 8))
    plt.plot(f_train_steps, f_train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    plt.scatter(f_train_steps, f_train_losses, color='blue', s=30, alpha=0.6)
    if f_eval_steps:
        plt.plot(f_eval_steps, f_eval_losses, 'r-', linewidth=2, label='Evaluation Loss', alpha=0.8)
        plt.scatter(f_eval_steps, f_eval_losses, color='red', s=30, alpha=0.6)

    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Evaluation Loss Progress\nGTO Halo Diffusion Model', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    if f_train_losses:
        min_train = min(f_train_losses)
        max_train = max(f_train_losses)
        current_train = f_train_losses[-1]
        plt.text(0.02, 0.98, f'Training Loss Range: {min_train:.2f} - {max_train:.2f}\nCurrent: {current_train:.2f}', 
                 transform=plt.gca().transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    if f_eval_losses:
        min_eval = min(f_eval_losses)
        max_eval = max(f_eval_losses)
        current_eval = f_eval_losses[-1]
        plt.text(0.02, 0.90, f'Eval Loss Range: {min_eval:.2f} - {max_eval:.2f}\nCurrent: {current_eval:.2f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    if f_train_steps:
        total_steps = 1300001
        current_step = max(f_train_steps)
        progress = (current_step / total_steps) * 100
        plt.text(0.02, 0.82, f'Progress: {current_step:,} / {total_steps:,} steps ({progress:.2f}%)', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig('training_loss_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Disabled for SSH/headless environments

    print(f"Plot saved as 'training_loss_plot.png'")
    if f_train_steps:
        print(f"Current step: {current_step}")
        print(f"Progress: {progress:.2f}%")
        print(f"Training loss range: {min(f_train_losses):.2f} - {max(f_train_losses):.2f}")
    if f_eval_losses:
        print(f"Evaluation loss range: {min(f_eval_losses):.2f} - {max(f_eval_losses):.2f}") 