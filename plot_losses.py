import matplotlib.pyplot as plt
import numpy as np
import re
import os
import glob

def find_latest_log():
    """Find the most recent log file in the runs directory."""
    log_pattern = "runs/*/*/*/logs"
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

    plt.figure(figsize=(12, 8))
    plt.plot(train_steps, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    plt.scatter(train_steps, train_losses, color='blue', s=30, alpha=0.6)
    if eval_steps:
        plt.plot(eval_steps, eval_losses, 'r-', linewidth=2, label='Evaluation Loss', alpha=0.8)
        plt.scatter(eval_steps, eval_losses, color='red', s=30, alpha=0.6)

    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Evaluation Loss Progress\nGTO Halo Diffusion Model', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    if train_losses:
        min_train = min(train_losses)
        max_train = max(train_losses)
        current_train = train_losses[-1]
        plt.text(0.02, 0.98, f'Training Loss Range: {min_train:.2f} - {max_train:.2f}\nCurrent: {current_train:.2f}', 
                 transform=plt.gca().transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    if eval_losses:
        min_eval = min(eval_losses)
        max_eval = max(eval_losses)
        current_eval = eval_losses[-1]
        plt.text(0.02, 0.90, f'Eval Loss Range: {min_eval:.2f} - {max_eval:.2f}\nCurrent: {current_eval:.2f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    total_steps = 1300001
    current_step = max(train_steps) if train_steps else 0
    progress = (current_step / total_steps) * 100
    plt.text(0.02, 0.82, f'Progress: {current_step:,} / {total_steps:,} steps ({progress:.2f}%)', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig('training_loss_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Disabled for SSH/headless environments

    print(f"Plot saved as 'training_loss_plot.png'")
    print(f"Current step: {current_step}")
    print(f"Progress: {progress:.2f}%")
    print(f"Training loss range: {min(train_losses):.2f} - {max(train_losses):.2f}")
    if eval_losses:
        print(f"Evaluation loss range: {min(eval_losses):.2f} - {max(eval_losses):.2f}") 