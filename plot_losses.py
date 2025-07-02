import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the data
data = []
with open('training_data.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            step = int(parts[0])
            loss = float(parts[1])
            data.append([step, loss])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['step', 'loss'])

# Separate training and evaluation losses
# Training losses are logged every 50 steps, evaluation every 100 steps
training_steps = []
training_losses = []
eval_steps = []
eval_losses = []

for i, row in df.iterrows():
    step = row['step']
    loss = row['loss']
    
    if step % 100 == 0 and i > 0:  # Evaluation step (every 100 steps)
        eval_steps.append(step)
        eval_losses.append(loss)
    else:  # Training step
        training_steps.append(step)
        training_losses.append(loss)

# Create the plot
plt.figure(figsize=(12, 8))

# Plot training loss
plt.plot(training_steps, training_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
plt.scatter(training_steps, training_losses, color='blue', s=30, alpha=0.6)

# Plot evaluation loss
plt.plot(eval_steps, eval_losses, 'r-', linewidth=2, label='Evaluation Loss', alpha=0.8)
plt.scatter(eval_steps, eval_losses, color='red', s=30, alpha=0.6)

# Customize the plot
plt.xlabel('Training Step', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Evaluation Loss Progress\nGTO Halo Diffusion Model', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Add some statistics
if training_losses:
    min_train = min(training_losses)
    max_train = max(training_losses)
    current_train = training_losses[-1]
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

# Add progress indicator
total_steps = 1300001
current_step = max(training_steps) if training_steps else 0
progress = (current_step / total_steps) * 100
plt.text(0.02, 0.82, f'Progress: {current_step:,} / {total_steps:,} steps ({progress:.2f}%)', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('training_loss_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved as 'training_loss_plot.png'")
print(f"Current step: {current_step}")
print(f"Progress: {progress:.2f}%")
print(f"Training loss range: {min(training_losses):.2f} - {max(training_losses):.2f}")
print(f"Evaluation loss range: {min(eval_losses):.2f} - {max(eval_losses):.2f}") 