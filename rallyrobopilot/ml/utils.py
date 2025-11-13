# rallyrobopilot/ml/utils.py

# --- MODIFICATION: Use a non-interactive backend for Matplotlib ---
# This line is crucial. It tells Matplotlib not to create a GUI window,
# which resolves the conflict when running from the command line.
# It must be called BEFORE importing pyplot.
import matplotlib
matplotlib.use('Agg')
# --- END MODIFICATION ---

import matplotlib.pyplot as plt

class LossPlotter:
    """A class to handle live plotting of training and validation loss."""
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.train_losses = []
        self.val_losses = []
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training and Validation Loss Evolution")
        self.ax.grid(True)
        
    def update(self, train_loss, val_loss):
        """Appends new losses and redraws the plot."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def save(self, filepath="loss_evolution.png"):
        """Saves the final plot to a file."""
        # Plot the final data just before saving
        self.ax.cla()
        self.ax.plot(self.train_losses, label='Training Loss', color='blue')
        self.ax.plot(self.val_losses, label='Validation Loss', color='orange')
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training and Validation Loss Evolution")
        self.ax.legend()
        self.ax.grid(True)
        
        self.fig.savefig(filepath)
        print(f"Loss plot saved to {filepath}")