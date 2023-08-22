import torchvision.transforms.functional as F
from tqdm import tqdm 
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd 

class PadCenterCrop(object):
    def __init__(self, size, pad_if_needed=True, fill=255, padding_mode='constant'):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill

    def __call__(self, img):

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        return F.center_crop(img, self.size)


class EarlyStopping:
    """
    Code written by Jeff Ellen (ellen@spawar.navy.mil). 
    """
    """Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience (int, optional): How long to wait after last time validation loss
            improved.
        verbose (bool, optional): If True, prints a message for each validation loss
            improvement.
        delta (float, optional): Minimum change in the monitored quantity to
            qualify as an improvement.
    """

    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta

        self.count = 0
        self.best_val_loss = None

    def step(self, val_loss):
        if self.best_val_loss is None or self.best_val_loss - val_loss > self.delta:
            if self.best_val_loss is not None:
                print(
                    f"Validation loss decreased"
                    f" ({self.best_val_loss:.4f} --> {val_loss:.4f})."
                )
            self.count = 0
            self.best_val_loss = val_loss
        else:
            self.count += 1
            print(
                f"Validation loss did not sufficiently decrease from"
                f" {self.best_val_loss:.4f}."
                f" Early stopping count: {self.count} / {self.patience}"
            )

    def should_stop(self):
        return self.count >= self.patience

    def should_save(self):
        return self.best_val_loss is not None and self.count == 0


def save_acc_loss_plots(num_epochs, train_loss, valid_loss, train_acc, valid_acc, logs_directory):
    loss_df = pd.DataFrame({
        'Train Loss': train_loss,
        'Validation Loss': valid_loss,
        'Train Accuracy': train_acc,
        'Validation Accuracy': valid_acc
    })
    os.chdir(logs_directory)
    loss_df.to_csv("loss_and_acc.csv")
    plt.figure(figsize=(12, 6))

    # Loss Plot
    plt.subplot(121)
    plt.title("Train Loss vs Validation Loss")
    plt.plot(range(1, num_epochs + 1), train_loss, label="Train Loss", color='blue')
    plt.plot(range(1, num_epochs + 1), valid_loss, label="Validation Loss", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(122)
    plt.title("Train Accuracy vs Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), train_acc, label="Train Accuracy", color='green')
    plt.plot(range(1, num_epochs + 1), valid_acc, label="Validation Accuracy", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Save the plot as a PNG image
    output_filename = 'loss_val_plot.png'
    output_filepath = os.path.join(logs_directory, output_filename)
    plt.tight_layout()
    plt.savefig(output_filepath)


