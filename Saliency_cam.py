# need to debug

import matplotlib.pyplot as plt
from typing import Tuple
import torch, torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate, utils

def saliency_map(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute saliency maps for SNN by backpropagating the max spike-rate logits
    w.r.t. input images. Returns saliency tensor of same spatial size as images.

    Args:
        model: Trained ConvSNN model
        images: Input batch tensor [B, C, H, W]
        labels: Target labels [B]
        num_steps: Number of time steps to accumulate spikes
        device: torch.device

    Returns:
        saliency: Tensor [B, H, W] of absolute gradient values
    """
    model.eval()
    images = images.to(device).requires_grad_(True)
    labels = labels.to(device)

    # Forward: accumulate spikes
    utils.reset(model)
    spike_sum = torch.zeros(images.size(0), model.fc.out_features, device=device)
    for _ in range(num_steps):
        spk = model(images)
        spike_sum += spk
    logits = spike_sum / num_steps

    # Pick the logit corresponding to the true class
    score = logits.gather(1, labels.unsqueeze(1)).squeeze()

    # Backward: compute gradients w.r.t. inputs
    model.zero_grad()
    score.sum().backward()

    # Saliency: absolute value of input gradients
    saliency = images.grad.abs().detach().cpu()
    # If multi-channel, take max over channels
    if saliency.dim() == 4:
        saliency, _ = saliency.max(dim=1)

    return saliency


def visualize_saliency(
    images: torch.Tensor,
    saliency: torch.Tensor,
    labels: torch.Tensor,
    num_images: int = 5
):
    """
    Plot original images and corresponding saliency maps side-by-side.
    """
    original = images.cpu()
    for i in range(min(num_images, images.size(0))):
        
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes.cla()
        axes[0].imshow(original[i, 0], cmap='gray')
        axes[0].set_title(f'Input - Label {labels[i].item()}')
        axes[0].axis('off')

        axes[1].imshow(saliency[i], cmap='hot')
        axes[1].set_title('Saliency Map')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()


# Example usage in main after loading model and data
# images, labels = get_samples(5, test_loader)
# sal = saliency_map(model, images.to(DEVICE), labels.to(DEVICE), NUM_STEPS, DEVICE)
# visualize_saliency(images, sal, labels)
