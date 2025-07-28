"""
Conv-SNN for MNIST with PGD adversarial testing
Author : Aarij Atiq
Date   : 2025-06-03 
# ssh vunetid@login.accre.vu
# to run: sbatch schedule.slurm
"""

import torch, torch.nn as nn
import torch.optim as optim
import snntorch as snn
from snntorch import surrogate, utils
from torch.utils.data import DataLoader
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
import numpy as np 
from data_loader import get_data_loaders
from PIL import Image
import random
import Saliency_cam
from utils import *
from SNNGradCAM import SNNGradCAM




# --------------------------------------------------
#  Model definition
# --------------------------------------------------
class ConvSNN(nn.Module):
    """
    Convolutional Spiking Neural Network (Leaky LIF neurons) for MNIST.
    """

    def __init__(self, beta: float, spike_grad , input_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global average pooling from 4×4 → 1×1 for 16 feature maps
        self.gap     = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc      = nn.Linear(16, N_CLASSES)
        self.lif3    = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True
        )
    
    
    def visualize(self, x: torch.Tensor):
        """
        for showing how SNNs use poisson-based encoding to encode image into spikes
        """ 
        z = self.conv1(x)
        z = self.relu1(z)
        spk1 = self.lif1(z)[0]
        return spk1
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one time step. Returns spikes of shape [B, N_CLASSES].
        """
        z = self.conv1(x)
        z = self.relu1(z)
        spk1 = self.lif1(z)[0]

        z = self.pool1(spk1)

        z = self.conv2(z)
        z = self.relu2(z)
        spk2 = self.lif2(z)[0]

        z = self.pool2(spk2)
        z = self.gap(z)                     # shape [B, 16, 1, 1]
        z = z.view(z.size(0), -1)           # flatten to [B, 16]
        z = self.dropout(z)

        membrane = self.fc(z)
        spk3 = self.lif3(membrane)[0]
        return spk3  # spike tensor [B, N_CLASSES]
    
     

def run_network(
    model: nn.Module,
    images: torch.Tensor,
    num_steps: int,
    device: torch.device
) -> torch.Tensor:
    """
    Runs `num_steps` forward passes on static input images, accumulates spikes,
    and returns average spike-rate (logits) for classification.
    """
    model.eval()
    utils.reset(model)  # Reset hidden states of all LIF layers
    batch_size = images.size(0)
    spike_sum = torch.zeros(batch_size, N_CLASSES, device=device)

    with torch.no_grad():
        for _ in range(num_steps):
            spk = model(images)      
            spike_sum += spk

    return spike_sum / num_steps  # average spike-rate logits


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device
) -> (float, float):
    """
    Train the model for one epoch and return (average_loss, accuracy_percent).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Accumulate spikes over time
        utils.reset(model)
        spike_sum = torch.zeros(images.size(0), N_CLASSES, device=device)
        for _ in range(num_steps):
            spk = model(images)
            spike_sum += spk
        logits = spike_sum / num_steps

        # Calculate losses and backpropagate
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy



def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    num_steps: int,
    device: torch.device
) -> (float, float):
    """
    Evaluate the model on clean (non-adversarial) data. Returns (average_loss, accuracy_percent).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = run_network(model, images, num_steps, device)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def visualize_images(model: nn.Module, original: torch.Tensor, adversarial: torch.Tensor, labels: torch.Tensor, num_images: int = 5):
    """
    Visualize a grid of original and adversarial images side by side.
    
    Args:
        original: Original input images tensor [B, C, H, W]
        adversarial: Adversarial images tensor [B, C, H, W]
        labels: Ground truth labels tensor [B]
        num_images: Number of image pairs to display
    """
    # Convert tensors to numpy arrays and move to CPU
    original = original.to(DEVICE)
    adversarial = adversarial.to(DEVICE)
    labels = labels.cpu().to(DEVICE)
    
    # Get spike outputs for both original and adversarial images
    with torch.no_grad():
        spk1_orig = model.visualize(original).cpu().numpy()
        spk1_adv = model.visualize(adversarial).cpu().numpy()
    
    # Create figure with subplots (3 columns: original, adversarial, spikes)
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 2*num_images))
    fig.suptitle('Original vs Adversarial Images with Spike Visualization', fontsize=16)

    # Convert tensors to numpy arrays and move to CPU
    original = original.cpu()
    adversarial = adversarial.cpu()
    labels = labels.cpu()
    
    for i in range(num_images):
        # Plot original image
        axes[i, 0].imshow(original[i, 0].numpy(), cmap='gray')
        axes[i, 0].set_title(f'Original (Label: {labels[i]})')
        axes[i, 0].axis('off')
        
        # # # Plot adversarial image
        axes[i, 1].imshow(adversarial[i, 0].numpy(), cmap='gray')
        axes[i, 1].set_title(f'Adversarial (Label: {labels[i]})')
        axes[i, 1].axis('off')
        
        # Plot spike visualization (average across channels)
        spike_vis = np.mean(spk1_orig[i], axis=0)  # Average across channels
        # non-averaged spike visualization
#        spike_vis = spk1_orig[0]
        axes[i, 2].imshow(spike_vis, cmap='gray')
        axes[i, 2].set_title('Spike Output (Layer 1)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def evaluate_adversarial(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    passes: int,
    device: torch.device,
    pgd_params: dict
) -> (float, float):
    """
    Evaluate the model on adversarial data generated by PGD. Returns (avg_loss, accuracy_percent).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Create progress bar for test batches
    pbar = tqdm(data_loader, desc="PGD Adversarial Evaluation", leave=True)
    
    # Store first batch for visualization if requested
    first_batch = None
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Store first batch for visualization
        if batch_idx == 0:
            first_batch = (images.clone(), labels.clone())
        
        # PGD attack
        x_adv = pgd_attack(
            model, images, labels,
            eps=pgd_params["eps"],
            alpha=pgd_params["alpha"],
            iters=pgd_params["num_steps"],
            num_steps=pgd_params["num_steps"],
            device=DEVICE,
        )
        # num_samples = 5
        # idx = 0
        # if idx < num_samples:

        #     spike_cam_single(x_adv, labels, model,
        #                      DEVICE=DEVICE, NUM_STEPS=passes)
        #     idx+=1

        # Evaluate model on adversarial examples
        logits = run_network(model, x_adv, passes, device)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        # Update progress bar with current accuracy
        current_acc = 100.0 * correct / total
        pbar.set_postfix({'Accuracy': f'{current_acc:.2f}%'})
        
        # Visualize first batch if requested
        if batch_idx == 0:
            visualize_images(model, first_batch[0], x_adv, first_batch[1])
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy





def collect_gradcams_over_pgd(
    model: nn.Module,
    image: torch.Tensor,     # single image [1,1,H,W] or [1,C,H,W]
    label: int,
    cam_layer: nn.Module,
    pgd_steps: int = 5,
    eps: float = 0.004,
    alpha: float = 0.1,
    num_steps: int = 40,
    device: torch.device = DEVICE
):
    """
    Returns a list of (adv_image, cam, pred) for each PGD step (including step 0: original).
    """
    # Initial setup
    image = image.to(device).detach().clone()
    label_tensor = torch.tensor([label], dtype=torch.long, device=device)

    adv = image.clone().detach().requires_grad_(True)
    orig = image.clone().detach()
    cam_list = []
    
    # For each step (including the original image as step 0)
    for step in range(pgd_steps + 1):
        # Compute Grad-CAM for current (adversarial) image
        cam_extractor = SNNGradCAM(model, cam_layer)
        cam, pred = cam_extractor.generate_cam(
            adv, target_class=label_tensor, num_steps=num_steps, device=device
        )
        cam_extractor.close()

        cam_list.append((
            adv.detach().cpu().clone(),  # perturbed image at this step
            cam[0].cpu().numpy(),        # Grad-CAM heatmap (0th index: batch of 1)
            int(pred[0].cpu().item())    # predicted label
        ))

        # If last step, break
        if step == pgd_steps:
            break

        # PGD gradient step
        adv.requires_grad = True
        utils.reset(model)
        spike_sum = torch.zeros(1, N_CLASSES, device=device)
        for _ in range(num_steps):
            spike_sum += model(adv)
        logits = spike_sum / num_steps
        loss = nn.CrossEntropyLoss()(logits, label_tensor)
        model.zero_grad()
        loss.backward()
        grad = adv.grad.data.sign()
        adv = adv + alpha * grad
        delta = torch.clamp(adv - orig, -eps, eps)
        adv = torch.clamp(orig + delta, 0.0, 1.0).detach().requires_grad_(True)
    
    return cam_list  # List of (image, cam, pred) for each PGD step




def main(
    test_model:bool = True,
    train_model: bool = False,
    do_pgd: bool = False,
    pgd_params: dict = None,
    save_path: str = './checkpoints/conv_snn.pth',
    load_path: str = None,
    dataset: str = "mnist",
    visualize_images= True
    
):
    """
    Main function to train and/or test the ConvSNN on MNIST.
    If train_model is False, only testing is performed (on a freshly initialized model).
    If do_pgd is True, adversarial test is also performed.
    save_path: where to save the model after training
    load_path: if provided, load model weights from this path before testing/adversarial
    """
    # 5.1 Data loaders
    if dataset == "cifar10":
        in_ch = 3
    else:
        in_ch = 1

    train_loader, test_loader = get_data_loaders(dataset, BATCH_SIZE)

    # 5.2 Model, loss, optimizer
    model = ConvSNN(beta=BETA, spike_grad=SPIKE_GRAD, input_channels=in_ch).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load model if path provided
    if load_path is not None and os.path.exists(load_path):
        print(f"Loading model weights from {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=DEVICE))

    if visualize_images:
        SNNGradCAM.spike_cam(test_loader, model)

    print("showing unadultered saliencies: ")
    images, labels = get_samples(5, test_loader)
    # sal = Saliency_cam.saliency_map(model, images.to(DEVICE), labels.to(DEVICE), NUM_STEPS, DEVICE)
    # Saliency_cam.visualize_saliency(images, sal, labels)


    train_losses, train_accs = [], []
    test_losses, test_accs   = [], []
    adv_losses, adv_accs     = [], []

    if train_model:
        # 5.3 Training loop
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, NUM_STEPS, DEVICE
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, NUM_STEPS, DEVICE
            )

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            print(
                f"Epoch [{epoch:2}/{NUM_EPOCHS}]  "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:5.2f}%  "
                f"Test Loss: {test_loss:.4f}  Acc: {test_acc:5.2f}%"
            )
        # Save model after training
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    else:
        # If not training, run a single clean eval pass
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, NUM_STEPS, DEVICE
        )
        print(
            f"[Clean Evaluation Only] Test Loss: {test_loss:.4f}  "
            f"Acc: {test_acc:5.2f}%"
        )
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    # # PGD adversarial testing (toggle-able)
    # if do_pgd:
    #     if pgd_params is None:
    #         pgd_params = PGD_PARAMS
    #     adv_loss, adv_acc = evaluate_adversarial(
    #         model, test_loader, criterion, PGD_PARAMS["num_steps"], DEVICE, pgd_params
    #     )
    #     adv_losses.append(adv_loss)
    #     adv_accs.append(adv_acc)
    #     print(
    #         f"[PGD Adversarial Test] Loss: {adv_loss:.4f}  "
    #         f"Acc: {adv_acc:5.2f}%"
    #     )

    #     print("showing perturbed saliencies: ")
    #     images, labels = get_samples(5, test_loader)
    #     sal = Saliency_cam.saliency_map(model, images.to(DEVICE), labels.to(DEVICE), NUM_STEPS, DEVICE)
    #     Saliency_cam.visualize_saliency(images, sal, labels)

    if do_pgd:
        samples, labels = get_samples(5, test_loader)
        for i in range(len(samples)):
            # Pick a test sample (e.g. from DataLoader)
            label =  labels[i]
            image = samples[i].unsqueeze(0)

            # Call the function (you can set pgd_steps=10, adjust as needed)
            cam_list = collect_gradcams_over_pgd(
                model, image, label, model.conv2, pgd_steps=10, eps=0.3, alpha=0.01, num_steps=NUM_STEPS, device=DEVICE
            )

            # Optionally, print final predicted class for the last step
            adv_label = cam_list[-1][2]

            # Plot the results
            SNNGradCAM.plot_gradcam_progress(cam_list, label, adv_label=adv_label)


    # 5.5 Plot training curves (only if we trained)
    if train_model:
        epochs = list(range(1, NUM_EPOCHS + 1))
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, test_losses,  label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("Loss Curves")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, label="Train")
        plt.plot(epochs, test_accs,  label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Curves")
        plt.ylim(0, 100)
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test ConvSNN and run adversarial attacks.")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--do_pgd', action='store_true', help='Run PGD adversarial test')
    parser.add_argument('--test', action='store_true', help='test the trained model')
    parser.add_argument('--save_path', type=str, default='./checkpoints/conv_snn.pth', help='Path to save model')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load model')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on')
    parser.add_argument('--test-model', action='store_true', help='Test model accuracy on clean data only (no PGD)')
    args = parser.parse_args()



    main(
        test_model = args.test,
        train_model=args.train,
        do_pgd=args.do_pgd,
        save_path=args.save_path,
        load_path=args.load_path,
        dataset=args.dataset


    )
 