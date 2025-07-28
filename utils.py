from snntorch import surrogate, utils
import random
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim


# --------------------------------------------------
# Hyperparameters and device setup
# --------------------------------------------------
# note: using MPS (Metal Performance Shaders) for mac. change to "cuda" for NVIDIA GPUS
SEED = 43  # Global seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BATCH_SIZE   = 64
NUM_EPOCHS   = 14             
NUM_STEPS    = 40                  # Number of time steps per forward pass of the SNN to accumulate spikes
LEARNING_RATE = 1e-3
BETA         = 0.5                 # LIF membrane decay constant
SPIKE_GRAD   = surrogate.fast_sigmoid()
N_CLASSES    = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PGD attack parameters
PGD_PARAMS = {
    "eps": 1.0/255.0 * 2.0,
    "alpha": 0.01,
    "num_steps": 1,
    "targeted": False
}



def get_samples(num_samples, data_loader, seed = SEED):  # Extract random samples from dataset

    dataset = data_loader.dataset
    rng = random.Random(seed)                 # local RNG
    indices = rng.sample(range(len(dataset)), k=num_samples)

    imgs, labs = [], []
    for idx in indices:
        img, lab = dataset[idx]
        imgs.append(img)
        labs.append(lab)
    images = torch.stack(imgs, dim=0)
    labels = torch.tensor(labs, dtype=torch.long)
    return images, labels


# self-defined PGD attack function
def pgd_attack( # TODO: can move this to an adv class / file to collect all techniques
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    alpha: float,
    iters: int,
    num_steps: int,
    device: torch.device,
):
    # original images
    orig = images.detach().to(device)
    adv = orig.clone().requires_grad_(True).to(device)
    model.to(device)

    for _ in range(iters):
        utils.reset(model)  # reset SNN state

        # run SNN for num_steps, accumulate spikes
        spike_sum = torch.zeros(adv.size(0), N_CLASSES, device=device)
        for t in range(num_steps):
            spk = model(adv)
            spike_sum += spk
        logits = spike_sum / num_steps

        loss = nn.CrossEntropyLoss()(logits, labels)
        model.zero_grad()
        loss.backward()  # fresh graph each iteration
        grad = adv.grad.data.sign()

        # PGD step
        adv = adv + alpha * grad
        delta = torch.clamp(adv - orig, -eps, eps)
        adv = torch.clamp(orig + delta, 0.0, 1.0).detach().requires_grad_(True)

    return adv.detach()
