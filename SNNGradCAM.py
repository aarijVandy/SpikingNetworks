
from __future__ import annotations
import torch, torch.nn.functional as F
from snntorch import utils
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from PIL import Image

class SNNGradCAM:
    """
    Compute Grad-CAM for any snnTorch network.
    Parameters
    ----------
    model         : nn.Module        - your network (set to .eval() inside)
    target_layer  : nn.Module        - conv / LIF layer whose feature-maps you want
    keep_graph    : bool (default=False) - set True if you will call backward() again
    """
    def __init__(self, model, target_layer, *, keep_graph: bool = False):
        self.model        = model
        self.target_layer = target_layer
        self.keep_graph   = keep_graph

        # Handles for hooks
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

        self.activations: torch.Tensor | None = None
        self.gradients  : torch.Tensor | None = None

    
    def _save_activation(self, module, inp, output):
        # output shape = [B,C,H,W]
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        # grad_out[0] shape = [B,C,H,W]
        self.gradients = grad_out[0].detach()


    def generate_cam(
        self,
        images      : torch.Tensor,
        target_class: torch.Tensor | None = None,
        *,
        num_steps   : int,
        device      : torch.device,
        normalize   : bool = True,
    ) -> torch.Tensor:
        """
        Returns a CAM for each image — tensor of shape [B,H,W] in 0‑1 range.
        """

        if images.dim() == 3:    # single image, no batch
            images = images.unsqueeze(0)  # → (1, C, H, W)

        B = images.size(0)
        self.model.eval()
        self.model.zero_grad()

        # Forward pass through time to accumulate spikes
        utils.reset(self.model)                 # clear membrane & spike traces
        spike_sum = torch.zeros(B, self.model.fc.out_features, device=device)

        for _ in range(num_steps):
            spike_sum += self.model(images)     # surrogate gradient flows here

        logits = spike_sum / num_steps          # average spike rate

        # 2. Create one‑hot mask for chosen class(es)
        pred = logits.argmax(dim=1)
        if target_class is None:
            target_class = pred

        one_hot = torch.zeros_like(logits)
        one_hot[torch.arange(B), target_class] = 1.0

        # 3. Back‑prop ‑— gradients w.r.t. target_layer feature‑maps get stored
        logits.backward(gradient=one_hot, retain_graph=self.keep_graph)

        # 4. Weight feature maps by global‑avg‑pooled gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * self.activations).sum(dim=1)            # [B,H,W]
        cam = F.relu(cam)

        # 5. Normalise each CAM to 0‑1 for easy display
        if normalize:
            cam_min = cam.flatten(1).min(dim=1)[0].view(B, 1, 1)
            cam_max = cam.flatten(1).max(dim=1)[0].view(B, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return (cam.cpu() , pred)


    def close(self):
        """Remove hooks to prevent memory-leaks (call when finished)."""
        self._fwd.remove()
        self._bwd.remove()



    def plot_gradcam_progress(cam_list, label, adv_label=None, cmap='jet'):
        """
        Plots the Grad-CAM sequence side-by-side for visual progression.

        cam_list: list of (image, cam, pred)
        label: original class
        adv_label: final adversarial prediction (optional)
        """
        num_steps = len(cam_list)
        fig, axes = plt.subplots(2, num_steps, figsize=(2.5 * num_steps, 5))
        for i, (img, cam, pred) in enumerate(cam_list):
            img_np = img[0,0].cpu().numpy() if img.ndim==4 else img.squeeze().cpu().numpy()

            # Plot perturbed image
            axes[0, i].imshow(img_np, cmap='gray')
            axes[0, i].set_title(f"Step {i}\nPred: {pred}")
            axes[0, i].axis('off')

            # Plot CAM overlay
            axes[1, i].imshow(img_np, cmap='gray')
            axes[1, i].imshow(cam, alpha=0.5, cmap=cmap)
            axes[1, i].set_title("Grad-CAM")
            axes[1, i].axis('off')
        plt.suptitle(f"PGD Grad-CAM Progression (True: {label}, Adv: {adv_label})", fontsize=14)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def spike_cam_single(image, label, model, DEVICE = torch.device("cpu"), NUM_STEPS = NUM_STEPS):
        """
        generates Class Activate Mapping for a single image
        """
        if image.dim() == 3:            # [C,H,W] → [1,C,H,W]
            image_cam = image.unsqueeze(0)

        cam_extractor = SNNGradCAM(model, model.conv2)
        cams,pred = cam_extractor.generate_cam(image_cam,
                                        target_class=label,
                                        num_steps=NUM_STEPS,
                                        device=DEVICE)    # [B,H,W]
        
        idx = 0
        fig, ax = plt.subplots(1, 2, figsize=(6,3))
        image_vis = image.squeeze().cpu()
        ax[0].imshow(image_vis, cmap='gray')
        
        # print(shape(image[idx].squeeze().cpu()))
        ax[0].set_title(f'Input — label {label.item()}'); ax[0].axis('off')

        # resize to overlay on orig image 
        cam_np = cams[0].cpu().numpy()
        cam_resized = np.array(Image.fromarray((cam_np * 255).astype(np.uint8))
                            .resize((28, 28), Image.BICUBIC)) / 255.0

        ax[1].imshow(image_vis, cmap='gray')
        ax[1].imshow(cam_resized, alpha=0.5, cmap='jet')

        ax[1].imshow(image[idx].squeeze().cpu(), cmap='gray')
        # cams[idx] = cams[idx].resize((28,28), Image.BICUBIC)   
        ax[1].imshow(cam_resized, alpha=0.5, cmap = 'jet')                 # heat‑map overlay
        ax[1].set_title(f'Grad‑CAM pred: {pred[0]}'); ax[1].axis('off')
        plt.tight_layout(); plt.show()

        # ax.cla()
        # ax.clf

        cam_extractor.close()          # clean up hooks

    @staticmethod
    def spike_cam(test_loader, model, DEVICE = torch.device("cpu"), NUM_STEPS = NUM_STEPS, num_images = 5):

        # pick a batch from your loader
        images,labels = get_samples(num_images, test_loader)
        # images, labels = next(iter(test_loader))
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # target the second conv layer (after ReLU or LIF both work;
        # here we tap into conv2 feature maps)
        cam_extractor = SNNGradCAM(model, model.conv2)

        for i in range(num_images):
            SNNGradCAM.spike_cam_single(images[i], labels[i], model)

        
        cam_extractor.close()          # clean up hooks

    
        