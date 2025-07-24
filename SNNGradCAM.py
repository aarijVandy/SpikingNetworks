
from __future__ import annotations
import torch, torch.nn.functional as F
from snntorch import utils

class SNNGradCAM:
    """
    Compute Grad‑CAM for any snnTorch network.
    Parameters
    ----------
    model         : nn.Module        – your network (set to .eval() inside)
    target_layer  : nn.Module        – conv / LIF layer whose feature‑maps you want
    keep_graph    : bool (default=False) – set True if you will call backward() again
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

    # ---------- hooks --------------------------------------------------------
    def _save_activation(self, module, inp, output):
        # output shape = [B,C,H,W]
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        # grad_out[0] shape = [B,C,H,W]
        self.gradients = grad_out[0].detach()

    # ---------- core routine -------------------------------------------------
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
        B = images.size(0)
        self.model.eval()
        self.model.zero_grad()

        # 1. Forward pass through time, accumulate spikes
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

    # ---------- tidy‑up ------------------------------------------------------
    def close(self):
        """Remove hooks to prevent memory‑leaks (call when finished)."""
        self._fwd.remove()
        self._bwd.remove()


        