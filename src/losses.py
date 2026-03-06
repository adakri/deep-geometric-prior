import torch
import numpy as np
import torch.nn as nn
from torch.nn import Module
from torch.func import vmap, jacrev
from fml.nn import SinkhornLoss, pairwise_distances
import ot

# idea of loss mixin from https://github.com/romyjw/sns/blob/main/neural_surfaces-main

# Generic Mixin for losses.
class Loss(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        
class ChamferLoss(Loss):
    """Chamfer distance loss."""
    def forward(self, pred, gt):
        assert pred.dim() == 3 and gt.dim() == 3, "pred and gt should be of shape (B, N, D)"
        assert isinstance(pred, torch.Tensor) and isinstance(gt, torch.Tensor), "pred and gt should be torch tensors"
        from pytorch3d.loss import chamfer_distance
        loss, _ = chamfer_distance(pred, gt)
        return loss, {'chamfer loss': loss.detach()}
    
        
class OTLoss(Loss):
    """OT distance loss."""
    def forward(self, x, y, cfg):
        sinkhorn_loss = SinkhornLoss(max_iters=cfg.loss.sinkhorn.max_sinkhorn_iters, return_transport_matrix=True)
        mse_loss = nn.MSELoss()
        with torch.no_grad():
            if cfg.loss.exact_emd.weight > 0:
                M = (
                    pairwise_distances(x.unsqueeze(0), y.unsqueeze(0))
                    .squeeze()
                    .cpu()
                    .squeeze()
                    .numpy()
                )
                p = ot.emd(np.ones(x.shape[0]), np.ones(x.shape[0]), M)
                p = torch.from_numpy(p.astype(np.float32)).to(cfg.device)
            if cfg.loss.sinkhorn.weight > 0:
                _, p = sinkhorn_loss(x.unsqueeze(0), y.unsqueeze(0))
                
            # Correspondences for plotting
            pi = p.squeeze().max(0)[1]
            
        loss = mse_loss(x[pi].unsqueeze(0), y.unsqueeze(0))
        
        return loss, pi

class MSELoss(Loss):
    """Mean squared error loss."""
    def forward(self, pred, gt):
        B   = gt.size(0)
        loss = (gt - pred).pow(2).view(B, -1).mean()
        return loss

class MAELoss(Loss):
    """Mean absolute error loss."""
    def forward(self, pred, gt):
        B   = gt.size(0)
        loss = (gt - pred).abs().view(B, -1).mean()
        return loss
    
def normalize_per_row(x):
    return (x - x.min(0, keepdim=True).values) / (x.max(0, keepdim=True).values - x.min(0, keepdim=True).values + 1e-8)

class IsometryLoss(Loss):
    """Forces the first fundamental form to be identity (isometry to the plane)."""
    def forward(self, J):
        Xu = J[:,:,0]
        Xv = J[:,:,1]
        E = (Xu * Xu).sum(dim=1)
        F = (Xu * Xv).sum(dim=1)
        G = (Xv * Xv).sum(dim=1)
        loss = ((E - 1)**2 + (G - 1)**2 + (F)**2).mean()
        return loss

class GaussianKLoss(Loss):
    """Placeholder gaussian curvature loss."""
    def forward(self, J):
        # no implementation yet, return zero
        loss = torch.zeros(J.size(0), device=J.device).mean()
        return loss

class AnisotropyLoss(Loss):
    """Encourages the singular values of J to coincide."""
    def forward(self, J):
        S = torch.linalg.svdvals(J)
        s1 = S[:,0]
        s2 = S[:,1]
        loss = (s1 - s2)**2
        loss = loss.mean()
        return loss

class IsotropicScalingLoss(Loss):
    """Penalises non‑uniform scaling in the parameterisation."""
    def forward(self, J):
        Xu = J[:,:,0]
        Xv = J[:,:,1]
        E = (Xu * Xu).sum(dim=1)
        F = (Xu * Xv).sum(dim=1)
        G = (Xv * Xv).sum(dim=1)
        loss = ((E - G)**2 + F**2).mean()
        return loss