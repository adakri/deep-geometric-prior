import torch
from torch.func import vmap, jacrev

def normalize_per_row(x):
    return (x - x.min(0, keepdim=True).values) / (x.max(0, keepdim=True).values - x.min(0, keepdim=True).values + 1e-8)

def isometry_loss(J):
    # Forces the first fundemantal form to be identity (isometry to the plane)
    
    Xu = J[:,:,0]
    Xv = J[:,:,1]
    E = (Xu * Xu).sum(dim=1)
    F = (Xu * Xv).sum(dim=1)
    G = (Xv * Xv).sum(dim=1)
    loss = ((E - 1)**2 + (G - 1)**2 + (F)**2)
    return loss.mean()

def Gaussian_k_loss(J):
    # Forces the first fundemantal form to be identity (isometry to the plane)
    
    return 

def anisotropy_loss(J):
    # singular values
    S = torch.linalg.svdvals(J)
    s1 = S[:,0]
    s2 = S[:,1]
    loss = (s1 - s2)**2
    return loss.mean()

def isotropic_scaling_loss(J):
    Xu = J[:,:,0]
    Xv = J[:,:,1]
    E = (Xu * Xu).sum(dim=1)
    F = (Xu * Xv).sum(dim=1)
    G = (Xv * Xv).sum(dim=1)
    loss = (E - G)**2 + F**2

    return loss.mean()