

import time
import copy
from fml.nn import SinkhornLoss, pairwise_distances
from src import geom
from src.losses import AnisotropyLoss, IsometryLoss, IsotropicScalingLoss, GaussianKLoss, ChamferLoss, OTLoss
import ot

import torch
import torch.nn as nn
import numpy as np
import point_cloud_utils as pcu

def optimize_patch(cfg, phi, uv, x, optimizers, output_dict):
    # Cache model with the lowest loss if --use-best is passed
    best_model = None
    best_loss = np.inf
    
    optimizer = optimizers['phi']
    uv_optimizer = optimizers['uv']
    
    sinkhorn_loss = SinkhornLoss(max_iters=cfg.loss.sinkhorn.max_sinkhorn_iters, return_transport_matrix=True)
    mse_loss = nn.MSELoss()
    
    for epoch in range(cfg.local_epochs):
        optimizer.zero_grad()
        uv_optimizer.zero_grad()

        epoch_start_time = time.time()
        y = phi(uv)
        #=============== Compute losses ===============#
        loss = 0.0
        loss_dict = {}
        
        # Either OT losses or chamfer
        use_ot = cfg.loss.sinkhorn.weight > 0 or cfg.loss.exact_emd.weight > 0
        use_chamfer = cfg.loss.chamfer.weight > 0        
        assert not (use_ot and use_chamfer), "OT losses and chamfer loss cannot be used together. Please set one of them to zero weight."

        # OT losses
        if use_ot:
            loss_ot, pi = OTLoss()(x, y, cfg)
            loss += loss_ot
            loss_dict["ot"] = loss.item()
        else:
            pi = None

        # chamfer loss
        if use_chamfer:
            loss_chamfer, _ = ChamferLoss()(x.unsqueeze(0), y.unsqueeze(0))
            loss += loss_chamfer
            loss_dict["chamfer"] = loss_chamfer.item()
            
        use_jacobian_losses = cfg.loss.isometry.weight > 0 or cfg.loss.anisotropy.weight > 0 or cfg.loss.isotropic_scaling.weight > 0 or cfg.loss.gaussian_k.weight > 0
        if use_jacobian_losses:
            J = geom.compute_jacobian(uv, y)
       
        # Regularization losses      
        if(cfg.loss.isometry.weight > 0):
             loss_reg = IsometryLoss()(J)
             loss += loss_reg
             loss_dict["reg"] = 0.5 *loss_reg.item()
             
        if(cfg.loss.anisotropy.weight > 0):
             loss_reg = AnisotropyLoss()(J)
             loss += loss_reg
             loss_dict["reg"] = 0.5 *loss_reg.item()
             
        if(cfg.loss.isotropic_scaling.weight > 0):
             loss_reg = IsotropicScalingLoss(J)
             loss += loss_reg
             loss_dict["reg"] = 0.5 *loss_reg.item()
             
        if(cfg.loss.gaussian_k.weight > 0):
             loss_reg = GaussianKLoss()(J)
             loss += loss_reg
             loss_dict["reg"] = 0.5 *loss_reg.item()


        #=============== END ===============#

        loss.backward()

        if cfg.optimization.use_best and loss.item() < best_loss:
            best_loss = loss.item()
            best_model = copy.deepcopy(phi.state_dict())

        epoch_end_time = time.time()

        if epoch % cfg.optimization.print_every == 0:
            loss_str = " ".join([f"{k}={v:0.5f}" for k, v in loss_dict.items()])
            print(
                "%d/%d: [Loss = %0.5f] [%s] [Time = %0.3f]"
                % (
                    epoch,
                    cfg.local_epochs,
                    loss.item(),
                    loss_str,
                    epoch_end_time - epoch_start_time,
                )
            )

        optimizer.step()
        #uv_optimizer.step()
        
    if cfg.optimization.use_best:
        phi.load_state_dict(best_model)

    output_dict["final_model"] = copy.deepcopy(phi.state_dict())

    torch.save(output_dict, cfg.output)
        
    return best_model, best_loss, pi