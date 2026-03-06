import copy
import time
import os 

import ot
import torch
import torch.nn as nn
import numpy as np
import point_cloud_utils as pcu
from fml.nn import SinkhornLoss, pairwise_distances
from pytorch3d.loss import chamfer_distance

import src.utils as utils
import src.optim as optim
import src.geom as geom
from src.viz import plot_reconstruction, plot_correspondences, plot_uv
from src.nns import MLP




def main():
    # Get custom config from CLI and YAML/save as run_config
    cfg = utils.get_config()

    # We'll populate this dictionary and save it as output
    output_dict = {
        "final_model": None,
        "uv": None,
        "x": None,
        "transform": None,
        "seed": utils.seed_everything(cfg.seed),
    }
    output_dict = utils.get_output_config(cfg, output_dict)
   
    # Preprocess
    
    # Read a point cloud and normals from a file, center it about its mean, and align it along its principle vectors
    x, n = utils.load_point_cloud_by_file_extension(
        cfg.mesh_filename, compute_normals=True
    )
    n = n[0]
    if(True):
        # Optionally, we downsample
        # https://github.com/fwilliams/point-cloud-utils/releases/tag/0.17.1
        idx = pcu.downsample_point_cloud_poisson_disk(x, 0.03)
        x = x[idx]
        n = n[idx]

    # Center the point cloud about its mean and align about its principle components
    x, transform = utils.transform_pointcloud(x, cfg.device)

    # Generate an initial set of UV samples in the plane
    uv = torch.tensor(
        pcu.lloyd_2d(x.shape[0]).astype(np.float32),
        requires_grad=True,
        device=cfg.device,
    )

    # Initialize the model for the surface
    phi = MLP(2, 3).to(cfg.device)
    # TODO: add new archis

    output_dict["uv"] = uv
    output_dict["x"] = x
    output_dict["transform"] = transform
    
    #plot_uv(uv.detach().cpu().numpy())

    optimizer = torch.optim.Adam(phi.parameters(), lr=cfg.learning_rate)
    uv_optimizer = torch.optim.Adam([uv], lr=cfg.learning_rate)
    

    # Cache correspondences to plot them later
    best_model, best_loss, pi = optim.optimize_patch(cfg, phi, uv, x, {'phi': optimizer, 'uv': uv_optimizer}, output_dict)

    if cfg.plot:
        #plot_uv(uv.detach().cpu().numpy())
        plot_reconstruction(uv=uv, x=x, transform=transform, model=phi, pad=1.0, n=128)
        if(pi is not None):
            plot_correspondences(model=phi, uv=uv, x=x, pi=pi)
            
    #================== Reconstruct dense point cloud and surface===============#
    print("Generating upsamples...")
    v, n = utils.upsample_surface([uv], [transform], [phi], ['cpu'],
                                  num_samples=500,
                                  compute_normals=True)
    print("Saving upsampled cloud...")
    pcu.save_mesh_vn(os.path.join(os.path.dirname(cfg.output), "upsampled.ply"), v, n)
    #===========   evaluate distances ===============#
    # TODO

    #=============== Bonus: Plot Gaussian curvature ===============#
    # Evaluate Gaussian curvature
    #plot_reconstruction(uv=uv, x=None, transform=None, model=phi, pad=1.0, scalar_field_func=geom.gaussian_curvature_fundamental)
    #plot_reconstruction(uv=uv, x=None, transform=None, model=phi, pad=1.0, scalar_field_func=geom.gaussian_curvature_det)

if __name__ == "__main__":
    main()
