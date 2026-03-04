import argparse
import copy
import time

import torch
import torch.nn as nn
import numpy as np
import point_cloud_utils as pcu

import src.utils as utils
from src.viz import plot_reconstruction, plot_correspondences, plot_uv
from src.nns import MLP
from fml.nn import SinkhornLoss, pairwise_distances
import ot



def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mesh_filename", type=str, help="Point cloud to reconstruct")
    argparser.add_argument("--plot", action="store_true", help="Plot the output when done training")
    argparser.add_argument("--local-epochs", "-nl", type=int, default=128, help="Number of local fitting iterations")
    argparser.add_argument("--global-epochs", "-ng", type=int, default=128, help="Number of global fitting iterations")
    argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="Step size for gradient descent")
    argparser.add_argument("--device", "-d", type=str, default="cuda", help="The device to use when fitting (either 'cpu' or 'cuda')")
    argparser.add_argument( "--loss", type=str, default="sinkhorn", choices=["sinkhorn", "emd", "chamfer", "hausdorff"], help="Loss function to use")
    argparser.add_argument("--max-sinkhorn-iters", "-si", type=int, default=32, help="Maximum number of Sinkhorn iterations")
    argparser.add_argument("--sinkhorn-epsilon", "-sl", type=float, default=1e-3, help="The reciprocal (1/lambda) of the \
                           sinkhorn regularization parameter.")
    argparser.add_argument("--output", "-o", type=str, default="out.pt", help="Destination to save the output reconstruction. \
        Note, the file produced by this script is not a mesh or a point cloud. To construct a dense point cloud, see export_point_cloud.py.")
    argparser.add_argument("--seed", "-s", type=int, default=-1, help="Random seed to use when initializing network weights. If the seed not positive, a seed is selected at random.")
    argparser.add_argument("--use-best", action="store_true", help="Use the model with the lowest loss")
    argparser.add_argument("--print-every", type=int, default=16, help="Print every N epochs")
    args = argparser.parse_args()

    # We'll populate this dictionary and save it as output
    output_dict = {
        "final_model": None,
        "uv": None,
        "x": None,
        "transform": None,
        "loss": args.loss,
        "global_epochs": args.global_epochs,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "max_sinkhorn_iters": args.max_sinkhorn_iters,
        "seed": utils.seed_everything(args.seed),
    }

    # Read a point cloud and normals from a file, center it about its mean, and align it along its principle vectors
    x, n = utils.load_point_cloud_by_file_extension(
        args.mesh_filename, compute_normals=True
    )
    n = n[0]
    if(True):
        # Optionally, we downsample
        # https://github.com/fwilliams/point-cloud-utils/releases/tag/0.17.1
        idx = pcu.downsample_point_cloud_poisson_disk(x, 0.03)
        x = x[idx]
        n = n[idx]

    # Center the point cloud about its mean and align about its principle components
    x, transform = utils.transform_pointcloud(x, args.device)

    # Generate an initial set of UV samples in the plane
    uv = torch.tensor(
        pcu.lloyd_2d(x.shape[0]).astype(np.float32),
        requires_grad=True,
        device=args.device,
    )

    # Initialize the model for the surface
    phi = MLP(2, 3).to(args.device)

    output_dict["uv"] = uv
    output_dict["x"] = x
    output_dict["transform"] = transform
    
    #plot_uv(uv.detach().cpu().numpy())

    optimizer = torch.optim.Adam(phi.parameters(), lr=args.learning_rate)
    uv_optimizer = torch.optim.Adam([uv], lr=args.learning_rate)
    sinkhorn_loss = SinkhornLoss(
        max_iters=args.max_sinkhorn_iters, return_transport_matrix=True
    )
    mse_loss = nn.MSELoss()

    # Cache correspondences to plot them later
    pi = None

    # Cache model with the lowest loss if --use-best is passed
    best_model = None
    best_loss = np.inf

    for epoch in range(args.local_epochs):
        optimizer.zero_grad()
        uv_optimizer.zero_grad()

        epoch_start_time = time.time()

        y = phi(uv)

        with torch.no_grad():
            if args.loss == "exact_emd":
                M = (
                    pairwise_distances(x.unsqueeze(0), y.unsqueeze(0))
                    .squeeze()
                    .cpu()
                    .squeeze()
                    .numpy()
                )
                p = ot.emd(np.ones(x.shape[0]), np.ones(x.shape[0]), M)
                p = torch.from_numpy(p.astype(np.float32)).to(args.device)
            elif args.loss == "chamfer":
                print()
            elif args.loss == "hausdorff":
                # https://github.com/Project-MONAI/MONAI/blob/dev/monai/metrics/hausdorff_distance.py
                print()
            else:
                _, p = sinkhorn_loss(x.unsqueeze(0), y.unsqueeze(0))
            pi = p.squeeze().max(0)[1]

        loss = mse_loss(x[pi].unsqueeze(0), y.unsqueeze(0))

        loss.backward()

        if args.use_best and loss.item() < best_loss:
            best_loss = loss.item()
            best_model = copy.deepcopy(phi.state_dict())

        epoch_end_time = time.time()

        if epoch % args.print_every == 0:
            print(
                "%d/%d: [Loss = %0.5f] [Time = %0.3f]"
                % (
                    epoch,
                    args.local_epochs,
                    loss.item(),
                    epoch_end_time - epoch_start_time,
                )
            )

        optimizer.step()
        #uv_optimizer.step()

    if args.use_best:
        phi.load_state_dict(best_model)

    output_dict["final_model"] = copy.deepcopy(phi.state_dict())

    torch.save(output_dict, args.output)

    if args.plot:
        #plot_uv(uv.detach().cpu().numpy())
        plot_reconstruction(uv, x, transform, phi, pad=1.0)
        plot_correspondences(phi, uv, x, pi)


if __name__ == "__main__":
    main()
