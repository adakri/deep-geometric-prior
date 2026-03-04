import argparse
import copy
import time

from matplotlib import mlab
import numpy as np
import ot
import point_cloud_utils as pcu
import torch
import torch.nn as nn
from fml.nn import SinkhornLoss, pairwise_distances

import src.utils as utils
from src.nns import MLP
from src.viz import plot_batch_patches, plot_batch_reconstruction


 
def move_optimizer_to_device(optimizer, device):
    state_devices = {}
    for i, state in enumerate(optimizer.state.values()):
        for k, v in state.items():
            if torch.is_tensor(v):
                key = k + "-" + str(i)
                dev = device[key] if isinstance(device, dict) else device
                state_devices[key] = v.device
                state[k] = v.to(dev)
    return state_devices


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mesh_filename", "-mf", type=str, help="Point cloud to reconstruct")
    argparser.add_argument("--radius", "-r", type=float, help="Patch radius (The parameter, r, in the paper)")
    argparser.add_argument("--padding", "-p", type=float, help="Padding factor for patches (The parameter, c, in the paper)")
    argparser.add_argument("--min_pts_per_patch", "-mpp", type=int,
                           help="Minimum number of allowed points inside a patch used to not fit to "
                                "patches with too little data")
    argparser.add_argument("--output", "-o", type=str, default="out",
                           help="Name for the output files: e.g. if you pass in --output out, the program will save "
                                "a dense upsampled point-cloud named out.ply, and a file containing reconstruction "
                                "metadata and model weights named out.pt. Default: out -- "
                                "Note: the number of points per patch in the upsampled point cloud is 64 by default "
                                "and can be set by specifying --upsamples-per-patch.")
    argparser.add_argument("--upsamples-per-patch", "-nup", type=int, default=8,
                           help="*Square root* of the number of upsamples per patch to generate in the output. i.e. if "
                                "you pass in --upsamples-per-patch 8, there will be 64 upsamples per patch.")
    argparser.add_argument("--angle-threshold", "-a", type=float, default=95.0,
                           help="Threshold (in degrees) used to discard points in "
                                "a patch whose normal is facing the wrong way.")
    argparser.add_argument("--local-epochs", "-nl", type=int, default=25,
                           help="Number of fitting iterations done for each chart to its points")
    argparser.add_argument("--global-epochs", "-ng", type=int, default=25,
                           help="Number of fitting iterations done to make each chart agree "
                                "with its neighboring charts")
    argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3,
                           help="Step size for gradient descent.")
    argparser.add_argument("--devices", "-d", type=str, default=["cuda"], nargs="+",
                           help="A list of devices on which to partition the models for each patch. For large inputs, "
                                "reconstruction can be memory and compute intensive. Passing in multiple devices will "
                                "split the load across these. E.g. --devices cuda:0 cuda:1 cuda:2")
    argparser.add_argument("--plot", action="store_true",
                           help="Plot the following intermediate states:. (1) patch neighborhoods, "
                                "(2) Intermediate reconstruction before global consistency step, "
                                "(3) Reconstruction after global consistency step. "
                                "This flag is useful for debugging but does not scale well to large inputs.")
    argparser.add_argument("--interpolate", action="store_true",
                           help="If set, then force all patches to agree with the input at overlapping points "
                                "(i.e. the reconstruction will try to interpolate the input point cloud). "
                                "Otherwise, we fit all patches to the average of overlapping patches at each point.")
    argparser.add_argument("--max-sinkhorn-iters", "-si", type=int, default=32,
                           help="Maximum number of Sinkhorn iterations")
    argparser.add_argument("--sinkhorn-epsilon", "-sl", type=float, default=1e-3,
                           help="The reciprocal (1/lambda) of the Sinkhorn regularization parameter.")
    argparser.add_argument("--seed", "-s", type=int, default=-1,
                           help="Random seed to use when initializing network weights. "
                                "If the seed not positive, a seed is selected at random.")
    argparser.add_argument( "--loss", type=str, default="sinkhorn", choices=["sinkhorn", "emd", "chamfer", "hausdorff"], 
                           help="Loss function to use")

    argparser.add_argument("--use-best", action="store_true",
                           help="Use the model with the lowest loss as the final result.")
    argparser.add_argument("--normal-neighborhood-size", "-ns", type=int, default=64,
                           help="Neighborhood size used to estimate the normals in the final dense point cloud. "
                                "Default: 64")
    argparser.add_argument("--save-pre-cc", action="store_true",
                           help="Save a copy of the model before the cycle consistency step")
    argparser.add_argument("--batch-size", type=int, default=-1, help="Split fitting MLPs into batches")
    args = argparser.parse_args()

    # We'll populate this dictionary and save it as output
    output_dict = {
        "pre_cycle_consistency_model": None,
        "final_model": None,
        "patch_uvs": None,
        "patch_idx": None,
        "patch_txs": None,
        "radius": args.radius,
        "padding": args.padding,
        "min_pts_per_patch": args.min_pts_per_patch,
        "angle_threshold": args.angle_threshold,
        "interpolate": args.interpolate,
        "global_epochs": args.global_epochs,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "devices": args.devices,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "loss": args.loss,
        "max_sinkhorn_iters": args.max_sinkhorn_iters,
        "seed": utils.seed_everything(args.seed),
        "batch_size": args.batch_size
    }

    # Read a point cloud and normals from a file, center it about its mean, and align it along its principle vectors
    x, n = utils.load_point_cloud_by_file_extension(args.mesh_filename, compute_normals=True)
    n = n[1]
    # Optionally, we downsample
    if(True):
        idx = pcu.downsample_point_cloud_poisson_disk(x, 0.01)
        x = x[idx]
        n = n[idx]
    
    # Compute a set of neighborhood (patches) and a uv samples for each neighborhood. Store the result in a list
    # of pairs (uv_j, xi_j) where uv_j are 2D uv coordinates for the j^th patch, and xi_j are the indices into x of
    # the j^th patch. We will try to reconstruct a function phi, such that phi(uv_j) = x[xi_j].
    print("Computing neighborhoods...")
    bbox_diag = np.linalg.norm(np.max(x, axis=0) - np.min(x, axis=0))
    patch_idx, patch_uvs, patch_xs, patch_tx = utils.compute_patches(x, n, args.radius*bbox_diag, args.padding,
                                                               angle_thresh=args.angle_threshold,
                                                               min_pts_per_patch=args.min_pts_per_patch)
    num_patches = len(patch_uvs)
    output_dict["patch_uvs"] = patch_uvs
    output_dict["patch_idx"] = patch_idx
    output_dict["patch_txs"] = patch_tx

    if args.plot:
        plot_batch_patches(x, patch_idx)

    # Initialize one model per patch and convert the input data to a pytorch tensor
    print("Creating models...")
    if args.batch_size > 0:
        num_batches = int(np.ceil(num_patches / args.batch_size))
        batch_size = args.batch_size
        print("Splitting fitting into %d batches" % num_batches)
    else:
        num_batches = 1
        batch_size = num_patches
    phi = nn.ModuleList([MLP(2, 3) for i in range(num_patches)])
    # Move to device
    x = torch.from_numpy(x.astype(np.float32)).to('cuda')
    
    phi_optimizers = []
    phi_optimizers_devices = []
    uv_optimizer = torch.optim.Adam(patch_uvs, lr=args.learning_rate)
    sinkhorn_loss = SinkhornLoss(max_iters=args.max_sinkhorn_iters, return_transport_matrix=True)
    mse_loss = nn.MSELoss()

    # Fit a function, phi_i, for each patch so that phi_i(patch_uvs[i]) = x[patch_idx[i]]. i.e. so that the function
    # phi_i "agrees" with the point cloud on each patch.
    #
    # We also store the correspondences between the uvs and points which we use later for the consistency step. The
    # correspondences are stored in a list, pi where pi[i] is a vector of integers used to permute the points in
    # a patch.
    pi = [None for _ in range(num_patches)]

    # Cache model with the lowest loss if --use-best is passed
    best_models = [None for _ in range(num_patches)]
    best_losses = [np.inf for _ in range(num_patches)]

    print("Training local patches...")
    for b in range(num_batches):
        print("Fitting batch %d/%d" % (b + 1, num_batches))
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, num_patches)
        optimizer_batch = torch.optim.Adam(phi[start_idx:end_idx].parameters(), lr=args.learning_rate)
        phi_optimizers.append(optimizer_batch)
        for i in range(start_idx, end_idx):
            dev_i = args.devices[i % len(args.devices)]
            phi[i] = phi[i].to(dev_i)
            patch_uvs[i] = patch_uvs[i].to(dev_i)
            patch_xs[i] = patch_xs[i].to(dev_i)
            
        for epoch in range(args.local_epochs):
            optimizer_batch.zero_grad()
            uv_optimizer.zero_grad()

            # sum_loss = torch.tensor([0.0]).to(args.devices[0])
            losses = []
            torch.cuda.synchronize()
            epoch_start_time = time.time()
            for i in range(start_idx, end_idx):
                uv_i = patch_uvs[i]
                x_i = patch_xs[i]
                y_i = phi[i](uv_i)

                with torch.no_grad():
                    if args.loss == "exact_emd":
                        M_i = pairwise_distances(x_i.unsqueeze(0), y_i.unsqueeze(0)).squeeze().cpu().squeeze().numpy()
                        p_i = ot.emd(np.ones(x_i.shape[0]), np.ones(y_i.shape[0]), M_i)
                        p_i = torch.from_numpy(p_i.astype(np.float32)).to(args.devices[0])
                    elif args.loss == "chamfer":
                        print()
                    elif args.loss == "hausdorff":
                        # https://github.com/Project-MONAI/MONAI/blob/dev/monai/metrics/hausdorff_distance.py
                        print()
                    else:
                        _, p_i = sinkhorn_loss(x_i.unsqueeze(0), y_i.unsqueeze(0))
                    pi_i = p_i.squeeze().max(0)[1]
                    pi[i] = pi_i

                loss_i = mse_loss(x_i[pi_i].unsqueeze(0), y_i.unsqueeze(0))

                if args.use_best and loss_i.item() < best_losses[i]:
                    best_losses[i] = loss_i.item()
                    model_copy = copy.deepcopy(phi[i]).to('cpu')
                    best_models[i] = copy.deepcopy(model_copy.state_dict())
                loss_i.backward()
                losses.append(loss_i)
                # sum_loss += loss_i.to(args.devices[0])

            # sum_loss.backward()
            sum_loss = sum([l.item() for l in losses])
            torch.cuda.synchronize()
            epoch_end_time = time.time()

            print("%d/%d: [Total = %0.5f] [Mean = %0.5f] [Time = %0.3f]" %
                  (epoch, args.local_epochs, sum_loss,
                   sum_loss / (end_idx - start_idx), epoch_end_time - epoch_start_time))
            optimizer_batch.step()
            uv_optimizer.step()
            
        for i in range(start_idx, end_idx):
            dev_i = 'cpu'
            phi[i] = phi[i].to(dev_i)
            patch_uvs[i] = patch_uvs[i].to(dev_i)
            patch_xs[i] = patch_xs[i].to(dev_i)
            pi[i] = pi[i].to(dev_i)
        optimizer_batch_devices = move_optimizer_to_device(optimizer_batch, 'cpu')
        phi_optimizers_devices.append(optimizer_batch_devices)
                    
        print("Done batch %d/%d" % (b + 1, num_batches))

    print("Mean best losses:", np.mean(best_losses[i]))
    
    if args.use_best:
        for i, phi_i in enumerate(phi):
            phi_i.load_state_dict(best_models[i])

    if args.save_pre_cc:
        output_dict["pre_cycle_consistency_model"] = copy.deepcopy(phi.state_dict())

    if args.plot:
        plot_batch_reconstruction(x, patch_uvs, patch_tx, phi, scale=1.0/args.padding)

    # Do a second, global, stage of fitting where we ask all patches to agree with each other on overlapping points.
    # If the user passed --interpolate, we ask that the patches agree on the original input points, otherwise we ask
    # that they agree on the average of predictions from patches overlapping a given point.
    if not args.interpolate:
        print("Computing patch means...")
        with torch.no_grad():
            patch_xs = utils.patch_means(pi, patch_uvs, patch_idx, patch_tx, phi, x, args.devices, num_batches)

    print("Training cycle consistency...")
    for b in range(num_batches):
        print("Fitting batch %d/%d" % (b + 1, num_batches))
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, num_patches)
        for i in range(start_idx, end_idx):
            dev_i = args.devices[i % len(args.devices)]
            phi[i] = phi[i].to(dev_i)
            patch_uvs[i] = patch_uvs[i].to(dev_i)
            patch_xs[i] = patch_xs[i].to(dev_i)
            pi[i] = pi[i].to(dev_i)
        optimizer = phi_optimizers[b]
        move_optimizer_to_device(optimizer, phi_optimizers_devices[b])
        for epoch in range(args.global_epochs):
            optimizer.zero_grad()
            uv_optimizer.zero_grad()

            sum_loss = torch.tensor([0.0]).to(args.devices[0])
            epoch_start_time = time.time()
            for i in range(start_idx, end_idx):
                uv_i = patch_uvs[i]
                x_i = patch_xs[i]
                y_i = phi[i](uv_i)
                pi_i = pi[i]
                loss_i = mse_loss(x_i[pi_i].unsqueeze(0), y_i.unsqueeze(0))

                if loss_i.item() < best_losses[i]:
                    best_losses[i] = loss_i.item()
                    model_copy = copy.deepcopy(phi[i]).to('cpu')
                    best_models[i] = copy.deepcopy(model_copy.state_dict())

                sum_loss += loss_i.to(args.devices[0])

            sum_loss.backward()
            epoch_end_time = time.time()

            print("%d/%d: [Total = %0.5f] [Mean = %0.5f] [Time = %0.3f]" %
                  (epoch, args.global_epochs, sum_loss.item(),
                   sum_loss.item() / (end_idx - start_idx), epoch_end_time-epoch_start_time))
            optimizer.step()
            uv_optimizer.step()
        for i in range(start_idx, end_idx):
            dev_i = 'cpu'
            phi[i] = phi[i].to(dev_i)
            patch_uvs[i] = patch_uvs[i].to(dev_i)
            patch_xs[i] = patch_xs[i].to(dev_i)
            pi[i] = pi[i].to(dev_i)
        move_optimizer_to_device(optimizer, 'cpu')
                    
    print("Mean best losses:", np.mean(best_losses[i]))
    for i, phi_i in enumerate(phi):
        phi_i.load_state_dict(best_models[i])

    output_dict["final_model"] = phi.state_dict()

    print("Generating dense point cloud...")
    v, n = utils.upsample_surface(patch_uvs, patch_tx, phi, args.devices,
                            scale=(1.0/args.padding),
                            num_samples=args.upsamples_per_patch,
                            normal_samples=args.normal_neighborhood_size,
                            num_batches=num_batches,
                            compute_normals=False)

    print("Saving dense point cloud...")
    pcu.save_mesh_vn(args.output + ".ply", v, n[1])

    print("Saving metadata...")
    torch.save(output_dict, args.output + ".pt")

    if args.plot:
        plot_batch_reconstruction(x, patch_uvs, patch_tx, phi, scale=1.0/args.padding)


if __name__ == "__main__":
    main()
