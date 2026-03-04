import src.utils as utils

# Mono

def plot_reconstruction(uv, x, transform, model, pad=1.0):
    import torch
    import numpy as np
    import open3d as o3d

    with torch.no_grad():
        n = 128
        translate, scale, rotate = transform
        uv_dense = utils.meshgrid_from_lloyd_ts(
            uv.cpu().numpy(), n, scale=pad
        ).astype(np.float32)
        uv_dense = torch.from_numpy(uv_dense).to(uv)
        y_dense = model(uv_dense)
        x_np = x.squeeze().cpu().numpy()
        mesh_v = y_dense.squeeze().cpu().numpy()
        mesh_f = utils.meshgrid_face_indices(n)

    # ---- Point cloud (input x) ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x_np)
    pcd.paint_uniform_color([0.8, 0.2, 0.2])

    # ---- Reconstructed mesh ----
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_v)
    # Duplicate inverse triangles (open3d back face culling)
    triangles = mesh_f.astype(np.int32)
    triangles_flipped = triangles[:, [0, 2, 1]]  
    all_triangles = np.vstack([triangles, triangles_flipped])

    mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    #mesh.triangles = o3d.utility.Vector3iVector(mesh_f.astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.2, 0.2, 0.8])

    o3d.visualization.draw_geometries([pcd, mesh])

def plot_correspondences(model, uv, x, pi):
    import torch
    import numpy as np
    import open3d as o3d

    with torch.no_grad():
        y = model(uv).detach().squeeze().cpu().numpy()
        x_np = x.detach().squeeze().cpu().numpy()
        x_corr = x[pi].detach().squeeze().cpu().numpy()

    # ---- Source points (red) ----
    pcd_x = o3d.geometry.PointCloud()
    pcd_x.points = o3d.utility.Vector3dVector(x_np)
    pcd_x.paint_uniform_color([1.0, 0.0, 0.0])

    # ---- Target points (green) ----
    pcd_y = o3d.geometry.PointCloud()
    pcd_y.points = o3d.utility.Vector3dVector(y)
    pcd_y.paint_uniform_color([0.0, 1.0, 0.0])

    # ---- Correspondence lines ----
    points = []
    lines = []

    for i in range(x_corr.shape[0]):
        points.append(x_corr[i])
        points.append(y[i])
        lines.append([2 * i, 2 * i + 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.paint_uniform_color([0.1, 0.1, 0.1])

    o3d.visualization.draw_geometries([pcd_x, pcd_y, line_set])

def plot_uv(uv):
    import matplotlib.pyplot as plt
    # Plot
    plt.figure()
    plt.scatter(uv[:, 0], uv[:, 1], s=10)

    # Regular grid appearance
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    plt.xlabel("U")
    plt.ylabel("V")
    plt.title("UV Positions on Regular Grid")
    plt.show()

# Multi

def plot_batch_reconstruction(x, patch_uvs, patch_tx, patch_models, scale=1.0):
    """
    Plot a dense, upsampled point cloud

    :param x: A [n, 3] tensor containing the input point cloud
    :param patch_uvs: A list of tensors, each of shape [n_i, 2] of UV positions for the given patch
    :param patch_tx: A list of tuples (t_i, s_i, r_i) of transformations (t_i is a translation, s_i is a scaling, and
                     r_i is a rotation matrix) which map the points in a neighborhood to a centered and whitened point
                     set
    :param patch_models: A list of neural networks representing the lifting function for each chart in the atlas
    :param scale: Scale parameter to sample uv values from a smaller or larger subset of [0, 1]^2 (i.e. scale*[0, 1]^2)
    :return: A list of tensors, each of shape [n_i, 3] where each tensor is the average prediction of the overlapping
             charts a the samples
    """
    import torch
    import numpy as np
    import open3d as o3d
    import colorsys

    with torch.no_grad():
        meshes_list = []
        x_np = x.squeeze().cpu().numpy()
        for i in range(len(patch_models)):
            uv = patch_uvs[i]
            n = 128
            translate, scale, rotate = patch_tx[i]
            uv_dense = utils.meshgrid_from_lloyd_ts(
                uv.cpu().numpy(), n, scale=scale.cpu().numpy()).astype(np.float32)
            uv_dense = torch.from_numpy(uv_dense).to(uv)
            
            y_dense = patch_models[i](uv_dense)
            
            x_np = x.squeeze().cpu().numpy()
            mesh_v = mesh_v = ((y_dense.squeeze() @ rotate.transpose(0, 1)) / scale - translate).cpu().numpy()
            mesh_f = utils.meshgrid_face_indices(n)
            
            # ---- Reconstructed mesh ----
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(mesh_v)
            # Duplicate inverse triangles (open3d back face culling)
            triangles = mesh_f.astype(np.int32)
            triangles_flipped = triangles[:, [0, 2, 1]]  
            all_triangles = np.vstack([triangles, triangles_flipped])

            mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
            mesh.compute_vertex_normals()
            # Evenly spaced hue
            hue = i / len(patch_models)
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = tuple(rgb)
            mesh.paint_uniform_color(list(color))
            meshes_list. append(mesh)
        # ---- Point cloud (input x) ----
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x_np)
        pcd.paint_uniform_color([0.8, 0.2, 0.2])

        o3d.visualization.draw_geometries([pcd] + meshes_list)
        
        
def plot_batch_patches(x, patch_idx):
    """
    Plot the points in each neighborhood in a different color.

    :param x: A [n, 3] tensor containing the input point cloud
    :param patch_idx: List of [n_i]-shaped tensors each indexing into x representing the points in a given neighborhood.
    """
    import numpy as np
    import colorsys
    import open3d as o3d
    
    pcd_list = []
    for i,idx_i in enumerate(patch_idx):
        # Evenly spaced hue
        hue = i / len(patch_idx)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color = tuple(rgb)
        sf = 0.1 + np.random.randn()*0.05
        x_i = x[idx_i]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x_i)
        pcd.paint_uniform_color(list(color))
        pcd_list.append(pcd)
        #o3d.visualization.draw_geometries([pcd])
        
    o3d.visualization.draw_geometries(pcd_list)

 