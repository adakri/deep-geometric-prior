import src.utils as utils
import matplotlib.pyplot as plt
import torch
import numpy as np
import open3d as o3d


# Mono
def plot_mesh_wscalarf(v,f,scalar_field_np):
    assert v.shape[0] == scalar_field_np.shape[0]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.compute_vertex_normals()
    scalar_field_normalized = (scalar_field_np - scalar_field_np.min()) / (scalar_field_np.max() - scalar_field_np.min())
    cmap = plt.get_cmap('jet')
    colors = cmap(scalar_field_normalized)[:, :3]  # Get RGB values from colormap
    print(f"Scalar field range: {scalar_field_np.min()} to {scalar_field_np.max()}")
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Options
    mesh_show_back_face = True
    mesh_show_wireframe = True
    o3d.visualization.draw_geometries([mesh], 
                                      mesh_show_back_face=mesh_show_back_face,
                                      mesh_show_wireframe=mesh_show_wireframe)


def plot_reconstruction(uv, model, x=None, transform=None, pad=1.0, scalar_field_func: callable=None, n: int=128):
    # If n is provided, upsample the uv grid to an n by n grid for visualization. 
    # Otherwise, use the original uv grid (if the scalar field is evaluated seperately for example).
    with torch.no_grad():
        uv_dense = utils.meshgrid_from_lloyd_ts(
            uv.cpu().numpy(), n, scale=pad
        ).astype(np.float32)
        uv_dense = torch.from_numpy(uv_dense).to(uv)
        mesh_f = utils.meshgrid_face_indices(n)
        y_dense = model(uv_dense)
        mesh_v = y_dense.squeeze().cpu().numpy()
        
    if(scalar_field_func is not None):
        scalar_field = scalar_field_func(model, uv_dense)
        scalar_field = scalar_field.detach().cpu().numpy()
        
        # Debug
        import igl
        o = igl.principal_curvature(mesh_v, mesh_f)
        v1, v2, k1, k2 = o[0], o[1], o[2], o[3]
        scalar_field = k1*k2
    else:
        scalar_field = None

    if(x is not None):
        # ---- Point cloud (input x) ----
        with torch.no_grad():
            x_np = x.squeeze().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x_np)
        pcd.paint_uniform_color([0.8, 0.2, 0.2])

    # ---- Reconstructed mesh ----
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_v)
    # Duplicate inverse triangles (open3d back face culling)
    #triangles = mesh_f.astype(np.int32)
    #triangles_flipped = triangles[:, [0, 2, 1]]  
    #all_triangles = np.vstack([triangles, triangles_flipped])
    #mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    
    mesh.triangles = o3d.utility.Vector3iVector(mesh_f.astype(np.int32))
    mesh.compute_vertex_normals()
    
    if(scalar_field is not None):
        # Map scalar field to colors
        if(torch.is_tensor(scalar_field)):
            scalar_field_np = scalar_field.detach().squeeze().cpu().numpy()
        else:
            scalar_field_np = scalar_field
        scalar_field_normalized = (scalar_field_np - scalar_field_np.min()) / (scalar_field_np.max() - scalar_field_np.min())
        cmap = plt.get_cmap('jet')
        colors = cmap(scalar_field_normalized)[:, :3]  # Get RGB values from colormap
        print(f"Scalar field range: {scalar_field_np.min()} to {scalar_field_np.max()}")
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        assert len(mesh.vertices) == colors.shape[0]
    else:
        mesh.paint_uniform_color([0.2, 0.2, 0.8])

    # Options
    mesh_show_back_face = True
    mesh_show_wireframe = True

    if(x is not None):
        o3d.visualization.draw_geometries([mesh, pcd], mesh_show_back_face=mesh_show_back_face,mesh_show_wireframe=mesh_show_wireframe)
    else:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=mesh_show_back_face,mesh_show_wireframe=mesh_show_wireframe)


def plot_correspondences(model, uv, x, pi):

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
    import colorsys
    
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

 