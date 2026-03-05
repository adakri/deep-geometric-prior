
import torch

# Geometry measures
def surface_area(v, f):
    """
    Compute the surface area of the batch of triangle meshes defined by v and f
    :param v: A [b, nv, 3] tensor where each [i, :, :] are the vertices of a mesh
    :param f: A [b, nf, 3] tensor where each [i, :, :] are the triangle indices into v[i, :, :] of the mesh
    :return: A tensor of shape [b, 1] with the surface area of each mesh
    """
    idx = torch.arange(v.shape[0])
    tris = v[:, f, :][idx, idx, :, :]
    a = tris[:, :, 1, :] - tris[:, :, 0, :]
    b = tris[:, :, 2, :] - tris[:, :, 0, :]
    areas = torch.sum(torch.norm(torch.cross(a, b, dim=2), dim=2)/2.0, dim=1)
    return areas


def arclength(x):
    """
    Compute the arclength of a curve sampled at a sequence of points.
    :param x: A [b, n, d] tensor of minibaches of d-dimensional point sequences.
    :return: A tensor of shape [b] where each entry, i, is estimated arclength for the curve samples x[i, :, :]
    """
    v = x[:, 1:, :] - x[:, :-1, :]
    return torch.norm(v, dim=2).sum(1)


def curvature_2d(x):
    """
    Compute the discrete curvature for a sequence of points on a curve lying in a 2D embedding space
    :param x: A [b, n, 2] tensor where each [i, :, :] is a sequence of n points lying along some curve
    :return: A [b, n-2] tensor where each [i, j] is the curvature at the point x[i, j+1, :]
    """
    # x has shape [b, n, d]
    b = x.shape[0]
    n_x = x.shape[1]
    n_v = n_x - 2

    v = x[:, 1:, :] - x[:, :-1, :]                         # v_i = x_{i+1} - x_i
    v_norm = torch.norm(v, dim=2)
    v = v / v_norm.view(b, n_x-1, 1)                       # v_i = v_i / ||v_i||
    v1 = v[:, :-1, :].contiguous().view(b * n_v, 1, 2)
    v2 = v[:, 1:, :].contiguous().view(b * n_v, 1, 2)
    c_c = torch.bmm(v1, v2.transpose(1, 2)).view(b, n_v)   # \theta_i = <v_i, v_i+1>
    return torch.acos(torch.clamp(c_c, min=-1.0, max=1.0)) / v_norm[:, 1:]


def normals_curve_2d(x):
    """
    Compute approximated normals for a sequence of point samples along a curve in 2D.
    :param x: A tensor of shape [b, n, 2] where each x[i, :, :] is a sequence of n 2d point samples on a curve
    :return: A tensor of shape [b, n, 2] where each [i, j, :] is the estimated normal for point x[i, j, :]
    """
    b = x.shape[0]
    n_x = x.shape[1]

    n = torch.zeros(x.shape)
    n[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    n[:, -1, :] = (x[:, -1, :] - x[:, -2, :])
    n = n[:, :, [1, 0]]
    n[:, :, 0] = -n[:, :, 0]
    n = n / torch.norm(n, dim=2).view(b, n_x, 1)
    n[:, 1:, :] = 0.5*(n[:, 1:, :] + n[:, :-1, :])
    n = n / torch.norm(n, dim=2).view(b, n_x, 1)
    return n

import torch

def gaussian_curvature_fundamental(phi, uv):
    """
    phi : function (N,2) -> (N,3)
    uv  : (N,2) tensor
    returns
    K : (N,) Gaussian curvature
    """
    from torch.func import jacrev, vmap
    from torch.func import hessian
    # first derivatives
    J = vmap(jacrev(phi))(uv)   # (N,3,2)
    Xu = J[:,:,0]
    Xv = J[:,:,1]
    # second derivatives
    H = vmap(hessian(phi))(uv)  # (N,3,2,2)
    Xuu = H[:,:,0,0]
    Xuv = H[:,:,0,1]
    Xvv = H[:,:,1,1]

    # normal
    cross = torch.cross(Xu, Xv, dim=1)
    n = cross / (torch.norm(cross, dim=1, keepdim=True) + 1e-8)
    # first fundamental form
    E = (Xu * Xu).sum(-1)
    F = (Xu * Xv).sum(-1)
    G = (Xv * Xv).sum(-1)
    # second fundamental form
    e = (n * Xuu).sum(-1)
    f = (n * Xuv).sum(-1)
    g = (n * Xvv).sum(-1)

    denom = E * G - F * F + 1e-12
    K = (e * g - f * f) / denom
    return K


def gaussian_curvature_det(phi, uv):
    from torch.func import jacrev, vmap
    
    def det3(a, b, c):
        M = torch.stack([a, b, c], dim=-1)
        return torch.det(M)

    # first derivatives
    from torch.func import hessian
    J = vmap(jacrev(phi))(uv)
    Xu = J[:,:,0]
    Xv = J[:,:,1]
    # second derivatives
    H = vmap(hessian(phi))(uv)  # (N,3,2,2)
    Xuu = H[:,:,0,0]
    Xuv = H[:,:,0,1]
    Xvv = H[:,:,1,1]
    A = det3(Xu, Xv, Xuu)
    B = det3(Xu, Xv, Xvv)
    C = det3(Xu, Xv, Xuv)

    cross = torch.cross(Xu, Xv, dim=1)
    denom = torch.norm(cross, dim=1)**4 + 1e-12

    K = (A * B - C**2) / denom
    return K
