import torch
import numpy as np
from src import geom
import src.utils as utils
from src.viz import plot_reconstruction, plot_mesh_wscalarf
import igl

def sphere_surface(R=0.3, n=200, device="cpu"):
    """
    Generates a sphere parameterization φ(u,v)
    with high Gaussian curvature.
    """

    u = torch.linspace(1e-3, np.pi - 1e-3, n, device=device)
    v = torch.linspace(0, 2*np.pi, n, device=device)

    U, V = torch.meshgrid(u, v, indexing="ij")
    uv = torch.stack([U.reshape(-1), V.reshape(-1)], dim=-1)

    def phi(uv):
        u = uv[0]
        v = uv[1]
        x = R * torch.sin(u) * torch.cos(v)
        y = R * torch.sin(u) * torch.sin(v)
        z = R * torch.cos(u)
        return torch.stack([x,y,z])
    
    def phi_plot(uv):
        u = uv[:,0]
        v = uv[:,1]
        x = R * torch.sin(u) * torch.cos(v)
        y = R * torch.sin(u) * torch.sin(v)
        z = R * torch.cos(u)
        return torch.stack([x,y,z], dim=-1)

    return uv, phi, phi_plot

n = 200
uv, phi, phi_batch = sphere_surface(R=0.3, n=n)

# Compute mesh gaussian curfacture using libIGL
y_dense = phi_batch(uv)
mesh_v = y_dense.squeeze().cpu().numpy()
mesh_f = utils.meshgrid_face_indices(n)
o = igl.principal_curvature(mesh_v, mesh_f)
v1, v2, k1, k2 = o[0], o[1], o[2], o[3]

plot_mesh_wscalarf(mesh_v, mesh_f, k1*k2)

plot_reconstruction(uv=uv, x=None, transform=None, model=phi_batch, pad=1.0, scalar_field=k1*k2)


# Compute gaussian curvature using autograd
Kf = geom.gaussian_curvature_fundamental(phi, uv)
Kp = geom.gaussian_curvature_det(phi, uv)

print("Gaussian curvature (fundamental): ", Kf.mean().item())
print("Gaussian curvature (det): ", Kp.mean().item())

plot_reconstruction(uv=uv, x=None, transform=None, model=phi_batch, pad=1.0, scalar_field=Kf)
plot_reconstruction(uv=uv, x=None, transform=None, model=phi_batch, pad=1.0, scalar_field=Kp)