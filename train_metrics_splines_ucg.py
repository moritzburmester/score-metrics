import torch
import math
import os

from manifolds import (
    RBFManifold,
    LandManifold,
    h_diag_RBF,
    h_diag_Land,
    normalize_metric,
    ScoreBasedManifold
)

from utils.toy_dataset import GaussianMixture
from utils.plotting import plot_geodesics_on_gmm_density
from stochman.geodesic import geodesic_minimizing_energy
from curves import CubicSpline
import torch.optim as optim
from utils.analytical_score import fast_gaussian_score


device = 'cuda'
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
n_curve_points = 100
lr = 5e-3
save_dir_curve = "saved_models_uniform"
curves_path = os.path.join(save_dir_curve, "curves_ucg_spline.pt")

NB_GAUSSIANS = 200
RADIUS = 8

angles = torch.linspace(0, math.pi, NB_GAUSSIANS)
means = RADIUS * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
means = means.to(device)
covar = torch.eye(2).unsqueeze(0).repeat(NB_GAUSSIANS, 1, 1)
weights = torch.ones(NB_GAUSSIANS) / NB_GAUSSIANS
stds = torch.ones(NB_GAUSSIANS, device=device)
weights = weights.to(device)
stds = stds.to(device)
mixture = GaussianMixture(means.to(device), covar.to(device), weight=weights.to(device))
samples = mixture.sample(1000).to(device)


x = torch.linspace(-10, 10, 120)
y = torch.linspace(-2, 10, 80)
X, Y = torch.meshgrid(x, y, indexing='ij')
pos = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)


metrics = {}

# RBF
h_rbf_model = h_diag_RBF(n_centers=30, data_to_fit=samples).to(device)
h_rbf = normalize_metric(h_rbf_model.h, pos)
metrics["RBF"] = RBFManifold(h_rbf)

# LAND
h_land_model = h_diag_Land(samples, gamma=1.0).to(device)
h_land = normalize_metric(h_land_model.h, pos)
metrics["LAND"] = LandManifold(h_land)

# SCORE variants
score_fn = lambda x, t: fast_gaussian_score(
    x.view(1, -1, 2),
    None,
    means,
    stds,
    weights
).view(-1, 2)
metrics["Magnitude"] = ScoreBasedManifold(score_fn, "magnitude", p=4)
metrics["Tangent"] = ScoreBasedManifold(score_fn, "tangent")
metrics["Mixed"] = ScoreBasedManifold(score_fn, "mixed", p=4)

x0 = means[10].unsqueeze(0).to(device)
x1 = means[-10].unsqueeze(0).to(device)
alpha = torch.linspace(0, 1, n_curve_points, device=device).view(1, n_curve_points)  # (1, T)


if os.path.exists(curves_path):
    print(f"Loading saved curves from {curves_path}...")
    curves = torch.load(curves_path)

else:
    curves = {}
    for name, manifold in metrics.items():
        print(f"Computing geodesic with {name}...")
  
        spline = CubicSpline(x0, x1, num_nodes=n_curve_points, requires_grad=True).to(device)
        
        geodesic_minimizing_energy(
            spline,
            manifold,
            optimizer=lambda params, lr=lr: optim.Adam(params, lr=lr),
            max_iter=5000,
            eval_grid=n_curve_points
        )

        curves[name] = spline(alpha).detach()
    torch.save(curves, curves_path)
    print(f"Saved curves to {curves_path}")

plot_geodesics_on_gmm_density(
    curves=curves,
    mixture=mixture,
    X=X,
    Y=Y,
    title="Geodesics on Semicircle Gaussian Mixture (Spline)",
    save_path="plots/gaussian_blobs/exp5/geodesics_semicircle_spline_fixed.png",
    device=device
)