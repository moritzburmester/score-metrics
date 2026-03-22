import torch
import os
from utils.analytical_score import create_alternating_line_gaussian_mixture, fast_gaussian_score
from manifolds import RBFManifold, LandManifold, ScoreBasedManifold, normalize_metric, h_diag_RBF, h_diag_Land
from utils.plotting import plot_geodesics_on_density
from geodesic_interpolant_net import SyntheticPairsDataset, train_geodesic
from stochman.geodesic import geodesic_minimizing_energy
from curves import CubicSpline
import torch.optim as optim

device = 'cuda'
os.makedirs("plots/gaussian_blobs/exp5", exist_ok=True)
save_dir_curve = "saved_models_gaussian"
curves_path = os.path.join(save_dir_curve, "curves_gaussians_spline.pt")

n_curve_points = 100
lr = 5e-3


means, stds, weights = create_alternating_line_gaussian_mixture(device=device)
gaussian_params = {'means': means, 'stds': stds, 'weights': weights}
dataset = SyntheticPairsDataset(
    manifold_type='gaussians',
    n_points=1000,
    device=device,
    gaussian_params=gaussian_params
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

reference_samples = dataset.points


x_grid = torch.linspace(0, 28, 100)
y_grid = torch.linspace(0, 18, 100)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
pos = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)

metrics = {}

# RBF
h_rbf_model = h_diag_RBF(n_centers=30, data_to_fit=reference_samples).to(device)
h_rbf = normalize_metric(h_rbf_model.h, pos)
metrics["rbf"] = RBFManifold(h_rbf)

# LAND
h_land_model = h_diag_Land(reference_samples, gamma=1.0).to(device)
h_land = normalize_metric(h_land_model.h, pos)
metrics["land"] = LandManifold(h_land)


score_fn = lambda x, t: fast_gaussian_score(x.view(1, -1, 2), None, means, stds, weights).view(-1, 2)
metrics["magnitude"] = ScoreBasedManifold(score_fn, "magnitude", p=4)
metrics["tangent"] = ScoreBasedManifold(score_fn, "tangent")
metrics["mixed"] = ScoreBasedManifold(score_fn, "mixed", p=4)


x0 = torch.tensor([2.0, 2.0], device=device).reshape((1, -1))  
x1 = torch.tensor([26.0, 2.0], device=device).reshape((1, -1))  
alpha = torch.linspace(0, 1, n_curve_points, device=device).view(1, n_curve_points) 


curves = {}

if os.path.exists(curves_path):
    print(f"Loading saved curves from {curves_path}...")
    curves = torch.load(curves_path)
else:
    for name, manifold in metrics.items():

        print(f"Optimizing {name}...")
  
        spline = CubicSpline(x0, x1, num_nodes=n_curve_points, requires_grad=True).to(device)
        

        opt = optim.Adam(spline.parameters(), lr=lr)

        geodesic_minimizing_energy(spline, manifold, optimizer=lambda params, lr=lr: optim.Adam(params, lr=lr), max_iter=5000, eval_grid=n_curve_points)

        curves[name] = spline(alpha).detach()

    torch.save(curves, curves_path)
    print(f"Saved curves to {curves_path}")


plot_geodesics_on_density(
    curves,
    means,
    stds,
    weights,
    title="Single Geodesic under Different Metrics (Cubic Spline)",
    save_path="plots/gaussian_blobs/exp5/geodesics_spline_curve_fixed.png",
    device=device
)