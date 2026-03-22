import torch
import numpy as np
import os
from utils.analytical_score import create_alternating_line_gaussian_mixture, fast_gaussian_score
from geodesic_interpolant_net import SyntheticPairsDataset, train_geodesic
from manifolds import (
    RBFManifold,
    LandManifold,
    h_diag_RBF,
    h_diag_Land,
    normalize_metric,
    ScoreBasedManifold
)
from utils.plotting import plot_geodesics_on_density, plot_score_vector_field_gaussian, plot_true_mixture_score_magnitude, plot_geodesic_on_score_analytic


device = 'cuda'
save_dir = "saved_models_gaussian"
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


metrics = {}

# RBF
h_rbf_model = h_diag_RBF(
    n_centers=30,
    data_to_fit=reference_samples
).to(device)

x = torch.linspace(0, 28, 100)
y = torch.linspace(0, 18, 100)
X, Y = torch.meshgrid(x, y, indexing='ij')
pos = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)

h_rbf = normalize_metric(h_rbf_model.h, pos)
metrics["rbf"] = RBFManifold(h_rbf)

# LAND
h_land_model = h_diag_Land(
    reference_samples,
    gamma=1.0
).to(device)

h_land = normalize_metric(h_land_model.h, pos)
metrics["land"] = LandManifold(h_land)

# SCORE variants 
score_fn = lambda x, t: fast_gaussian_score(
    x.view(1,-1,2), None, means, stds, weights
).view(-1,2)

metrics["magnitude"] = ScoreBasedManifold(score_fn, "magnitude", p=1)
metrics["tangent"] = ScoreBasedManifold(score_fn, "tangent")
metrics["mixed"] = ScoreBasedManifold(score_fn, "mixed", p=1)


trained_models = {}

for name, manifold in metrics.items():
    model_path = os.path.join(save_dir, f"{name}_geodesic_gaussian.pt")
    if os.path.exists(model_path):
        print(f"Loading saved model for {name}...")
        model = torch.load(model_path, map_location=device)
    else: 
        print(f"\nTraining {name} metric...")

        model = train_geodesic(
            dataloader=dataloader,
            manifold=manifold,
            D=2,
            n_points=100,
            n_steps=5000,
            lr=5e-3,
            device=device
        )
        torch.save(model, model_path)
        print(f"Saved {name} model to {model_path}")
    trained_models[name] = model

x0 = torch.tensor([[2, 2]], device=device)
x1 = torch.tensor([[26, 2]], device=device)

n_curve_points = 100
t_vals = torch.linspace(0,1,n_curve_points, device=device).view(1,n_curve_points,1)

x0_exp = x0.repeat(1,n_curve_points,1)
x1_exp = x1.repeat(1,n_curve_points,1)

curves = {}

for name, model in trained_models.items():

    c_flat = model(
        x0_exp.view(-1,2),
        x1_exp.view(-1,2),
        t_vals.view(-1,1)
    )

    curve = (1-t_vals)*x0_exp + t_vals*x1_exp + \
            2*t_vals*(1-t_vals)*c_flat.view(1,n_curve_points,2)

    curves[name] = curve


plot_geodesics_on_density(
    curves,
    means,
    stds,
    weights,
    title="Geodesics under Different Metrics (Density Landscape)",
    save_path="plots/gaussian_blobs/exp5/geodesics_density_all.png",
    device=device
)