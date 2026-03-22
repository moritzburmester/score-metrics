import torch
import math
import os

from geodesic_interpolant_net import SyntheticPairsDataset, train_geodesic
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

device = 'cuda'
save_dir = "saved_models_weighted"
os.makedirs(save_dir, exist_ok=True)

# =========================================================
# DATASET (WEIGHTED SEMICIRCLE GAUSSIAN MIXTURE)
# =========================================================
NB_GAUSSIANS = 200
RADIUS = 8

angles = torch.linspace(0, math.pi, NB_GAUSSIANS)
means = RADIUS * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
covar = torch.eye(2).unsqueeze(0).repeat(NB_GAUSSIANS, 1, 1)

# --- weighted semicircle (like original non-uniform) ---
linspace = torch.cat([torch.linspace(1,30,90), torch.ones(11)*30])
weights = torch.cat([linspace, linspace[1:-1].flip(0)])
weights = weights / weights.sum()

mixture = GaussianMixture(
    means.to(device),
    covar.to(device),
    weight=weights.to(device)
)

# sample training data
samples = mixture.sample(1000).to(device)

dataset = SyntheticPairsDataset(
    manifold_type='gaussians',
    n_points=1000,
    device=device,
    gaussian_params={
        "means": means.to(device),
        "stds": torch.ones(NB_GAUSSIANS, device=device),
        "weights": weights.to(device)
    }
)


dataset.points = samples
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)


x = torch.linspace(-10, 10, 120)
y = torch.linspace(-2, 10, 80)
X, Y = torch.meshgrid(x, y, indexing='ij')
pos = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device) 

metrics = {}


h_rbf_model = h_diag_RBF(n_centers=30, data_to_fit=samples).to(device)
h_rbf = normalize_metric(h_rbf_model.h, pos)
metrics["RBF"] = RBFManifold(h_rbf)


h_land_model = h_diag_Land(samples, gamma=1.0).to(device)
h_land = normalize_metric(h_land_model.h, pos)
metrics["LAND"] = LandManifold(h_land)


score_fn = lambda x, t: mixture.score_log_p(x)
metrics["Magnitude"] = ScoreBasedManifold(score_fn, "magnitude", p=1)
metrics["Tangent"] = ScoreBasedManifold(score_fn, "tangent")
metrics["Mixed"] = ScoreBasedManifold(score_fn, "mixed", p=1)


trained_models = {}
for name, manifold in metrics.items():
    model_path = os.path.join(save_dir, f"{name}_geodesic_wcg.pt")
    
    if os.path.exists(model_path):
        print(f"Loading saved model for {name}...")
        model = torch.load(model_path, map_location=device)
    else:
        print(f"\nTraining {name}...")
        model = train_geodesic(
            dataloader=dataloader,
            manifold=manifold,
            D=2,
            n_points=100,
            n_steps=2500,
            lr=5e-3,
            device=device
        )
        torch.save(model, model_path)
        print(f"Saved {name} model to {model_path}")
    
    trained_models[name] = model

# =========================================================
# GENERATE CURVES
# =========================================================
x0 = means[10].unsqueeze(0).to(device)
x1 = means[-10].unsqueeze(0).to(device)
n_curve_points = 60
t_vals = torch.linspace(0, 1, n_curve_points, device=device).view(1, n_curve_points, 1)

x0_exp = x0.repeat(1, n_curve_points, 1)
x1_exp = x1.repeat(1, n_curve_points, 1)

curves = {}
for name, model in trained_models.items():
    c_flat = model(
        x0_exp.view(-1, 2),
        x1_exp.view(-1, 2),
        t_vals.view(-1, 1)
    )
    curve = (1 - t_vals) * x0_exp + t_vals * x1_exp + \
            2 * t_vals * (1 - t_vals) * c_flat.view(1, n_curve_points, 2)
    curves[name] = curve

# =========================================================
# PLOT
# =========================================================
plot_geodesics_on_gmm_density(
    curves=curves,
    mixture=mixture,
    X=X,
    Y=Y,
    title="Geodesics on Weighted Semicircle Gaussian Mixture",
    save_path="plots/gaussian_blobs/exp5/geodesics_weighted_semicircle.png",
    device=device
)