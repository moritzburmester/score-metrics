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
from utils.plotting import (
plot_geodesics_on_density,
plot_score_vector_field_gaussian,
plot_true_mixture_score_magnitude, 
plot_geodesic_on_score_analytic,
plot_geodesics_on_score)

from configs.ksphere.k1_dim import get_config
from models import utils as mutils
from lightning_modules.utils import create_lightning_module
from lightning_data_modules.utils import create_lightning_datamodule
from lightning_callbacks.utils import get_callbacks
from lightning_data_modules import GaussianBlobDataset, TwoMoonsDataset, KSphereDataset
from lightning_modules import BaseSdeGenerativeModel, KSphereGroundTruthModel #need for lightning module registration
from lightning_modules.utils import create_lightning_module
from models import fcn

device = 'cuda'
save_dir = "saved_models_circle"

config = get_config()


# setup model
pl_module = create_lightning_module(config)
pl_module = type(pl_module).load_from_checkpoint(config.model.checkpoint_path)
pl_module.configure_sde(config)

device = config.device
pl_module = pl_module.to(device)
pl_module.eval()

score_model = pl_module.score_model
sde = pl_module.sde
score_fn = mutils.get_score_fn(
    sde, score_model, conditional=False, train=False, continuous=True
)

# sample from model

pl_module.eval()
n_reference_samples = 1000  
t_sample = 0.4  

with torch.no_grad():
    samples_tuple = pl_module.sample(num_samples=n_reference_samples, show_evolution=True)

    if isinstance(samples_tuple, (tuple, list)):
        final_samples = samples_tuple[0]             
        sampling_info = samples_tuple[1]
        trajectory = sampling_info['evolution']    
        times = sampling_info['times']              
    else:
        final_samples = samples_tuple
        trajectory = torch.stack([final_samples])
        times = torch.tensor([0.])


    step_idx = int((np.abs(times.cpu().numpy() - t_sample)).argmin())
    reference_samples = trajectory[step_idx]      
    reference_samples = reference_samples.to(device)

def create_diffusion_pairs(reference_samples, batch_size=32):
    n_samples = reference_samples.shape[0]
    while True:
        idx0 = torch.randint(0, n_samples, (batch_size,))
        idx1 = torch.randint(0, n_samples, (batch_size,))
        yield reference_samples[idx0], reference_samples[idx1]

dataloader = create_diffusion_pairs(reference_samples, batch_size=32)

# metrics
metrics = {}

# RBF
h_rbf_model = h_diag_RBF(
    n_centers=30,
    data_to_fit=reference_samples
).to(device)

x = torch.linspace(-2, 2, 100)
y = torch.linspace(-2, 2, 100)
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
t = torch.tensor([0.4], device=device)
metrics["magnitude"] = ScoreBasedManifold(score_fn, "magnitude", p=1, t=t)
metrics["tangent"] = ScoreBasedManifold(score_fn, "tangent", t=t)
metrics["mixed"] = ScoreBasedManifold(score_fn, "mixed", p=1, t=t)
trained_models = {}

for name, manifold in metrics.items():
    model_path = os.path.join(save_dir, f"{name}_geodesic_circle.pt")

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
            device=device,
        )

        torch.save(model, model_path)
        print(f"Saved {name} model to {model_path}")

    trained_models[name] = model

x0 = torch.tensor([[-1.2, 0.0]], device=device)
x1 = torch.tensor([[ 1.2, 0.0]], device=device)
                  
n_curve_points = 50
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

plot_bounds = (-2, 2) 

plot_geodesics_on_score(
    curves=curves,          
    score_fn=score_fn,      
    xlim=plot_bounds,
    ylim=plot_bounds,
    title="Geodesics under Different Metrics (Score Landscape)",
    save_path="plots/1d_sphere/exp10/curves_on_score.png",
    n_grid=400,
    t=0.4,
    device=device
)