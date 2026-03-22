import torch
import os
import torch.optim as optim
from geodesic_interpolant_net import SyntheticPairsDataset
from manifolds import RBFManifold, LandManifold, ScoreBasedManifold, h_diag_RBF, h_diag_Land, normalize_metric
from stochman.geodesic import geodesic_minimizing_energy
from curves import CubicSpline
from utils.plotting import plot_geodesics_on_score
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
save_dir = "plots/1d_sphere/exp10"
save_dir_curve = "saved_models_circle"
os.makedirs(save_dir, exist_ok=True)

n_curve_points = 50
lr = 5e-3
t_score = torch.tensor([0.4], device=device)
curves_path = os.path.join(save_dir_curve, "curves_circle_spline.pt")


config = get_config()
pl_module = create_lightning_module(config)
pl_module = type(pl_module).load_from_checkpoint(config.model.checkpoint_path)
pl_module.configure_sde(config)
pl_module = pl_module.to(device)
pl_module.eval()

score_model = pl_module.score_model
sde = pl_module.sde
score_fn = mutils.get_score_fn(
    sde, score_model, conditional=False, train=False, continuous=True
)

def score_fn_wrapper(x, t=None):
    if t is not None:
        if t.numel() == 1:
            t_flat = t.repeat(x.shape[0])
        else:
            t_flat = t.reshape(-1)
        scores = score_fn(x, t_flat)
    else:
        scores = score_fn(x, None)
    return scores


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

    step_idx = int((torch.abs(times - t_sample)).argmin())
    reference_samples = trajectory[step_idx].to(device) 

def create_diffusion_pairs(reference_samples, batch_size=32):
    n_samples = reference_samples.shape[0]
    while True:
        idx0 = torch.randint(0, n_samples, (batch_size,))
        idx1 = torch.randint(0, n_samples, (batch_size,))
        yield reference_samples[idx0], reference_samples[idx1]

dataloader = create_diffusion_pairs(reference_samples, batch_size=32)


metrics = {}

# RBF manifold
h_rbf_model = h_diag_RBF(n_centers=30, data_to_fit=reference_samples).to(device)
pos = reference_samples
h_rbf = normalize_metric(h_rbf_model.h, pos)
metrics["rbf"] = RBFManifold(h_rbf)

# Land manifold
h_land_model = h_diag_Land(reference_samples, gamma=1.0).to(device)
h_land = normalize_metric(h_land_model.h, pos)
metrics["land"] = LandManifold(h_land)

# Score-based manifolds
metrics["magnitude"] = ScoreBasedManifold(score_fn_wrapper, "magnitude", p=1, t=t_score)
metrics["tangent"] = ScoreBasedManifold(score_fn_wrapper, "tangent", t=t_score)
metrics["mixed"] = ScoreBasedManifold(score_fn_wrapper, "mixed", p=1, t=t_score)


x0 = torch.tensor([[-1.2, 0.0]], device=device)
x1 = torch.tensor([[ 1.2, 0.0]], device=device)
alpha = torch.linspace(0, 1, n_curve_points, device=device).view(1, n_curve_points)


if os.path.exists(curves_path):
    print(f"Loading saved curves from {curves_path}...")
    curves = torch.load(curves_path)
else:
    curves = {}
    for name, manifold in metrics.items():
        print(f"Optimizing {name}...")
        spline = CubicSpline(x0, x1, num_nodes=n_curve_points, requires_grad=True).to(device)
        geodesic_minimizing_energy(
            spline,
            manifold,
            optimizer=lambda params, lr: torch.optim.Adam(params, lr=lr),
            max_iter=1500,
            eval_grid=n_curve_points
        )
        curves[name] = spline(alpha).detach()
    torch.save(curves, curves_path)
    print(f"Saved curves to {curves_path}")


plot_geodesics_on_score(
    curves=curves,
    score_fn=score_fn_wrapper,
    xlim=(-2, 2),
    ylim=(-2, 2),
    title="Geodesics under Different Metrics (Diffusion Samples + Score Network)",
    save_path=os.path.join(save_dir, "curves_on_score_spline.png"),
    n_grid=400,
    t=0.4,
    device=device
)