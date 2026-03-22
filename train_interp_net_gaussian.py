import torch
import numpy as np 
from utils.analytical_score import gaussian_mixture_score, create_sine_slalom_gaussian_mixture, create_alternating_line_gaussian_mixture
from geodesic_interpolant_net import SyntheticPairsDataset, train_geodesic, CurvatureNet
from utils.plotting import plot_score_vector_field_gaussian, plot_true_mixture_score_magnitude, plot_geodesic_on_score_analytic
from score_manifolds import ScoreBasedManifold
## define score fn 
def fast_gaussian_score(points, t, means, stds, weights):
    """
    Numerically stable score of a Gaussian mixture.
    
    points: (B,T,D)
    means: (K,D)
    stds: (K,)
    weights: (K,)
    """

    B, T, D = points.shape
    K = means.shape[0]

    # Expand
    x = points.unsqueeze(2)                 # (B,T,1,D)
    means = means.view(1,1,K,D)             # (1,1,K,D)
    stds2 = (stds**2).view(1,1,K,1)         # (1,1,K,1)

    # Differences
    diff = x - means                        # (B,T,K,D)

    # ----- LOG PROBABILITIES (stable) -----
    exponent = -0.5 * (diff**2).sum(-1) / stds2.squeeze(-1)   # (B,T,K)

    log_weights = torch.log(weights.view(1,1,K))               # (1,1,K)
    log_normalizer = (D/2) * torch.log(2 * torch.pi * stds2.squeeze(-1))  # (1,1,K)

    log_probs = log_weights + exponent - log_normalizer        # (B,T,K)

    # ----- STABILIZE WITH LOG-SUM-EXP -----
    max_log_probs, _ = torch.max(log_probs, dim=2, keepdim=True)   # (B,T,1)

    probs = torch.exp(log_probs - max_log_probs)   # safe exponentials
    probs = probs / probs.sum(dim=2, keepdim=True)  # normalized responsibilities

    # ----- SCORE -----
    score_components = -diff / stds2              # (B,T,K,D)

    score = (probs.unsqueeze(-1) * score_components).sum(dim=2)  # (B,T,D)

    return score

device = 'cuda'

means_alt, stds_alt, weights_alt = create_alternating_line_gaussian_mixture(device=device)

#plot_score_vector_field_gaussian(    means=means,    stds=stds,    weights=weights,    title="Slalom Gaussian Mixture Score Field",    n_grid=30,    xlim=(0, 9),    ylim=(-3, 5),    device='cuda',    path='plots/gaussian_blobs/exp5/slalom.png')

'''print("Plotting True Mixture...")

plot_true_mixture_score_magnitude(
    score_fn=lambda points, t: fast_gaussian_score(points, t, means_alt, stds_alt, weights_alt),
    means=means_alt,
    stds=stds_alt,
    weights=weights_alt,
    title="Alternating Line Gaussian Mixture Score Magnitude",
    n_grid=400,
    xlim=(0, 28),
    ylim=(0, 18),
    device='cuda',
    path='plots/gaussian_blobs/exp5/alt_mag.png'
)
'''
gaussian_params = {'means': means_alt, 'stds': stds_alt, 'weights': weights_alt}

dataset = SyntheticPairsDataset(
    manifold_type='gaussians',
    n_points=1000,
    device=device,
    gaussian_params=gaussian_params
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

train_mode = True # False

if train_mode:
    #metrics = ['magnitude', 'tangent', 'normal', 'mixed']
    p_list = [1, 2, 3, 4, 5, 6]
    lr_list = [5e-3]#[1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    for lr in lr_list: 
        # types: magnitude, tangent, normal, mixed
        manifold = ScoreBasedManifold(
        score_fn=lambda x, t: fast_gaussian_score(x.view(1,-1,2), None, means_alt, stds_alt, weights_alt).view(-1,2),
        metric_type='magnitude',
        alpha=1.0,
        beta=1.0,
        gamma=1,
        p=4
        )

        print("Optimizing Geodesic...")
        model = train_geodesic(
            dataloader=dataloader,
            manifold=manifold,
            D=2,
            n_points=100,
            n_steps=5000,
            lr=lr,
            device=device
        )

        x0 = torch.tensor([[2, 2]], device=device)
        x1 = torch.tensor([[26, 2]], device=device)
        n_curve_points = 50
        t_vals = torch.linspace(0,1,n_curve_points, device=device).view(1,n_curve_points,1)
        x0_exp = x0.repeat(1,n_curve_points,1)
        x1_exp = x1.repeat(1,n_curve_points,1)
        c_flat = model(x0_exp.view(-1,2), x1_exp.view(-1,2), t_vals.view(-1,1))
        curve = (1-t_vals)*x0_exp + t_vals*x1_exp + 2*t_vals*(1-t_vals)*c_flat.view(1,n_curve_points,2)

        #print("Plotting...")
        plot_geodesic_on_score_analytic(curve, 
                                        score_fn=lambda points, t: fast_gaussian_score(points, t, means_alt, stds_alt, weights_alt),
                                        means=means_alt, stds=stds_alt, weights=weights_alt,
                                        title=f"Geodesic using magnitude with lr = {lr} Metric",
                                        save_path=f"plots/gaussian_blobs/exp5/geodesic_on_score_magnitude_lr={lr}.png",
                                        n_grid=400,
                                        device=device)

if not train_mode:
    print("Loading model...")
    checkpoint = torch.load("saved_models/geodesic_model.pt")
    model = CurvatureNet(checkpoint['D']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    means = checkpoint['means']
    stds = checkpoint['stds']
    weights = checkpoint['weights']
