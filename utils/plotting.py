import torch
import numpy as np
import matplotlib.pyplot as plt
import os 

def plot_score_vector_field_gaussian(
    means,
    stds,
    weights,
    title='Score Vector Field',
    n_grid=20,
    xlim=(-1, 8),
    ylim=(-2, 4),
    device='cuda',
    path=None
):
    """
    Plot the score vector field of a Gaussian mixture.
    
    means: (K,2) tensor of Gaussian means
    stds: (K,) tensor of standard deviations
    weights: (K,) tensor of mixture weights (should sum to 1)
    """
    # Make grid
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)
    XYpairs = np.stack([X.flatten(), Y.flatten()], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float32, device=device, requires_grad=True)  # (n_grid*n_grid, 2)

    # Ensure weights sum to 1
    weights = weights / weights.sum()

    K = means.shape[0]
    D = means.shape[1]

    # Vectorized score computation
    xs_exp = xs.unsqueeze(1)                   # (N,1,D)
    means_exp = means.view(1,K,D)             # (1,K,D)
    stds2 = (stds**2).view(1,K,1)            # (1,K,1)
    weights_exp = weights.view(1,K,1)        # (1,K,1)

    diff = xs_exp - means_exp                 # (N,K,D)
    exponent = -0.5 * (diff**2).sum(-1) / stds2.squeeze(-1)  # (N,K)
    normalizer = (2*np.pi*stds2.squeeze(-1))**(D/2)          # (1,K)
    densities = weights_exp.squeeze(-1) * torch.exp(exponent) / normalizer  # (N,K)

    mixture_density = densities.sum(dim=1, keepdim=True)  # (N,1)
    score_components = -diff / stds2                     # (N,K,D)
    weighted_scores = (densities.unsqueeze(-1) * score_components).sum(dim=1) / mixture_density  # (N,D)

    scores = weighted_scores.detach().cpu().numpy().reshape(n_grid, n_grid, 2)

    # Plot
    fig, ax = plt.subplots(figsize=(8,8))
    q = ax.quiver(X, Y, scores[:,:,0], scores[:,:,1], color='blue', alpha=0.7)
    ax.scatter(means[:,0].cpu(), means[:,1].cpu(), c='red', s=100, label='Means')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_title(title, pad=20)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    plt.tight_layout()
    
    if path is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.show()#


def plot_true_mixture_score_magnitude(
    score_fn,
    means,
    stds,
    weights,
    title='Log True Mixture Score Magnitude',
    n_grid=400,
    xlim=(-1, 7),
    ylim=(-6, 2),
    device='cuda',
    path=None
    ):
    """
    Plot log-magnitude of score using a provided score function (stable).
    """

    # ---- Grid ----
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)

    XYpairs = np.stack([X.flatten(), Y.flatten()], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float32, device=device)

    # ---- Use your STABLE score function ----
    with torch.no_grad():
        # reshape to (B=1, T=N, D=2)
        scores = score_fn(xs.view(1, -1, 2), None)
        scores = scores.view(-1, 2)

        score_magnitude = torch.norm(scores, dim=1)
        score_magnitude = score_magnitude.cpu().numpy().reshape(n_grid, n_grid)

        log_magnitude = np.log(score_magnitude + 1e-8)

    # ---- Plot ----
    plt.figure(figsize=(10,10))
    plt.contourf(X, Y, log_magnitude, cmap='viridis', alpha=0.8, levels=100)
    plt.colorbar(label='log(||score|| + ε)')

    plt.scatter(means[:,0].cpu(), means[:,1].cpu(), c='red', s=100, label='Means')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.1)

    plt.show()

def plot_geodesic_on_score_analytic(curve, score_fn, 
                                    means=None, stds=None, weights=None,
                                    title="Geodesic on Score Landscape",
                                    save_path="plots/geodesic_on_score.png",
                                    n_grid=400,
                                    device='cuda'):

    if not isinstance(curve, torch.Tensor):
        curve = torch.tensor(curve, dtype=torch.float, device=device)

    # Determine plotting bounds
    if means is not None and stds is not None:
        means_np = means.cpu().numpy() if torch.is_tensor(means) else np.array(means)
        stds_np = stds.cpu().numpy() if torch.is_tensor(stds) else np.array(stds)
        x_min = (means_np[:,0] - 3*stds_np).min()
        x_max = (means_np[:,0] + 3*stds_np).max()
        y_min = (means_np[:,1] - 3*stds_np).min()
        y_max = (means_np[:,1] + 3*stds_np).max()
    else:
        x_min, x_max = curve[:,0].min().item(), curve[:,0].max().item()
        y_min, y_max = curve[:,1].min().item(), curve[:,1].max().item()

    eps_x = (x_max - x_min) * 1e-3 + 1e-6
    eps_y = (y_max - y_min) * 1e-3 + 1e-6
    x_min, x_max = x_min - eps_x, x_max + eps_x
    y_min, y_max = y_min - eps_y, y_max + eps_y

    # Generate grid
    x_vals = np.linspace(x_min, x_max, n_grid)
    y_vals = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x_vals, y_vals)
    XYpairs = np.stack([X.ravel(), Y.ravel()], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float, device=device)

    # Compute score magnitudes
    with torch.no_grad():
        if means is not None and stds is not None and weights is not None:
            scores = score_fn(xs.view(1, -1, 2), None)
            scores = scores.view(-1, 2)
        else:
            scores = score_fn(xs)

        score_magnitude = torch.norm(scores, dim=1).cpu().numpy().reshape(n_grid, n_grid)
        log_magnitude = np.log(score_magnitude + 1e-8)

    # Plot
    plt.figure(figsize=(8,8))
    plt.contourf(X, Y, log_magnitude, cmap='viridis', alpha=0.8, levels=100)
    plt.colorbar(label='log(||score|| + ε)')

    curve_np = curve.squeeze(0).cpu().detach().numpy()  # removes batch dim
    plt.plot(curve_np[:,0], curve_np[:,1], 'r-o', label='Geodesic', linewidth=2, markersize=4)
    plt.scatter(curve_np[0,0], curve_np[0,1], color='green', s=80, label='Start')
    plt.scatter(curve_np[-1,0], curve_np[-1,1], color='blue', s=80, label='End')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal', adjustable='box')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved geodesic plot to {save_path}")