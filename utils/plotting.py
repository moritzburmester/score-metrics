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
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)
    XYpairs = np.stack([X.flatten(), Y.flatten()], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float32, device=device, requires_grad=True)  # (n_grid*n_grid, 2)

    weights = weights / weights.sum()

    K = means.shape[0]
    D = means.shape[1]

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

    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)

    XYpairs = np.stack([X.flatten(), Y.flatten()], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float32, device=device)

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

    x_vals = np.linspace(x_min, x_max, n_grid)
    y_vals = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x_vals, y_vals)
    XYpairs = np.stack([X.ravel(), Y.ravel()], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float, device=device)

    with torch.no_grad():
        if means is not None and stds is not None and weights is not None:
            scores = score_fn(xs.view(1, -1, 2), None)
            scores = scores.view(-1, 2)
        else:
            scores = score_fn(xs)

        score_magnitude = torch.norm(scores, dim=1).cpu().numpy().reshape(n_grid, n_grid)
        log_magnitude = np.log(score_magnitude + 1e-8)

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


def gaussian_mixture_density(xs, means, stds, weights):
    """
    xs: (N,2)
    returns: (N,)
    """
    K = means.shape[0]
    D = means.shape[1]

    xs_exp = xs.unsqueeze(1)              # (N,1,D)
    means_exp = means.view(1,K,D)         # (1,K,D)
    stds2 = (stds**2).view(1,K,1)         # (1,K,1)

    diff = xs_exp - means_exp             # (N,K,D)

    exponent = -0.5 * (diff**2).sum(-1) / stds2.squeeze(-1)
    normalizer = (2*np.pi*stds2.squeeze(-1))**(D/2)

    densities = weights.view(1,K) * torch.exp(exponent) / normalizer

    return densities.sum(dim=1)  # (N,)

def plot_geodesics_on_density(
    curves_dict,
    means,
    stds,
    weights,
    title="Geodesics on Density",
    save_path="plots/geodesics_density.png",
    n_grid=300,
    device='cuda'
):

 
    means_np = means.cpu().numpy()
    stds_np = stds.cpu().numpy()

    x_min = (means_np[:,0] - 3*stds_np).min()
    x_max = (means_np[:,0] + 3*stds_np).max()
    y_min = (means_np[:,1] - 3*stds_np).min()
    y_max = (means_np[:,1] + 3*stds_np).max()


    x_vals = np.linspace(x_min, x_max, n_grid)
    y_vals = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x_vals, y_vals)

    XYpairs = np.stack([X.ravel(), Y.ravel()], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float32, device=device)

    with torch.no_grad():
        density = gaussian_mixture_density(xs, means, stds, weights)
        density = density.cpu().numpy().reshape(n_grid, n_grid)
        log_density = np.log(density + 1e-8)


    plt.figure(figsize=(9,9))
    plt.contourf(X, Y, log_density, cmap='viridis', levels=80, alpha=0.85)


    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label("log density")

    colors = {
        "rbf": "red",
        "land": "blue",
        "magnitude": "orange",
        "tangent": "green",
        "mixed": "purple"
    }

    markers = {
        "rbf": "o",
        "land": "s",
        "magnitude": "^",
        "tangent": "x",
        "mixed": "d"
    }


    for name, curve in curves_dict.items():
        curve_np = curve.squeeze(0).detach().cpu().numpy()

        plt.plot(
            curve_np[:,0],
            curve_np[:,1],
            color=colors[name],
            marker=markers[name],
            markevery=5,
            linewidth=2,
            label=name
        )

    plt.scatter(means[:,0].cpu(), means[:,1].cpu(),
                c='white', s=80, edgecolors='black', label='means')

    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print("Saved Plot!")
    plt.show()


def plot_geodesic_on_density(curve,
                            means,
                            stds,
                            weights,
                            title="Geodesic on Density",
                            save_path="plots/geodesic_density.png",
                            n_grid=400,
                            device='cuda'):

    if not isinstance(curve, torch.Tensor):
        curve = torch.tensor(curve, dtype=torch.float, device=device)

    means = means.to(device)
    stds = stds.to(device)
    weights = weights / weights.sum()

    # ---- bounds ----
    means_np = means.cpu().numpy()
    stds_np = stds.cpu().numpy()

    x_min = (means_np[:,0] - 3*stds_np).min()
    x_max = (means_np[:,0] + 3*stds_np).max()
    y_min = (means_np[:,1] - 3*stds_np).min()
    y_max = (means_np[:,1] + 3*stds_np).max()

    x_vals = np.linspace(x_min, x_max, n_grid)
    y_vals = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x_vals, y_vals)

    XYpairs = np.stack([X.ravel(), Y.ravel()], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float32, device=device)

    # ---- density computation ----
    with torch.no_grad():
        xs_exp = xs.unsqueeze(1)                # (N,1,2)
        means_exp = means.view(1,-1,2)          # (1,K,2)
        stds2 = (stds**2).view(1,-1,1)          # (1,K,1)
        weights_exp = weights.view(1,-1,1)      # (1,K,1)

        diff = xs_exp - means_exp               # (N,K,2)
        exponent = -0.5 * (diff**2).sum(-1) / stds2.squeeze(-1)

        normalizer = (2*np.pi*stds2.squeeze(-1))**(1)  # D=2 → exponent already handled
        densities = weights_exp.squeeze(-1) * torch.exp(exponent) / normalizer

        mixture_density = densities.sum(dim=1)
        log_density = torch.log(mixture_density + 1e-8)

        log_density = log_density.cpu().numpy().reshape(n_grid, n_grid)

    # ---- plot ----
    plt.figure(figsize=(8,8))
    plt.contourf(X, Y, log_density, cmap='viridis', alpha=0.8, levels=100)
    plt.colorbar(label='log p(x)')

    curve_np = curve.squeeze(0).cpu().detach().numpy()
    plt.plot(curve_np[:,0], curve_np[:,1], 'r-o', linewidth=2, markersize=4, label='Geodesic')

    plt.scatter(curve_np[0,0], curve_np[0,1], color='green', s=80, label='Start')
    plt.scatter(curve_np[-1,0], curve_np[-1,1], color='blue', s=80, label='End')

    plt.scatter(means[:,0].cpu(), means[:,1].cpu(), c='white', s=50, label='Means')

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved density geodesic plot to {save_path}")


def plot_geodesics_on_gmm_density(curves, mixture, X, Y, title="Geodesics", save_path=None, device='cpu'):
    plt.figure(figsize=(8, 6))

    pos = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)
    density = mixture.prob(pos).view(X.shape).detach().cpu()
    
    plt.contourf(
        X.cpu(),
        Y.cpu(),
        density,
        levels=40,
        cmap='Blues',
        alpha=0.7
    )
    
    colors = ['red', 'green', 'orange', 'purple', 'black', 'cyan']
    markers = ['o', 's', '^', 'D', '*', 'x']
    

    for i, (name, curve) in enumerate(curves.items()):
        curve = curve.squeeze(0).detach().cpu() 
        plt.plot(curve[:, 0], curve[:, 1], label=name, color=colors[i % len(colors)], linewidth=2)
        plt.scatter(curve[:, 0], curve[:, 1], color=colors[i % len(colors)],
                    marker=markers[i % len(markers)], s=20)
    
    first_curve = list(curves.values())[0].squeeze(0).detach().cpu()
    plt.scatter(first_curve[0,0], first_curve[0,1], c='black', s=80)
    plt.scatter(first_curve[-1,0], first_curve[-1,1], c='black', s=80)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        print("Saved plot!")
    
    plt.show()

def plot_geodesics_on_score(curves, score_fn,
                            title="Geodesics on Score Landscape",
                            save_path=None,
                            n_grid=400, t=0.01,
                            xlim=(-2, 2), ylim=(-2, 2),
                            device='cpu'):

    x_vals = np.linspace(xlim[0], xlim[1], n_grid)
    y_vals = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x_vals, y_vals)
    XYpairs = np.stack([X.ravel(), Y.ravel()], axis=1)
    xs = torch.tensor(XYpairs, dtype=torch.float, device=device)
    ts = torch.full((n_grid*n_grid,), t, dtype=torch.float, device=device)

    with torch.no_grad():
        scores = score_fn(xs, ts)  # (N,2)
        score_magnitude = torch.norm(scores, dim=1).cpu().numpy().reshape(n_grid, n_grid)
        log_magnitude = np.log(score_magnitude + 1e-8)

    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, log_magnitude, levels=100, cmap='viridis', alpha=0.8)
    plt.colorbar(label='log(||score|| + ε)')

    colors = ['red', 'green', 'orange', 'purple', 'black', 'cyan']
    markers = ['o', 's', '^', 'D', '*', 'x']

    for i, (name, curve) in enumerate(curves.items()):
        curve_np = curve.squeeze(0).detach().cpu().numpy() if curve.dim() == 3 else curve.detach().cpu().numpy()
        plt.plot(curve_np[:,0], curve_np[:,1], color=colors[i % len(colors)], 
                 marker=markers[i % len(markers)], label=name, linewidth=2, markersize=4)
        plt.scatter(curve_np[0,0], curve_np[0,1], color='green', s=60, zorder=5)  # start
        plt.scatter(curve_np[-1,0], curve_np[-1,1], color='blue', s=60, zorder=5)  # end

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().set_aspect('equal', adjustable='box')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved geodesic plot to {save_path}")
    else:
        plt.show()