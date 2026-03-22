import torch
import numpy as np  

def gaussian_mixture_score(x, means, stds, weights):
    """
    x: (1,D)
    returns: score at x, shape (1,D)
    """
    K = len(means)
    x = x.to(means.device)
    densities = []
    component_scores = []

    for k in range(K):
        diff = x - means[k]
        var = stds[k]**2
        exponent = -0.5 * (diff**2).sum(dim=1) / var
        normalizer = (2 * np.pi * var)**(x.shape[1]/2)
        Nk = torch.exp(exponent) / normalizer
        densities.append(weights[k] * Nk)

        score_k = -diff / var
        component_scores.append(score_k)

    densities = torch.stack(densities, dim=1)           # (1,K)
    component_scores = torch.stack(component_scores, dim=1)  # (1,K,D)

    mixture_density = densities.sum(dim=1, keepdim=True)      # (1,1)
    weighted_scores = (densities.unsqueeze(2) * component_scores).sum(dim=1)
    score = weighted_scores / mixture_density
    return score  # (1,D)


def create_sine_slalom_gaussian_mixture(
    start_point=(1.0, 1.0),
    end_point=(8.0, 1.0),
    K=8,
    std=0.5,
    device='cuda',
    amplitude=2.0
):
    """
    Create a Gaussian mixture along a sine-wave 'slalom'.

    Returns:
        means: (K,2) tensor
        stds: (K,) tensor
        weights: (K,) tensor
    """
    D = 2
    weight = 1.0 / K

    x_coords = np.linspace(start_point[0], end_point[0], K)
    y_coords = start_point[1] + amplitude * np.sin(np.linspace(0, 2*np.pi, K))  # sine slalom

    means = torch.tensor(np.stack([x_coords, y_coords], axis=1), dtype=torch.float32, device=device)
    stds = torch.tensor([std]*K, dtype=torch.float32, device=device)
    weights = torch.tensor([weight]*K, dtype=torch.float32, device=device)

    return means, stds, weights

def create_alternating_line_gaussian_mixture(
    start_point=(1.0, 1.0),
    end_point=(8.0, 1.0),
    K=15,
    std=0.5,
    device='cuda',
    offset=1.0
):
    """
    Create a Gaussian mixture along a straight line with alternating offsets above/below the line.

    Returns:
        means: (K,2) tensor
        stds: (K,) tensor
        weights: (K,) tensor
    """
    D = 2
    weight = 1.0 / K

    x_coords = np.array([2, 6, 4, 8, 7, 10, 11, 14, 17, 18, 20, 21, 24, 22, 26])
    # Alternating offsets along y-axis
    y_coords = np.array([2, 4, 7, 8, 11, 12, 15, 14, 15, 12, 8, 11, 7, 4, 2])

    means = torch.tensor(np.stack([x_coords, y_coords], axis=1), dtype=torch.float32, device=device)
    stds = torch.tensor([std]*K, dtype=torch.float32, device=device)
    weights = torch.tensor([weight]*K, dtype=torch.float32, device=device)

    return means, stds, weights