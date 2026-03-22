import torch
import torch.nn as nn
import torch.optim as optim
from unicodedata import decimal
import io
import PIL
import pickle
import numpy as np
import copy
import os
from models import utils as mutils
from matplotlib import pyplot as plt
from models.utils import get_score_fn
from configs.ksphere.k1_dim import get_config
#from configs.dimension_estimation.paper.euclidean_data.two_moons.config import get_config
#from configs.dimension_estimation.paper.euclidean_data.gaussian_blobs.config_multiple import get_config
import random 
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
from utils.analytical_score import gaussian_mixture_score

seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class CurvatureNet(nn.Module):
    def __init__(self, D, hidden_dim=128):
        super(CurvatureNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * D + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, D)
        )

    def forward(self, x0, xT, t):
        # x0, xT: (N, D)
        # t: (N, 1)
        x_cat = torch.cat([x0, xT, t], dim=1)
        return self.net(x_cat)

class SyntheticPairsDataset(Dataset):
    def __init__(self, manifold_type='circle', n_points=1000, t_steps=50, device='cpu',
                 gaussian_params=None, noise_std=0.0, random_seed=42):

        super().__init__()
        self.manifold_type = manifold_type
        self.n_points = n_points
        self.t_steps = t_steps
        self.device = device
        self.noise_std = noise_std
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        if manifold_type == 'circle':
            theta = np.linspace(0, 2*np.pi, n_points)
            x = np.cos(theta)
            y = np.sin(theta)
            self.points = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
        elif manifold_type == 'two_moons':
            X, _ = make_moons(n_samples=n_points, noise=noise_std)
            self.points = torch.tensor(X, dtype=torch.float32)
        elif manifold_type == 'gaussians':
            if gaussian_params is None:
                raise ValueError("gaussian_params must be provided for Gaussian mixture")
            means = gaussian_params['means']
            stds = gaussian_params['stds']
            weights = gaussian_params['weights']
            self.points = self.sample_gaussian_mixture(n_points, means, stds, weights)
        else:
            raise NotImplementedError(f"Unknown manifold type: {manifold_type}")

        self.points = self.points.to(device)

        self.t_vals = torch.linspace(0, 1, t_steps, device=device).view(1, t_steps, 1)

    def sample_gaussian_mixture(self, n, means, stds, weights):
        """
        Sample n points from a Gaussian mixture
        means: (K,D)
        stds: (K,)
        weights: (K,)
        """
        K, D = means.shape
        samples = []
        weights_np = weights.cpu().numpy()
        weights_np = weights_np / weights_np.sum()
        idxs = np.random.choice(K, size=n, p=weights_np)
        for k in idxs:
            point = np.random.normal(loc=means[k].cpu().numpy(), scale=stds[k].cpu(), size=D)
            samples.append(point)
        return torch.tensor(np.stack(samples), dtype=torch.float32)

    def __len__(self):
        return 1000000  # arbitrary large number for infinite sampling

    def __getitem__(self, idx):
        idx0 = np.random.randint(0, self.points.shape[0])
        idx1 = np.random.randint(0, self.points.shape[0])

        x0 = self.points[idx0]
        x1 = self.points[idx1]

        return x0, x1
    
def train_geodesic(
    dataloader,
    manifold,   
    D,
    n_points=50,
    n_steps=2000,
    lr=1e-3,
    device='cuda'
):
    model = CurvatureNet(D).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dt = 1.0 / (n_points - 1)

    for step, (x0_batch, x1_batch) in enumerate(dataloader):
        if step >= n_steps:
            break

        x0_batch = x0_batch.to(device)
        x1_batch = x1_batch.to(device)

        B = x0_batch.shape[0]

        t_vals = torch.linspace(0, 1, n_points, device=device)
        t_batch = t_vals.view(1, n_points, 1).repeat(B, 1, 1)

        x0_exp = x0_batch.unsqueeze(1).repeat(1, n_points, 1)
        x1_exp = x1_batch.unsqueeze(1).repeat(1, n_points, 1)

        c_flat = model(
            x0_exp.reshape(-1, D),
            x1_exp.reshape(-1, D),
            t_batch.reshape(-1, 1)
        )
        c = c_flat.view(B, n_points, D)

        curve = (1 - t_batch) * x0_exp + t_batch * x1_exp + 2 * t_batch * (1 - t_batch) * c

        velocities = (curve[:, 1:] - curve[:, :-1]) / dt
        points = curve[:, :-1]  # (B, n_points-1, D)

        G = manifold.metric(points)  # (B, n_points-1, D, D)

        v = velocities.unsqueeze(2)  # (B,N,1,D)
        energy = torch.matmul(torch.matmul(v, G), v.transpose(2, 3))
        energy = 0.5 * energy.squeeze(-1).squeeze(-1) * dt

        loss = energy.sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step} | energy {loss.item():.4f}")

    return model