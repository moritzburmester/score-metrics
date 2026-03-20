import torch
import torch.nn as nn
import torch.optim as optim
from unicodedata import decimal
import io
import torchvision.transforms as transforms
import PIL
import pickle
import numpy as np
import copy
import os
from models import utils as mutils
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from models.utils import get_score_fn
from configs.dimension_estimation.paper.euclidean_data.ksphere.k1_dim import get_config
#from configs.dimension_estimation.paper.euclidean_data.two_moons.config import get_config
#from configs.dimension_estimation.paper.euclidean_data.gaussian_blobs.config_multiple import get_config
import random 

seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.datasets import make_moons

class CurvatureNet(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2*dim + 1, hidden), # start, end, t -> 2d+1
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x0, x1, t):
        inp = torch.cat([x0, x1, t], dim=1)
        return self.net(inp)

class SyntheticPairsDataset(Dataset):
    """
    Dataset for geodesic interpolation on synthetic manifolds.
    Supports: circle (1D sphere), two moons, Gaussian mixture.
    Returns batches of pairs (x0, x1) for training.
    """
    def __init__(self, manifold_type='circle', n_points=1000, t_steps=50, device='cpu',
                 gaussian_params=None, noise_std=0.0, random_seed=42):
        """
        manifold_type: 'circle', 'two_moons', 'gaussians'
        n_points: number of base points to sample
        t_steps: number of interpolation steps for curves
        gaussian_params: dict with keys: means (K,D), stds (K,), weights (K,)
        """
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

        # Precompute t_vals for curve expansion
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
        idxs = np.random.choice(K, size=n, p=weights)
        for k in idxs:
            point = np.random.normal(loc=means[k].cpu().numpy(), scale=stds[k], size=D)
            samples.append(point)
        return torch.tensor(np.stack(samples), dtype=torch.float32)

    def __len__(self):
        # number of points, we can sample batches without restriction
        return 1000000  # arbitrary large number for infinite sampling

    def __getitem__(self, idx):
        """
        Returns a pair (x0_curve, x1_curve) expanded along t_steps
        """
        idx0 = np.random.randint(0, self.points.shape[0])
        idx1 = np.random.randint(0, self.points.shape[0])
        x0 = self.points[idx0]
        x1 = self.points[idx1]

        # Expand along t_steps
        x0_curve = x0.view(1, -1).repeat(self.t_steps, 1)  # (t_steps, D)
        x1_curve = x1.view(1, -1).repeat(self.t_steps, 1)  # (t_steps, D)
        t_curve = self.t_vals.view(self.t_steps, 1)        # (t_steps,1)

        return x0_curve, x1_curve, t_curve
    

print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.backends.cudnn.version())


device = 'cuda'
gaussian_params = {
    'means': torch.tensor([[3., -2.], [0.,0.], [5.,-4.]], device=device),
    'stds': torch.tensor([1.0, 0.7, 0.5], device=device),
    'weights': torch.tensor([0.2, 0.5, 0.3], device=device)
}

dataset = SyntheticPairsDataset(manifold_type='gaussians', n_points=1000,
                                t_steps=50, device=device,
                                gaussian_params=gaussian_params)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

for x0_batch, x1_batch, t_batch in dataloader:
    print(x0_batch.shape, x1_batch.shape, t_batch.shape)
    break
# Output: (32, 50, D) (32, 50, D) (50,1)  (t_batch can be broadcast)