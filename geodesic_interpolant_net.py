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
    # defined as in EBM Paper 
    def __init__(self, num_channel):
        super(Curve_Net, self).__init__()
        self.num_channel = num_channel
        self.net = nn.Sequential(
            nn.Linear(64 * 3, self.num_channel),
            nn.ELU(),
            nn.Linear(self.num_channel, self.num_channel),
            nn.ELU(),
            nn.Linear(self.num_channel, self.num_channel),
            nn.ELU(),
            nn.Linear(self.num_channel, self.num_channel),
            nn.ELU(),
            nn.Linear(self.num_channel, self.num_channel),
            nn.ELU(),
            nn.Linear(self.num_channel, 64)
        )

        # self.net.apply(init_weights)

    def forward(self, x0, xT, t):
        x_cat = torch.cat([x0, xT, t], dim=1)
        return self.net(x_cat)

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
    

def train_geodesic_interpolant_batched(
    x0_batch, x1_batch,         # BxD tensors
    score_fn,
    n_points=20,
    n_steps=2000,
    lr=1e-3,
    device='cpu',
    alpha=1.0,
    beta=1.0
):
    """
    Train curvature network to interpolate geodesics between batches of points.
    
    x0_batch: BxD tensor of starting points
    x1_batch: BxD tensor of ending points
    score_fn: function(points, t) -> BxT x D, maps points to score vectors
    n_points: number of discrete points along curve
    n_steps: number of optimization steps
    alpha, beta: metric scaling hyperparameters
    """
    
    B, D = x0_batch.shape
    dt = 1.0 / (n_points - 1)

    model = CurvatureNet(D).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    t_vals = torch.linspace(0,1,n_points,device=device).view(1, n_points, 1)  # 1 x T x 1

    # Repeat t_vals across batch
    t_batch = t_vals.repeat(B, 1, 1)  # B x T x 1

    # Expand x0 and x1 along the curve
    x0_exp = x0_batch.unsqueeze(1).repeat(1, n_points, 1)  # B x T x D
    x1_exp = x1_batch.unsqueeze(1).repeat(1, n_points, 1)  # B x T x D

    for step in range(n_steps):
        optimizer.zero_grad()

        # Flatten for network: (B*T) x (2D + 1)
        x0_flat = x0_exp.view(-1, D)
        x1_flat = x1_exp.view(-1, D)
        t_flat = t_batch.view(-1, 1)

        c_flat = model(x0_flat, x1_flat, t_flat)
        c = c_flat.view(B, n_points, D)

        # Interpolated curve
        curve = (1 - t_batch) * x0_exp + t_batch * x1_exp + 2 * t_batch * (1 - t_batch) * c

        # Compute velocities along time
        velocities = (curve[:,1:,:] - curve[:,:-1,:]) / dt  # B x (T-1) x D
        points = curve[:,:-1,:]  # B x (T-1) x D

        # Compute batch metric G from score function
        ts = t_batch[:,:-1,:]  # B x (T-1) x 1
        scores = score_fn(points, ts)  # B x (T-1) x D
        I = torch.eye(D, device=device).unsqueeze(0).unsqueeze(0)  # 1 x 1 x D x D
        I = I.expand(B, n_points-1, D, D)  # B x (T-1) x D x D

        # replace with modular metric here 
        G = (alpha * scores.norm(dim=2, keepdim=True).unsqueeze(-1)**4 + beta) * I  # B x (T-1) x D x D

        # Compute kinetic energy: v^T G v
        v = velocities.unsqueeze(2)  # B x (T-1) x 1 x D
        energy = torch.matmul(torch.matmul(v, G), v.transpose(2,3))  # B x (T-1) x 1 x 1
        energy = 0.5 * energy.squeeze(-1).squeeze(-1) * dt  # B x (T-1)
        loss = energy.sum(dim=1).mean()  # average over batch

        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step} energy {loss.item():.4f}")

    # Return final batched curves
    with torch.no_grad():
        c_flat = model(x0_exp.view(-1,D), x1_exp.view(-1,D), t_batch.view(-1,1))
        c = c_flat.view(B, n_points, D)
        curve = (1 - t_batch) * x0_exp + t_batch * x1_exp + 2 * t_batch * (1 - t_batch) * c

    return curve  # B x T x D    

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