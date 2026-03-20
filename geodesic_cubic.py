import torch
from stochman.geodesic import geodesic_minimizing_energy
from stochman.curves import CubicSpline


p0, p1 = torch.randn(1, 2), torch.randn(1, 2)
curve = CubicSpline(p0, p1)
print(1)