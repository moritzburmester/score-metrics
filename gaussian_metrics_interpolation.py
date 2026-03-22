import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.toy_dataset import GaussianMixture

print(torch.cuda.is_available())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- Helper Functions ----------
def linear_normalization(maxi, mini, target_max, target_min):
    alpha = (target_max - target_min)/(maxi - mini)
    beta = target_min - alpha*mini
    return alpha, beta

def normalize_diag(h, dataset_sample, mini=1e-4, maxi=2):
    all_h = h(dataset_sample)
    if len(all_h.shape) == 2:
        alpha, beta = [], []
        for i in range(all_h.shape[1]):
            a, b = linear_normalization(all_h[:,i].max(), all_h[:,i].min(), maxi, mini)
            alpha.append(a)
            beta.append(b)
        return torch.tensor(alpha).unsqueeze(0), torch.tensor(beta).unsqueeze(0)

# ---------- Metrics ----------

class ScoreMagnitudeMetric(nn.Module):
    """
    Diagonal metric based on the magnitude of the score function.
    g(x) = (alpha * ||s(x)|| + beta) * I
    """
    def __init__(self, score_fn, pos_sample, mini=1e-3, maxi=1, exponent=4):
        super().__init__()
        self.score_fn = score_fn
        self.exponent = exponent

        # compute normalization factors
        with torch.no_grad():
            scores = self.score_fn(pos_sample)  # shape [N,D]
            norm_vals = scores.norm(dim=1, keepdim=True) ** self.exponent
            alpha, beta = linear_normalization(norm_vals.max().item(), norm_vals.min().item(), maxi, mini)
        self.alpha = alpha
        self.beta = beta

    def g(self, x):
        I = torch.eye(x.shape[1], device=x.device).unsqueeze(0)  # shape [1, D, D]
        scores = self.score_fn(x)  # shape [B,D]
        norm_power = scores.norm(dim=1, keepdim=True) ** self.exponent  # [B,1]
        return (self.alpha * norm_power + self.beta) * I  # [B,D,D]

    def kinetic(self, x, x_dot):
        G = self.g(x)  # [B,D,D]
        # diagonal metric: elementwise multiply
        return torch.einsum('bi,bij,bj->b', x_dot, G, x_dot)


class ScoreTangentMetric(nn.Module):
    """
    Tangent-based metric: g(x) = I + s(x) s(x)^T
    """
    def __init__(self, score_fn):
        super().__init__()
        self.score_fn = score_fn

    def g(self, x):
        I = torch.eye(x.shape[1], device=x.device).unsqueeze(0)  # [1,D,D]
        s = self.score_fn(x).unsqueeze(2)  # [B,D,1]
        G = I + torch.bmm(s, s.transpose(1,2))  # [B,D,D]
        return G

    def kinetic(self, x, x_dot):
        G = self.g(x)
        return torch.einsum('bi,bij,bj->b', x_dot, G, x_dot)


class ScoreNormCorrectedMetric(nn.Module):
    """
    Corrected magnitude metric: g(x) = I - s(x) s(x)^T / (1 + ||s||^2)
    """
    def __init__(self, score_fn):
        super().__init__()
        self.score_fn = score_fn

    def g(self, x):
        I = torch.eye(x.shape[1], device=x.device).unsqueeze(0)
        s = self.score_fn(x)  # [B,D]
        norm_sq = (s**2).sum(dim=1, keepdim=True)  # [B,1]
        outer = s.unsqueeze(2) * s.unsqueeze(1)  # [B,D,D]
        G = I - outer / (1 + norm_sq.unsqueeze(-1))
        return G

    def kinetic(self, x, x_dot):
        G = self.g(x)
        return torch.einsum('bi,bij,bj->b', x_dot, G, x_dot)

class h_diag_RBF(nn.Module):
    def __init__(self, n_centers, data_size=2, data_to_fit=None, kappa=1):
        super().__init__()
        self.K = n_centers
        self.data_size = data_size
        self.kappa = kappa
        self.W = nn.Parameter(torch.rand(self.K,1))
        sigmas = torch.ones(self.K, data_size)

        if data_to_fit is not None:
            clustering_model = KMeans(n_clusters=self.K).fit(data_to_fit.cpu().numpy())
            clusters = torch.tensor(clustering_model.cluster_centers_, dtype=torch.float32)
            self.register_buffer('C', clusters)
            labels = clustering_model.labels_
            for k in range(self.K):
                points = data_to_fit[labels == k]
                if len(points) > 0:
                    variance = ((points - clusters[k]) ** 2).mean(0)
                    sigmas[k,:] = torch.sqrt(variance)
        else:
            self.register_buffer('C', torch.zeros(self.K, self.data_size))

        self.register_buffer('lamda', 0.5 / (self.kappa*sigmas)**2)

    def h(self, x_t):
        dist2 = torch.cdist(x_t, self.C)**2
        phi_x = torch.exp(-0.5 * self.lamda[None,:,:] * dist2[:,:,None])
        return phi_x.sum(dim=1)

class h_diag_Land(nn.Module):
    def __init__(self, reference_sample, gamma=0.2):
        super().__init__()
        self.reference_sample = reference_sample
        self.gamma = gamma

    def weighting_function(self, x):
        pairwise_sq_dist = ((x[:,None,:]-self.reference_sample[None,:,:])**2).sum(-1)
        return torch.exp(-pairwise_sq_dist/(2*self.gamma**2))

    def h(self, x):
        weights = self.weighting_function(x)
        diffs = self.reference_sample[None,:,:] - x[:,None,:]
        return torch.einsum('bn,bnd->bd', weights, diffs**2)

class DiagonalRiemannianMetric(nn.Module):
    def __init__(self, h, euclid_weight=0):
        super().__init__()
        self.h = h
        self.euclid_weight = euclid_weight

    def g(self, x):
        return self.h(x)

    def kinetic(self, x, x_dot):
        g = self.g(x)
        return torch.einsum('bi,bi->b', x_dot, (self.euclid_weight+g)*x_dot)

class ConformalRiemannianMetric(nn.Module):
    def __init__(self, h, euclidian_weight=0):
        super().__init__()
        self.h = h
        self.euclidian_weight = euclidian_weight

    def g(self, x):
        return self.h(x)

    def kinetic(self, x, x_dot):
        g = self.g(x)
        return (self.euclidian_weight+g)*(x_dot.pow(2).sum(-1))

# ---------- Utility for metrics normalization ----------
def normalize_metric(h_func, pos, mini=1e-3, maxi=1):
    alpha, beta = normalize_diag(h_func, pos, mini=mini, maxi=maxi)
    return lambda x: 1/(alpha*h_func(x)+beta)

# ---------- Synthetic Gaussian Mixtures ----------
def create_gaussian_mixture(NB_GAUSSIANS=200, RADIUS=8, weights=None):
    mean_angles = torch.linspace(0, math.pi, NB_GAUSSIANS, dtype=torch.float32)
    MEAN = RADIUS * torch.stack([torch.cos(mean_angles), torch.sin(mean_angles)], dim=1)
    COVAR = torch.eye(2).unsqueeze(0).repeat(NB_GAUSSIANS,1,1)
    if weights is None:
        weights = torch.ones(NB_GAUSSIANS)/NB_GAUSSIANS
    return GaussianMixture(center_data=MEAN, covar=COVAR, weight=weights), MEAN, COVAR

# ---------- Geodesic Optimization ----------
def optimize_geodesic(metric, z0, z1, T_STEPS=50, EPOCH=5000, lr=1e-3):
    dt = 1.0/(T_STEPS-1)
    z_t = (1 - torch.linspace(0,1,T_STEPS).unsqueeze(1))*z0 + torch.linspace(0,1,T_STEPS).unsqueeze(1)*z1
    z_i = z_t[1:-1].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z_i], lr=lr)

    for ep in range(EPOCH):
        optimizer.zero_grad()
        z_t_full = torch.cat([z0, z_i, z1], dim=0)
        z_dot = (z_t_full[1:] - z_t_full[:-1])/dt
        energy = metric.kinetic(z_t_full[:-1], z_dot)
        energy_inv = metric.kinetic(torch.flip(z_t_full, [0])[:-1], torch.flip(z_dot, [0]))
        loss = (0.5*(energy+energy_inv)*dt).sum()
        loss.backward()
        optimizer.step()
    return torch.cat([z0, z_i.detach(), z1], dim=0)

# ---------- Plot Function ----------
def plot_trajectories(energy_landscape, pos, trajectories, MEAN):
    x_p = pos[:,0].view(62,100)
    y_p = pos[:,1].view(62,100)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.contourf(x_p, y_p, energy_landscape.view(62,100).cpu(), 20, cmap='Blues_r', alpha=0.8, levels=20)

    for metric_name, z_t in trajectories.items():
        ax.plot(z_t[:,0].cpu(), z_t[:,1].cpu(), label=metric_name)
        ax.scatter(z_t[0,0].cpu(), z_t[0,1].cpu(), color='red', s=50)
        ax.scatter(z_t[-1,0].cpu(), z_t[-1,1].cpu(), color='green', s=50)

    ax.scatter(MEAN[:,0].cpu(), MEAN[:,1].cpu(), color='black', s=10, alpha=0.5)
    ax.set_axis_off()
    plt.legend()
    plt.show()

# ---------- Main Evaluation ----------
if __name__ == '__main__':
    # Dataset 1: uniform
    mixture1, MEAN1, _ = create_gaussian_mixture()
    pos = torch.cat(torch.meshgrid(torch.linspace(-10,10,100), torch.linspace(-2.5,10,62)),dim=0).reshape(2,-1).T.to(DEVICE)
    energy1 = mixture1.energy(pos)

    # Metric definitions
    true_p = lambda x: mixture1.prob(x)
    alpha, beta = linear_normalization(true_p(pos).max(), true_p(pos).min(), 1, 1e-3)
    inv_p_metric = ConformalRiemannianMetric(lambda x: 1/(alpha*true_p(x)+beta))

    # RBF / LAND metrics
    reference_samples = mixture1.sample(1000).to(DEVICE)
    h_rbf = h_diag_RBF(n_centers=30, data_to_fit=reference_samples).to(DEVICE)
    rbf_metric = DiagonalRiemannianMetric(normalize_metric(h_rbf.h, pos))

    h_land = h_diag_Land(reference_samples, gamma=1)
    land_metric = DiagonalRiemannianMetric(normalize_metric(h_land.h, pos))

    metrics = {'inv_p': inv_p_metric, 'RBF': rbf_metric, 'LAND': land_metric}

    # Optimize geodesics
    trajectories = {}
    z0 = MEAN1[10].unsqueeze(0).to(DEVICE)
    z1 = MEAN1[-10].unsqueeze(0).to(DEVICE)
    for name, metric in metrics.items():
        print(f'Optimizing geodesic for {name}...')
        traj = optimize_geodesic(metric, z0, z1, T_STEPS=50, EPOCH=2000)
        trajectories[name] = traj

    plot_trajectories(energy1, pos, trajectories, MEAN1)

    # Repeat for Dataset 2 (non-uniform)
    linspace = torch.cat([torch.linspace(1,30,90), torch.ones(11)*30])
    weights2 = torch.cat([linspace, linspace[1:-1].flip(0)]) / torch.sum(torch.cat([linspace, linspace[1:-1].flip(0)]))
    mixture2, MEAN2, _ = create_gaussian_mixture(weights=weights2)
    energy2 = mixture2.energy(pos)

    # Metrics for mixture2
    true_p2 = lambda x: mixture2.prob(x)
    alpha, beta = linear_normalization(true_p2(pos).max(), true_p2(pos).min(), 1, 1e-3)
    inv_p_metric2 = ConformalRiemannianMetric(lambda x: 1/(alpha*true_p2(x)+beta))

    reference_samples2 = mixture2.sample(1000).to(DEVICE)
    h_rbf2 = h_diag_RBF(n_centers=50, data_to_fit=reference_samples2).to(DEVICE)
    rbf_metric2 = DiagonalRiemannianMetric(normalize_metric(h_rbf2.h, pos))

    h_land2 = h_diag_Land(reference_samples2, gamma=1)
    land_metric2 = DiagonalRiemannianMetric(normalize_metric(h_land2.h, pos))

    metrics2 = {'inv_p': inv_p_metric2, 'RBF': rbf_metric2, 'LAND': land_metric2}

    trajectories2 = {}
    z0 = MEAN2[10].unsqueeze(0).to(DEVICE)
    z1 = MEAN2[-10].unsqueeze(0).to(DEVICE)
    for name, metric in metrics2.items():
        print(f'Optimizing geodesic for {name} (non-uniform)...')
        traj = optimize_geodesic(metric, z0, z1, T_STEPS=50, EPOCH=2000)
        trajectories2[name] = traj

    plot_trajectories(energy2, pos, trajectories2, MEAN2)

    # Placeholder for score-based metrics
    print('Add your custom score-based metrics evaluation here.')
