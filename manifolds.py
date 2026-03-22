import torch
from stochman.manifold import Manifold
import torch.nn as nn
from sklearn.cluster import KMeans


class DiagonalManifold(Manifold):
    def __init__(self, h_func):
        super().__init__()
        self.h_func = h_func

    def metric(self, points: torch.Tensor):
        if points.dim() == 2:
            points = points.unsqueeze(0)  # (B=1, N, D)

        B, N, D = points.shape
        points_flat = points.reshape(-1, D)  
        h_vals = self.h_func(points_flat)  # (B*N, D)
        h_vals = torch.clamp(h_vals, min=1e-6)

        G = torch.diag_embed(h_vals)  # (B*N, D, D)

        
        G = G.view(B, N, D, D)

        # conditional collapse only if B==1
        if B == 1:
            return G.view(N, D, D)  # (N, D, D) for convenience
        else:
            return G  # (B, N, D, D)


class ScoreBasedManifold(Manifold):
    def __init__(self, score_fn, metric_type="magnitude",
                 alpha=1.0, beta=1.0, gamma=1.0, p=1, t=None):
        super().__init__()
        self.score_fn = score_fn
        self.metric_type = metric_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.p = p
        self.t = t

    def metric(self, points: torch.Tensor):
        if points.dim() == 2:
            points = points.unsqueeze(0)  # (B=1, N, D)

        B, N, D = points.shape
        points_flat = points.reshape(-1, D)

         # pass t if available, else None
        if self.t is not None:
            if self.t.numel() == 1:  # single time for all points
                t_flat = self.t.repeat(points_flat.shape[0])
            else:
                t_flat = self.t.reshape(-1)
            scores = self.score_fn(points_flat, t_flat)  # (B*N, D)
        else:
            scores = self.score_fn(points_flat, None)
        scores = scores.view(B, N, D)

        I = torch.eye(D, device=points.device).view(1, 1, D, D)
        I = I.expand(B, N, D, D)

        norm_sq = (scores ** 2).sum(dim=2, keepdim=True)  # (B,N,1)
        norm = torch.sqrt(norm_sq + 1e-12)
        outer = scores.unsqueeze(3) * scores.unsqueeze(2)  # (B,N,D,D)


        if self.metric_type == "normal":
            G = I - outer / (1.0 + norm_sq.unsqueeze(-1))
        elif self.metric_type == "magnitude":
            scale = self.alpha * (norm ** self.p) + self.beta
            G = scale.unsqueeze(-1) * I
        elif self.metric_type == "tangent":
            G = I + outer
        elif self.metric_type == "mixed":
            scale = self.alpha * (norm ** self.p) + self.beta
            G_mag = scale.unsqueeze(-1) * I
            G_tan = I + outer
            G = G_mag + self.gamma * G_tan
        else:
            raise ValueError(f"Unknown metric_type: {self.metric_type}")

        # Conditional: collapse batch dimension if B=1
        G = G.view(B, N, D, D)

        # collapse only if B==1
        if B == 1:
            return G.view(N, D, D)
        else:
            return G


class RBFManifold(DiagonalManifold):
    def __init__(self, h_rbf):
        super().__init__(h_rbf)

class LandManifold(DiagonalManifold):
    def __init__(self, h_land):
        super().__init__(h_land)

def normalize_metric(h_func, pos, mini=1e-3, maxi=1):
    alpha, beta = normalize_diag(h_func, pos, mini=mini, maxi=maxi)

    def h_normalized(x):
        return 1 / (alpha.to(x.device) * h_func(x) + beta.to(x.device))

    return h_normalized

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
            clusters = torch.tensor(
                clustering_model.cluster_centers_,
                dtype=torch.float32,
                device=data_to_fit.device
            )
            self.register_buffer('C', clusters)
            labels = torch.tensor(
                clustering_model.labels_,
                device=data_to_fit.device
            )
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
        return torch.exp(-pairwise_sq_dist / (2*self.gamma**2 + 1e-8))

    def h(self, x):
        weights = self.weighting_function(x)
        diffs = self.reference_sample[None,:,:] - x[:,None,:]
        return torch.einsum('bn,bnd->bd', weights, diffs**2)

def linear_normalization(maxi, mini, target_max, target_min):
    alpha = (target_max - target_min)/(maxi - mini)
    beta = target_min - alpha*mini
    return alpha, beta

def normalize_diag(h, dataset_sample, mini=1e-4, maxi=2):
    all_h = h(dataset_sample)

    if all_h.dim() != 2:
        raise ValueError("h must return shape (N,D)")

    alpha, beta = [], []
    for i in range(all_h.shape[1]):
        a, b = linear_normalization(
            all_h[:, i].max(),
            all_h[:, i].min(),
            maxi,
            mini
        )
        alpha.append(a)
        beta.append(b)

    return torch.tensor(alpha).unsqueeze(0), torch.tensor(beta).unsqueeze(0)