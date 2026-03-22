import torch
from stochman.manifold import Manifold

class ScoreBasedManifold(Manifold):
    def __init__(self, score_fn, metric_type="magnitude",
                 alpha=1.0, beta=1.0, gamma=1.0, p=1):
        super().__init__()
        self.score_fn = score_fn
        self.metric_type = metric_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.p = p

    def metric(self, points: torch.Tensor):
        """
        points: (B,N,D) or (N,D)
        returns: (B,N,D,D)
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)

        B, N, D = points.shape
        points_flat = points.reshape(-1, D)

        # ---- score ----
        scores = self.score_fn(points_flat, None)  # (B*N, D)
        scores = scores.view(B, N, D)

        # ---- identity ----
        I = torch.eye(D, device=points.device).view(1, 1, D, D)
        I = I.expand(B, N, D, D)

        # ---- common terms ----
        norm_sq = (scores ** 2).sum(dim=2, keepdim=True)  # (B,N,1)
        norm = torch.sqrt(norm_sq + 1e-12)               # stable
        outer = scores.unsqueeze(3) * scores.unsqueeze(2)  # (B,N,D,D)

        # ---- metric types ----
        if self.metric_type == "normal":
            # G = I - ss^T / (1 + ||s||^2)
            G = I - outer / (1.0 + norm_sq.unsqueeze(-1))

        elif self.metric_type == "magnitude":
            # G = (alpha * ||s||^p + beta) * I
            scale = self.alpha * (norm ** self.p) + self.beta
            G = scale.unsqueeze(-1) * I

        elif self.metric_type == "tangent":
            # G = I + ss^T
            G = I + outer

        elif self.metric_type == "mixed":
            # G = (alpha||s||^p + beta)I + gamma(I + ss^T)
            scale = self.alpha * (norm ** self.p) + self.beta
            G_mag = scale.unsqueeze(-1) * I
            G_tan = I + outer
            G = G_mag + self.gamma * G_tan

        else:
            raise ValueError(f"Unknown metric_type: {self.metric_type}")

        return G