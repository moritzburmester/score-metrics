import torch
from stochman.manifold import Manifold

class ScoreBasedManifold(Manifold):
    """
    A manifold whose metric is defined from a score function.
    Supports different variants:
        - 'norm': g_norm
        - 'tangent': g_tangent
        - 'magnitude': g_magnitude
        - 'custom': user-defined alpha, beta scaling
    """

    def __init__(self, score_fn, metric_type="magnitude", alpha=1.0, beta=1.0):
        super().__init__()
        self.score_fn = score_fn
        self.metric_type = metric_type
        self.alpha = alpha
        self.beta = beta

    def metric(self, points: torch.Tensor):
        """
        Compute the metric tensor G at points, using the chosen score metric.
        points: NxD or BxNxD
        """
        orig_shape = points.shape
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dim
        B, N, D = points.shape
        points_flat = points.view(-1, D)

        # ----- Compute score -----
        ts = torch.zeros(points_flat.shape[0], device=points.device)  # optionally pass time
        scores = self.score_fn(points_flat, ts)  # (B*N)xD
        scores = scores.view(B, N, D)

        # ----- Build metric -----
        I = torch.eye(D, device=points.device).unsqueeze(0).expand(B, D, D)  # BxDxD

        if self.metric_type == "norm":
            # g_norm: G = I - outer / (1 + ||s||^2)
            norm_sq = (scores**2).sum(dim=2, keepdim=True)  # BxN x1
            outer = scores.unsqueeze(3) * scores.unsqueeze(2)  # BxN x D x D
            G = I.unsqueeze(1) - outer / (1 + norm_sq.unsqueeze(-1))  # B x N x D x D

        elif self.metric_type == "tangent":
            # g_tangent: G = I + s s^T
            G = I.unsqueeze(1) + scores.unsqueeze(3) * scores.unsqueeze(2)

        elif self.metric_type == "magnitude":
            # g_magnitude: G = (alpha*||s||^4 + beta) * I
            norm = scores.norm(dim=2, keepdim=True)  # B x N x 1
            scale = self.alpha * (norm**4) + self.beta
            G = scale.unsqueeze(-1) * I.unsqueeze(1)

        else:
            raise ValueError(f"Unknown metric_type: {self.metric_type}")

        if orig_shape == points.shape[1:]:
            return G[0]  # remove batch dim
        return G  # B x N x D x D