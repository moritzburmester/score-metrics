import torch
import torch.nn as nn
import math
import torch.distributions as dist

class GaussianMixture(nn.Module):
    """
    Multivariate Gaussian mixture with efficient probability, energy, and score computations.
    All operations are device-safe for CPU or CUDA.
    """
    def __init__(self, center_data, covar, weight=None):
        super().__init__()
        device = center_data.device

        # Core parameters
        self.register_buffer('means', center_data)           # (n_gaussians, D)
        self.register_buffer('covs', covar)                  # (n_gaussians, D, D)
        self.n_gaussians = center_data.shape[0]
        self.dim = center_data.shape[1]

        # Inverse covariances and log determinants
        self.register_buffer('precisions', torch.linalg.inv(covar))  # (n_gaussians, D, D)
        self.register_buffer('log_dets', torch.logdet(covar))       # (n_gaussians,)

        # Weights: uniform if not provided
        if weight is None:
            weight = torch.ones(self.n_gaussians, device=device) / self.n_gaussians
        else:
            weight = weight.to(device)
        self.register_buffer('weight', weight)

        # Constant for Gaussian density
        self.register_buffer('const', torch.tensor(self.dim * torch.log(torch.tensor(2.0 * torch.pi)), device=device))

    def prob(self, x: torch.Tensor):
        """
        Probability p(x) under the mixture. Returns (batch_size,)
        """
        x = x.to(self.means.device)
        delta = x.unsqueeze(1) - self.means.unsqueeze(0)                # (B, n_gaussians, D)
        maha = torch.einsum('bni,nij,bnj->bn', delta, self.precisions, delta)
        log_prob_each = -0.5 * (maha + self.log_dets + self.const)
        weighted_log_probs = log_prob_each + torch.log(self.weight)
        log_prob = torch.logsumexp(weighted_log_probs, dim=1)
        return torch.exp(log_prob)

    def energy(self, x: torch.Tensor):
        """
        Energy function: -log p(x)
        """
        return -torch.log(self.prob(x) + 1e-9)

    def score_log_p(self, x: torch.Tensor):
        """
        Score of log probability: ∇_x log p(x)
        """
        x = x.requires_grad_(True)
        logp = torch.log(self.prob(x) + 1e-9)
        grad = torch.autograd.grad(logp.sum(), x, create_graph=True)[0]
        return grad

    def score_p(self, x: torch.Tensor):
        """
        Gradient of probability p(x)
        """
        x = x.requires_grad_(True)
        p = self.prob(x)
        grad = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        return grad

    def p_log_p(self, x: torch.Tensor):
        """
        p(x) * log p(x)
        """
        p = self.prob(x)
        return p * torch.log(p + 1e-9)

    def score_p_log_p(self, x: torch.Tensor):
        """
        Gradient of p(x) * log p(x)
        """
        x = x.requires_grad_(True)
        p_logp = self.p_log_p(x)
        grad = torch.autograd.grad(p_logp.sum(), x, create_graph=True)[0]
        return grad

    def sample(self, num_samples: int, gaussian_number=None, cov_div=1):
        """
        Sample points from the mixture.
        If gaussian_number is specified, samples only from that component.
        """
        device = self.means.device
        if gaussian_number is not None:
            sampling_cat = torch.zeros(self.n_gaussians, device=device)
            sampling_cat[gaussian_number] = 1.0
        else:
            sampling_cat = self.weight

        cat = dist.Categorical(sampling_cat)
        idx = cat.sample((num_samples,))                       # (num_samples,)

        means = self.means[idx]                                # (num_samples, D)
        covs = self.covs[idx] / cov_div                        # (num_samples, D, D)

        eps = torch.randn(num_samples, self.dim, device=device)
        L = torch.linalg.cholesky(covs)
        samples = means + torch.einsum('bij,bj->bi', L, eps)
        return samples


if __name__ == "__main__":
    dataset = 'circle'
    if dataset == 'circle':
        nb_gaussians = 12
        mean_ = (torch.linspace(0, 360, nb_gaussians + 1)[0:-1] * math.pi / 180)
        mean = 8 * torch.stack([torch.cos(mean_), torch.sin(mean_)], dim=1)

        covar = torch.tensor([[1.0, 0], [0, 1.0]]).unsqueeze(0).repeat(len(mean), 1, 1)

        mixture = GaussianMixture(center_data=mean, covar=covar)

        y_p, x_p = torch.meshgrid(torch.linspace(-10, 10, 100), torch.linspace(-10, 10, 100), indexing='xy')
        pos = torch.cat([x_p.flatten().unsqueeze(1), y_p.flatten().unsqueeze(1)], dim=1)
        proba_landscape = mixture.prob(pos)
        energy_landcape = mixture.energy(pos)
        samples = mixture.sample(1000)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(3*4 + 2, 4), dpi=100)
        ax[0].scatter(samples[:, 0], samples[:, 1])
        im = ax[1].contourf(y_p, x_p, proba_landscape.view(100, 100).detach().cpu(), 20,
                            cmap='viridis',
                            alpha=0.6,
                            zorder=0,
                            levels=50)
        fig.colorbar(im, ax=ax[1])
        #im = ax[2].contour(y_p, x_p, energy_landcape.view(100, 100).detach().cpu(), levels=10, cmap='jet', linewidths=1.2, alpha=0.5)
        im = ax[2].contourf(y_p, x_p, energy_landcape.view(100, 100).detach().cpu(), 20,
                            cmap='cividis',
                            alpha=0.4,
                            zorder=0,
                           levels=10)
        fig.colorbar(im, ax=ax[2])
        plt.show()

        #samples = mixture.sample(1000, gaussian_number=4)
        #fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
        #ax.scatter(samples[:, 0], samples[:, 1])
        #plt.show()

    elif dataset == 'path':
        nb_gaussians = 12
        mean = torch.tensor([
            [ 6.7964, -4.5376],
            [ 1.9683, -3.7165],
            [-0.7590,  4.0027],
            [ 0.3242, -6.7576],
            [-5.2979, -1.0563],
            [ 5.2038, -7.2539],
            [ 5.7657, -6.2070],
            [ 7.1202, -0.1358],
            [-4.3318,  6.0616],
            [-5.4306,  7.7579],
            [-6.1806,  1.8314],
            [ 5.4688, -5.2565]])
        #mean_ = (torch.linspace(0, 360, nb_gaussians + 1)[0:-1] * math.pi / 180)
        #mean = 8 * torch.stack([torch.cos(mean_), torch.sin(mean_)], dim=1)

        covar = torch.tensor([[1.0, 0], [0, 1.0]]).unsqueeze(0).repeat(len(mean), 1, 1)

        mixture = GaussianMixture(center_data=mean, covar=covar)

        y_p, x_p = torch.meshgrid(torch.linspace(-10, 10, 100), torch.linspace(-10, 10, 100), indexing='xy')
        pos = torch.cat([x_p.flatten().unsqueeze(1), y_p.flatten().unsqueeze(1)], dim=1)
        proba_landscape = mixture.prob(pos)
        energy_landcape = mixture.energy(pos)
        samples = mixture.sample(1000)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(3 * 4 + 2, 4), dpi=100)
        ax[0].scatter(samples[:, 0], samples[:, 1])
        im = ax[1].contourf(y_p, x_p, proba_landscape.view(100, 100).detach().cpu(), 20,
                            cmap='viridis',
                            alpha=0.6,
                            zorder=0,
                            levels=50)
        fig.colorbar(im, ax=ax[1])
        im = ax[2].contourf(y_p, x_p, energy_landcape.view(100, 100).detach().cpu(), 20,
                            cmap='viridis',
                            alpha=0.6,
                            zorder=0,
                            levels=50)
        fig.colorbar(im, ax=ax[2])
        plt.show()
    a=1