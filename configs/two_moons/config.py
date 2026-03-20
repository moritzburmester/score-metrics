# coding=utf-8
import ml_collections
import torch
from datetime import timedelta
from configs.default import get_default_configs


def get_config():
    config = get_default_configs()

    # logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_path = 'logs/two_moons/'
    logging.log_name = 'two_moons_exp1'
    logging.top_k = 5
    logging.every_n_epochs = 50
    logging.envery_timedelta = timedelta(minutes=1)

    logging.svd_frequency = 50
    logging.save_svd = True
    logging.svd_points = 25

    # training
    training = config.training
    training.mode = 'train'
    training.gpus = 1
    training.accelerator = 'gpu'
    training.lightning_module = 'base'
    training.batch_size = 500
    training.num_epochs = int(1e20)
    training.n_iters = int(1e20)
    training.likelihood_weighting = True
    training.continuous = True
    training.sde = 'vesde'

    training.visualization_callback = ['ScoreSpectrumVisualization']
    training.show_evolution = False

    # validation
    validation = config.validation
    validation.batch_size = 500

    # sampling (IMPORTANT: better than your circle setup)
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'none'
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.15

    # data
    config.data = data = ml_collections.ConfigDict()
    data.datamodule = 'TwoMoons'
    data.create_dataset = False
    data.split = [0.8, 0.1, 0.1]
    data.data_samples = 50000
    data.use_data_mean = False

    data.noise_std = 0
    data.dim = 2
    data.num_channels = 0
    data.shape = [2]

    # model
    config.model = model = ml_collections.ConfigDict()
    model.checkpoint_path = "logs/two_moons/two_moons_exp1/checkpoints/best/epoch=3463--eval_loss_epoch=4.679.ckpt"
    model.sigma_max = 4
    model.sigma_min = 1e-2

    model.name = 'fcn'
    model.state_size = 2
    model.hidden_layers = 3
    model.hidden_nodes = 256
    model.dropout = 0.0
    model.scale_by_sigma = False
    model.num_scales = 1000
    model.ema_rate = 0.9999

    # optimization
    optim = config.optim
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    config.dim_estimation = ml_collections.ConfigDict()

    return config