from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd

import matplotlib
import importlib
import os
import functools
import itertools
import torch
from score_sde_pytorch.losses import get_optimizer
from score_sde_pytorch.models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tqdm
import io
import score_sde_pytorch.likelihood as likelihood
import score_sde_pytorch.controllable_generation as controllable_generation
from score_sde_pytorch.utils import restore_checkpoint
import score_sde_pytorch.models as models
from score_sde_pytorch.models import utils as mutils
from score_sde_pytorch.models import ncsnv2
from score_sde_pytorch.models import ncsnpp
from score_sde_pytorch.models import ddpm as ddpm_model
from score_sde_pytorch.models import layerspp
from score_sde_pytorch.models import layers
from score_sde_pytorch.models import normalization
import score_sde_pytorch.sampling as sampling
from score_sde_pytorch.likelihood import get_likelihood_fn
from score_sde_pytorch.sde_lib import VESDE, VPSDE, subVPSDE
from score_sde_pytorch.sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import score_sde_pytorch.datasets as datasets
from score_sde_pytorch.configs.ve import celebahq_256_ncsnpp_continuous as configs

# load score model

sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}

ckpt_path = "logs/pretrained/checkpoint_48.pth"
config = configs.get_config()  
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
sde.discrete_sigmas = sde.discrete_sigmas.to(config.device)
sampling_eps = 1e-5

batch_size =   4 #@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_path, state, config.device)
ema.copy_to(score_model.parameters())


def image_grid(x):
  size = config.data.image_size
  channels = config.data.num_channels
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img

def show_samples(x):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.show()

def save_samples(x, save_dir="samples", filename="sample.png"):
    os.makedirs(save_dir, exist_ok=True)
    
    x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
    img = image_grid(x)
    
    save_path = os.path.join(save_dir, filename)
    
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Saved samples to {save_path}")

#@title PC sampling
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.075 #@param {"type": "number"}
n_steps =  1#@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)

x, n = sampling_fn(score_model)

save_samples(x, save_dir="samples", filename="sample_0.png")
