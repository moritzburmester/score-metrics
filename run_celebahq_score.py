import torch
print(torch.cuda.is_available())
import importlib
import sde_lib
from models import utils as mutils
from sampling.sampling import get_pc_sampler
import torchvision.utils as vutils
from models import ncsnpp
from lightning_modules.utils import create_lightning_module
from lightning_modules import BaseSdeGenerativeModel
from lightning_modules.BaseSdeGenerativeModel import BaseSdeGenerativeModel
import numpy as np 
from utils import restore_checkpoint
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import os
from sampling import sampling

# -----------------------------
# 1. Load config
# -----------------------------
config_module = importlib.import_module("configs.ve.celebahq_256_ncsnpp_continuous")
config = config_module.get_config()
config.data.shape = [config.data.num_channels, config.data.image_size, config.data.image_size]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.device = device
print("Device:", device)

# -----------------------------
# 2. Create model & EMA
# -----------------------------
score_model = mutils.create_model(config)
optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)

state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)

# -----------------------------
# 3. Load checkpoint safely
# -----------------------------
ckpt_path = "logs/pretrained/checkpoint_48.pth"
loaded_state = torch.load(ckpt_path, map_location=device)

if 'model' in loaded_state:
    score_model.load_state_dict(loaded_state['model'], strict=False)
else:
    raise ValueError("Checkpoint contains neither 'model' nor 'ema' keys.")

score_model = score_model.to(device)
score_model.eval()
print("✅ Model loaded (EMA skipped)")

# -----------------------------
# 4. Setup SDE and sampler
# -----------------------------
sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,
                    sigma_max=config.model.sigma_max,
                    N=config.model.num_scales)

batch_size = 1  # number of samples
shape = (batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)

sampling_eps = 1e-5  # small epsilon for VE-SDE
predictor = sampling.ReverseDiffusionPredictor
corrector = sampling.LangevinCorrector
snr = 0.16
n_steps = 1
probability_flow = False

sampling_fn = get_pc_sampler(
    sde, shape, predictor, corrector,
    inverse_scaler=lambda x: (x + 1.) / 2.,
    snr=snr, n_steps=n_steps,
    probability_flow=probability_flow,
    continuous=True,
    eps=sampling_eps,
    device=device
)

# -----------------------------
# 5. Sample from model
# -----------------------------
with torch.no_grad():
    samples, _ = sampling_fn(score_model)

# -----------------------------
# 6. Save images
# -----------------------------
os.makedirs("samples", exist_ok=True)
vutils.save_image(samples, "samples/reference_samples.png", nrow=1)
print("✅ Samples shape:", samples.shape)
print("🎉 Saved reference_samples.png")