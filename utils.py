import torch
import os
import logging

def restore_checkpoint(ckpt_path, state, device):
    """Restore a checkpoint from a file into the given state dict."""
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_path}. Returning the same state.")
        return state

    loaded_state = torch.load(ckpt_path, map_location=device)

    # Load optimizer, model, EMA, and step if they exist
    if 'optimizer' in loaded_state:
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
    if 'model' in loaded_state:
        state['model'].load_state_dict(loaded_state['model'], strict=False)
    if 'ema' in loaded_state:
        state['ema'].load_state_dict(loaded_state['ema'])
    if 'step' in loaded_state:
        state['step'] = loaded_state['step']

    return state


def save_checkpoint(ckpt_path, state):
    """Save the state dict to a checkpoint file."""
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(saved_state, ckpt_path)
    print(f"Checkpoint saved at {ckpt_path}")