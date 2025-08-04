# Script to load a checkpoint, generate samples, and check thrust values in the generated data
import sys
import os
sys.path.insert(0, os.path.abspath("Reflected-Diffusion"))

import models.ncsnpp  # Ensure 'ncsnpp' model is registered

import torch
import numpy as np
import argparse
from omegaconf import OmegaConf
from models import utils as mutils
from sde_lib import RVESDE
from sampling import get_sampling_fn

def load_config(base_config_path, data_config_path, model_config_path):
    base_cfg = OmegaConf.load(base_config_path)
    # Wrap data config under 'data' if not already
    data_cfg_raw = OmegaConf.load(data_config_path)
    data_cfg = OmegaConf.create({'data': data_cfg_raw}) if 'data' not in data_cfg_raw else data_cfg_raw
    # Wrap model config under 'model' if not already
    model_cfg_raw = OmegaConf.load(model_config_path)
    model_cfg = OmegaConf.create({'model': model_cfg_raw}) if 'model' not in model_cfg_raw else model_cfg_raw
    # Merge configs: base <- data <- model
    cfg = OmegaConf.merge(base_cfg, data_cfg, model_cfg)
    return cfg

def main(checkpoint_path, base_config_path, data_config_path, model_config_path, num_samples, output_dir):
    print("[INFO] Loading config...")
    cfg = load_config(base_config_path, data_config_path, model_config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print("[INFO] Creating model and loading checkpoint...")
    score_model = mutils.create_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if 'model' in state:
        score_model.load_state_dict(state['model'], strict=False)
    else:
        score_model.load_state_dict(state, strict=False)
    score_model.eval()
    print("[INFO] Model architecture loaded:")
    print(score_model)

    print("[INFO] Setting up SDE and sampling function...")
    sde = RVESDE(sigma_min=cfg.sde.sigma_min, sigma_max=cfg.sde.sigma_max, N=cfg.sde.num_scales)
    sampling_shape = (num_samples, cfg.data.num_channels, cfg.data.image_size, cfg.data.image_width)
    sampling_fn = get_sampling_fn(cfg, sde, sampling_shape, 1e-5, device)

    print("[INFO] Preparing class labels (linspace [0,1])...")
    class_labels = np.linspace(0, 1, num_samples, dtype=np.float32).reshape(-1, 1)
    class_labels = torch.from_numpy(class_labels).to(device)
    print(f"Using class_labels: {class_labels.flatten().cpu().numpy()}")

    print("[INFO] Generating samples...")
    with torch.no_grad():
        samples, _ = sampling_fn(score_model, class_labels=class_labels)
        samples = samples.cpu().numpy()

    print("[INFO] Keeping normalized values (no unnormalization)...")
    # No mean/std unnormalization applied - keeping normalized values

    print("[INFO] Checking thrust values for each sample...")
    samples_flat = samples.reshape(num_samples, -1)
    thrust_out_of_bounds = []
    for i, vec in enumerate(samples_flat):
        vec67 = vec[:67]
        thrust = vec67[4:64]
        thrust_min, thrust_max = thrust.min(), thrust.max()
        if (thrust < 0).any() or (thrust > 1).any():
            thrust_out_of_bounds.append((i, thrust_min, thrust_max))
            print(f"Sample {i}: thrust min={thrust_min:.4f}, max={thrust_max:.4f} -> FAILURE (out of bounds)")
        else:
            print(f"Sample {i}: thrust min={thrust_min:.4f}, max={thrust_max:.4f} -> SUCCESS (all in [0,1])")

    if not thrust_out_of_bounds:
        print(f"All {num_samples} samples have thrust values within [0, 1].")
    else:
        print(f"{len(thrust_out_of_bounds)} out of {num_samples} samples have thrust values out of bounds.")

    if output_dir:
        print(f"[INFO] Saving samples to {output_dir} ...")
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "samples.npy"), samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--base_config", type=str, default="Reflected-Diffusion/configs/train.yaml", help="Path to base train.yaml config")
    parser.add_argument("--data_config", type=str, default="Reflected-Diffusion/configs/data/gto_halo.yaml", help="Path to data config YAML file")
    parser.add_argument("--model_config", type=str, default="Reflected-Diffusion/configs/model/ncsnpp.yaml", help="Path to model config YAML file")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save generated samples")
    args = parser.parse_args()
    main(args.checkpoint, args.base_config, args.data_config, args.model_config, args.num_samples, args.output_dir) 