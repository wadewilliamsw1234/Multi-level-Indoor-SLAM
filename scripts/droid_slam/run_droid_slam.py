#!/usr/bin/env python3
"""Run DROID-SLAM and save trajectory in TUM format."""

import sys
sys.path.append('/opt/DROID-SLAM')
sys.path.append('/opt/DROID-SLAM/droid_slam')

import os
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from droid import Droid

def image_stream(imagedir, calib, stride, target_size=None):
    """Image generator - resizes to dimensions divisible by 8."""
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    
    image_list = sorted(glob.glob(os.path.join(imagedir, "*.png"))) + \
                 sorted(glob.glob(os.path.join(imagedir, "*.jpg")))
    
    orig_w, orig_h = 720, 540
    
    for t, imfile in enumerate(image_list[::stride]):
        img = Image.open(imfile)
        
        if target_size is not None:
            img = img.resize(target_size, Image.BILINEAR)
            scale_x = target_size[0] / orig_w
            scale_y = target_size[1] / orig_h
        else:
            scale_x, scale_y = 1.0, 1.0
        
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y
        
        image = torch.from_numpy(np.array(img)).permute(2,0,1)
        intrinsics = torch.as_tensor([fx_scaled, fy_scaled, cx_scaled, cy_scaled])
        timestamp = float(os.path.basename(imfile).replace('.png', '').replace('.jpg', ''))
        
        yield timestamp, image[None], intrinsics

def save_trajectory_tum(timestamps, poses, output_path):
    with open(output_path, 'w') as f:
        for ts, pose in zip(timestamps, poses):
            f.write(f"{ts:.6f} {pose[0]:.9f} {pose[1]:.9f} {pose[2]:.9f} "
                   f"{pose[3]:.9f} {pose[4]:.9f} {pose[5]:.9f} {pose[6]:.9f}\n")
    print(f"Saved {len(timestamps)} poses to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="/opt/DROID-SLAM/models/droid.pth")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--buffer", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=4.0)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=16)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    args = parser.parse_args()
    
    args.stereo = False
    args.upsample = False
    args.disable_vis = True
    args.image_size = [320, 448]
    args.frontend_device = "cuda"
    args.backend_device = "cuda"
    
    torch.multiprocessing.set_start_method('spawn')
    
    droid = None
    timestamps = []
    target_size = (448, 320)
    
    print(f"Processing images from {args.imagedir}")
    print(f"Resizing to {target_size[0]}x{target_size[1]} to fit 8GB VRAM")
    
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride, target_size=target_size)):
        timestamps.append(t)
        
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)
    
    print("Running final optimization...")
    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride, target_size=target_size))
    
    # Handle both tensor and numpy array cases
    if hasattr(traj_est, 'cpu'):
        poses = traj_est.cpu().numpy()
    else:
        poses = np.array(traj_est)
    
    n_poses = len(poses)
    if n_poses != len(timestamps):
        print(f"Note: {len(timestamps)} frames -> {n_poses} keyframe poses")
        timestamps = timestamps[:n_poses]
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_trajectory_tum(timestamps, poses, args.output)

if __name__ == "__main__":
    main()
