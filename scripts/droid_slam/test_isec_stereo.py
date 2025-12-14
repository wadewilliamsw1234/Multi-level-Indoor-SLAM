"""
DROID-SLAM evaluation script for ISEC dataset (stereo mode)
Based on test_euroc.py pattern

Usage:
    python test_isec_stereo.py --datapath /results/images/5th_floor --stereo --disable_vis
"""

import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from pathlib import Path
from torch.multiprocessing import Process
from droid import Droid
from droid_async import DroidAsync

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def image_stream(datapath, image_size=[320, 512], stereo=False, stride=1):
    """
    Image generator for ISEC dataset
    
    Expected directory structure:
        datapath/
            left/
                000000.png
                000001.png
                ...
            right/
                000000.png
                000001.png
                ...
    """
    
    # ISEC camera calibration for cam1 (left) and cam3 (right)
    # From cams_calib.yaml - these are the ORIGINAL intrinsics
    # cam1 intrinsics: [fx=893.63, fy=893.97, cx=376.95, cy=266.57]
    # cam3 intrinsics: [fx=890.41, fy=890.60, cx=370.45, cy=281.40]
    # Baseline: 0.328m
    
    # Original image size
    ht0, wd0 = 540, 720
    
    # Left camera (cam1) calibration
    K_l = np.array([
        893.6263545058326, 0.0, 376.95348001716707,
        0.0, 893.9655105687939, 266.57152598273194,
        0.0, 0.0, 1.0
    ]).reshape(3, 3)
    
    d_l = np.array([-0.21272110177039052, 0.18283401892861978, 
                    -0.00018083866109219808, 0.0011164116025029272, 0.0])
    
    # Right camera (cam3) calibration  
    K_r = np.array([
        890.4086004258235, 0.0, 370.4507082650829,
        0.0, 890.6037389430937, 281.39530822498827,
        0.0, 0.0, 1.0
    ]).reshape(3, 3)
    
    d_r = np.array([-0.21797193098489498, 0.19738427890710094,
                    -0.00024175998655538, 0.0007953028907710908, 0.0])
    
    # For stereo, we need to rectify the images
    # Baseline between cam1 and cam3: ~0.328m
    # We'll compute rectification maps
    
    # Rotation and translation from cam1 to cam3 (from cams_calib.yaml T_cn_cnm1 chain)
    # This is approximate - ideally load from actual calibration
    R = np.eye(3)  # Cameras are roughly parallel
    T = np.array([0.328379, 0.000510, 0.001146])  # Translation vector (baseline)
    
    # Compute stereo rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_l, d_l, K_r, d_r, (wd0, ht0), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    
    # Create undistort+rectify maps
    map_l = cv2.initUndistortRectifyMap(K_l, d_l, R1, P1, (wd0, ht0), cv2.CV_32F)
    map_r = cv2.initUndistortRectifyMap(K_r, d_r, R2, P2, (wd0, ht0), cv2.CV_32F)
    
    # Rectified intrinsics (from P1 matrix)
    fx = P1[0, 0]
    fy = P1[1, 1]
    cx = P1[0, 2]
    cy = P1[1, 2]
    intrinsics_vec = [fx, fy, cx, cy]
    
    print(f"Rectified intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"Baseline: {abs(P2[0,3]/P2[0,0]):.4f}m")
    
    # Find images
    left_dir = os.path.join(datapath, 'left')
    right_dir = os.path.join(datapath, 'right')
    
    if not os.path.exists(left_dir):
        # Maybe images are directly in datapath with naming convention
        images_left = sorted(glob.glob(os.path.join(datapath, '*_left.png')))
        if not images_left:
            images_left = sorted(glob.glob(os.path.join(datapath, 'left*.png')))
        if not images_left:
            # Try numbered format
            images_left = sorted(glob.glob(os.path.join(datapath, '*.png')))
            print(f"Warning: Using all images from {datapath}, assuming monocular")
    else:
        images_left = sorted(glob.glob(os.path.join(left_dir, '*.png')))
        
    if stereo:
        if os.path.exists(right_dir):
            images_right = sorted(glob.glob(os.path.join(right_dir, '*.png')))
        else:
            images_right = [x.replace('left', 'right') for x in images_left]
    
    print(f"Found {len(images_left)} left images")
    if stereo:
        print(f"Found {len(images_right)} right images")
    
    # Apply stride
    images_left = images_left[::stride]
    if stereo:
        images_right = images_right[::stride]
    
    data_list = []
    for t, imgL_path in enumerate(tqdm(images_left, desc="Loading images")):
        # Extract timestamp from filename (assuming format like 000000.png or timestamp.png)
        fname = os.path.basename(imgL_path)
        try:
            tstamp = float(fname.replace('.png', '').replace('.jpg', ''))
        except:
            tstamp = float(t)
        
        # Load and rectify left image
        imgL = cv2.imread(imgL_path)
        if imgL is None:
            print(f"Warning: Could not read {imgL_path}")
            continue
            
        imgL_rect = cv2.remap(imgL, map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)
        images = [imgL_rect]
        
        # Load and rectify right image if stereo
        if stereo:
            imgR_path = images_right[t] if t < len(images_right) else None
            if imgR_path and os.path.isfile(imgR_path):
                imgR = cv2.imread(imgR_path)
                if imgR is not None:
                    imgR_rect = cv2.remap(imgR, map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)
                    images.append(imgR_rect)
                else:
                    continue
            else:
                continue
        
        # Resize to target size
        images = [cv2.resize(img, (image_size[1], image_size[0])) for img in images]
        
        # Convert to tensor [N, 3, H, W]
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to(dtype=torch.float32)
        
        # Scale intrinsics
        intrinsics = torch.as_tensor(intrinsics_vec).float()
        intrinsics[0] *= image_size[1] / wd0
        intrinsics[1] *= image_size[0] / ht0
        intrinsics[2] *= image_size[1] / wd0
        intrinsics[3] *= image_size[0] / ht0
        
        data_list.append((stride * t, images, intrinsics))
    
    print(f"Prepared {len(data_list)} frames for processing")
    return data_list


def save_trajectory_tum(traj_est, tstamps, output_path):
    """Save trajectory in TUM format"""
    with open(output_path, 'w') as f:
        for t, pose in zip(tstamps, traj_est):
            # pose is [tx, ty, tz, qx, qy, qz, qw] or [tx, ty, tz, qw, qx, qy, qz]
            # TUM format: timestamp tx ty tz qx qy qz qw
            if len(pose) == 7:
                tx, ty, tz = pose[0], pose[1], pose[2]
                # Check quaternion order - DROID uses wxyz internally
                qw, qx, qy, qz = pose[3], pose[4], pose[5], pose[6]
                f.write(f"{t:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    print(f"Saved trajectory to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", required=True, help="path to ISEC sequence (containing left/ and right/ subdirs)")
    parser.add_argument("--output", default="/results/trajectories/droid_slam/trajectory.txt", help="output trajectory path")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", nargs=2, type=int, default=[320, 512])
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--stride", type=int, default=1, help="frame stride for loading")
    parser.add_argument("--process_stride", type=int, default=2, help="stride for processing (like EuRoC uses 2)")

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--keyframe_thresh", type=float, default=3.0)
    parser.add_argument("--frontend_thresh", type=float, default=17.5)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)

    parser.add_argument("--backend_thresh", type=float, default=24.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=2)

    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    print("=" * 60)
    print("DROID-SLAM ISEC Stereo Evaluation")
    print("=" * 60)
    print(f"Datapath: {args.datapath}")
    print(f"Stereo: {args.stereo}")
    print(f"Image size: {args.image_size}")
    print(f"Buffer: {args.buffer}")
    print(f"Stride (load): {args.stride}, Stride (process): {args.process_stride}")
    print("=" * 60)

    # Load all images
    images = image_stream(args.datapath, image_size=args.image_size, 
                          stereo=args.stereo, stride=args.stride)
    
    if len(images) == 0:
        print("ERROR: No images loaded!")
        sys.exit(1)

    # Initialize DROID
    print("\nInitializing DROID-SLAM...")
    droid = DroidAsync(args) if args.asynchronous else Droid(args)

    # Process with stride (like EuRoC does stride=2)
    print(f"\nProcessing {len(images[::args.process_stride])} frames...")
    for (t, image, intrinsics) in tqdm(images[::args.process_stride], desc="Tracking"):
        droid.track(t, image, intrinsics=intrinsics)

    # Terminate and fill in missing poses
    print("\nRunning global optimization...")
    traj_est = droid.terminate(images)

    # Extract timestamps
    tstamps = [data[0] for data in images]

    # Save trajectory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_trajectory_tum(traj_est, tstamps, args.output)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Trajectory saved to: {args.output}")
    print(f"Total poses: {len(traj_est)}")
    
    # Print trajectory stats
    positions = traj_est[:, :3]
    trajectory_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    print(f"Trajectory length: {trajectory_length:.2f}m")
    
    # Print start/end drift
    drift = np.linalg.norm(positions[-1] - positions[0])
    print(f"Start-to-end drift: {drift:.3f}m ({100*drift/trajectory_length:.2f}%)")
    print("=" * 60)
