#!/usr/bin/env python3
"""
Compute correct camera-IMU transforms for ISEC dataset.

The ISEC dataset provides:
- cam2_imu_calib.yaml: T_cam_imu for cam2 (center front camera)
- cams_calib.yaml: Inter-camera transforms (cam0 as reference)

For VINS-Fusion and Basalt, we need T_imu_cam for cam1 and cam3 (stereo pair).

Transform chain:
  T_imu_cam1 = T_imu_cam2 @ T_cam2_cam1
  T_imu_cam3 = T_imu_cam2 @ T_cam2_cam3
  
Where:
  T_imu_cam2 = inv(T_cam2_imu)
  T_cam2_cam1 = inv(T_cam1_cam0) @ T_cam2_cam0  (chain through cam0)
  T_cam2_cam3 = inv(T_cam2_cam0) @ T_cam3_cam0  (chain through cam0)
"""

import numpy as np
import yaml
import json
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_kalibr_transform(yaml_data, key='T_cam_imu'):
    """Load 4x4 transform from Kalibr YAML format."""
    if isinstance(yaml_data, dict):
        data = yaml_data.get(key) or yaml_data.get('cam0', {}).get(key)
    else:
        data = yaml_data
    
    return np.array(data).reshape(4, 4)


def load_cams_calib(yaml_path):
    """Load camera calibration from Kalibr multi-camera output."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    cameras = {}
    for cam_key in ['cam0', 'cam1', 'cam2', 'cam3', 'cam4']:
        if cam_key in data:
            cam_data = data[cam_key]
            cameras[cam_key] = {
                'intrinsics': cam_data.get('intrinsics', []),
                'distortion': cam_data.get('distortion_coeffs', []),
                'resolution': cam_data.get('resolution', []),
                'T_cn_cnm1': np.array(cam_data.get('T_cn_cnm1', np.eye(4))).reshape(4, 4) 
                             if 'T_cn_cnm1' in cam_data else None
            }
    
    return cameras


def load_cam_imu_calib(yaml_path):
    """Load camera-IMU calibration from Kalibr output."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    T_cam_imu = np.array(data['cam0']['T_cam_imu']).reshape(4, 4)
    
    # IMU parameters if available
    imu_params = {}
    if 'imu0' in data:
        imu = data['imu0']
        imu_params = {
            'acc_noise': imu.get('accelerometer_noise_density', 0.001),
            'gyr_noise': imu.get('gyroscope_noise_density', 0.0001),
            'acc_walk': imu.get('accelerometer_random_walk', 0.00001),
            'gyr_walk': imu.get('gyroscope_random_walk', 0.000001),
            'rate': imu.get('update_rate', 200.0)
        }
    
    return T_cam_imu, imu_params


def invert_transform(T):
    """Invert a 4x4 homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    
    return T_inv


def transform_to_pose(T):
    """Convert 4x4 transform to position and quaternion."""
    pos = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3])
    quat = rot.as_quat()  # [x, y, z, w]
    
    return {
        'px': float(pos[0]),
        'py': float(pos[1]),
        'pz': float(pos[2]),
        'qx': float(quat[0]),
        'qy': float(quat[1]),
        'qz': float(quat[2]),
        'qw': float(quat[3])
    }


def compute_stereo_transforms(cams_calib, cam2_imu_calib):
    """
    Compute T_imu_cam1 and T_imu_cam3 for stereo pair.
    
    Args:
        cams_calib: Camera calibration dict from cams_calib.yaml
        cam2_imu_calib: T_cam2_imu from cam2_imu_calib.yaml
    
    Returns:
        T_imu_cam1, T_imu_cam3
    """
    # T_imu_cam2 = inv(T_cam2_imu)
    T_imu_cam2 = invert_transform(cam2_imu_calib)
    
    # Build transforms from cam0 (reference frame)
    # T_cam1_cam0 comes from cams_calib
    # In Kalibr format, T_cn_cnm1 gives transform from cam(n-1) to cam(n)
    
    # Assuming chain: cam0 -> cam1 -> cam2 -> cam3 -> cam4
    # T_cam1_cam0 = T_cn_cnm1 for cam1
    T_cam1_cam0 = cams_calib['cam1'].get('T_cn_cnm1')
    
    # T_cam2_cam0 = T_cam2_cam1 @ T_cam1_cam0
    T_cam2_cam1 = cams_calib['cam2'].get('T_cn_cnm1')
    if T_cam2_cam1 is not None and T_cam1_cam0 is not None:
        T_cam2_cam0 = T_cam2_cam1 @ T_cam1_cam0
    else:
        # Fallback: assume identity if not available
        T_cam2_cam0 = np.eye(4)
    
    # T_cam3_cam0 = T_cam3_cam2 @ T_cam2_cam0
    T_cam3_cam2 = cams_calib['cam3'].get('T_cn_cnm1')
    if T_cam3_cam2 is not None:
        T_cam3_cam0 = T_cam3_cam2 @ T_cam2_cam0
    else:
        T_cam3_cam0 = np.eye(4)
    
    # Now compute T_cam2_cam1 and T_cam2_cam3
    T_cam0_cam2 = invert_transform(T_cam2_cam0)
    T_cam0_cam1 = invert_transform(T_cam1_cam0) if T_cam1_cam0 is not None else np.eye(4)
    T_cam0_cam3 = invert_transform(T_cam3_cam0)
    
    # T_cam2_cam1 = T_cam2_cam0 @ T_cam0_cam1
    T_cam2_cam1_final = T_cam0_cam2 @ T_cam1_cam0 if T_cam1_cam0 is not None else np.eye(4)
    T_cam2_cam1_final = invert_transform(T_cam2_cam1_final)
    
    # T_cam2_cam3 = T_cam2_cam0 @ T_cam0_cam3 
    T_cam2_cam3 = T_cam0_cam2 @ T_cam3_cam0
    
    # Final transforms
    # T_imu_cam1 = T_imu_cam2 @ T_cam2_cam1
    T_imu_cam1 = T_imu_cam2 @ T_cam2_cam1_final
    
    # T_imu_cam3 = T_imu_cam2 @ T_cam2_cam3
    T_imu_cam3 = T_imu_cam2 @ T_cam2_cam3
    
    return T_imu_cam1, T_imu_cam3


def generate_vins_config(T_imu_cam1, T_imu_cam3, cam1_intrinsics, cam3_intrinsics,
                         imu_params, output_path):
    """Generate VINS-Fusion config file."""
    
    config = f"""%YAML:1.0

#--------------------------------------------------------------------------------------------
# Common Parameters
#--------------------------------------------------------------------------------------------

imu: 1
num_of_cam: 2

imu_topic: "/vectornav/imu"
image0_topic: "/camera_array/cam1/image_raw"
image1_topic: "/camera_array/cam3/image_raw"

output_path: "/results/vins_fusion"

#--------------------------------------------------------------------------------------------
# Camera 0 (Left - cam1) Parameters
#--------------------------------------------------------------------------------------------

model_type: PINHOLE
camera_name: camera

image_width: {cam1_intrinsics['resolution'][0]}
image_height: {cam1_intrinsics['resolution'][1]}

distortion_parameters:
    k1: {cam1_intrinsics['distortion'][0]}
    k2: {cam1_intrinsics['distortion'][1]}
    p1: {cam1_intrinsics['distortion'][2]}
    p2: {cam1_intrinsics['distortion'][3]}

projection_parameters:
    fx: {cam1_intrinsics['intrinsics'][0]}
    fy: {cam1_intrinsics['intrinsics'][1]}
    cx: {cam1_intrinsics['intrinsics'][2]}
    cy: {cam1_intrinsics['intrinsics'][3]}

#--------------------------------------------------------------------------------------------
# Extrinsic Parameters (Body/IMU to Camera)
#--------------------------------------------------------------------------------------------

body_T_cam0: !!opencv-matrix
    rows: 4
    cols: 4
    dt: d
    data: [{', '.join([f'{x:.10f}' for x in T_imu_cam1.flatten()])}]

body_T_cam1: !!opencv-matrix
    rows: 4
    cols: 4
    dt: d
    data: [{', '.join([f'{x:.10f}' for x in T_imu_cam3.flatten()])}]

#--------------------------------------------------------------------------------------------
# IMU Parameters
#--------------------------------------------------------------------------------------------

acc_n: {imu_params.get('acc_noise', 0.001)}
gyr_n: {imu_params.get('gyr_noise', 0.0001)}
acc_w: {imu_params.get('acc_walk', 0.00001)}
gyr_w: {imu_params.get('gyr_walk', 0.000001)}
g_norm: 9.81007

#--------------------------------------------------------------------------------------------
# Feature Tracking Parameters
#--------------------------------------------------------------------------------------------

max_cnt: 150
min_dist: 25
freq: 10
F_threshold: 1.0
show_track: 0
flow_back: 1

#--------------------------------------------------------------------------------------------
# Optimization Parameters
#--------------------------------------------------------------------------------------------

max_solver_time: 0.04
max_num_iterations: 8
keyframe_parallax: 10.0
estimate_extrinsic: 0  # 0: fixed, 1: optimize, 2: estimate initial

#--------------------------------------------------------------------------------------------
# Loop Closure (DISABLED for benchmarking)
#--------------------------------------------------------------------------------------------

loop_closure: 0
"""
    
    with open(output_path, 'w') as f:
        f.write(config)
    
    print(f"Generated VINS-Fusion config: {output_path}")


def generate_basalt_config(T_imu_cam1, T_imu_cam3, cam1_intrinsics, cam3_intrinsics,
                           imu_params, output_path):
    """Generate Basalt calibration JSON file."""
    
    config = {
        "value0": {
            "T_imu_cam": [
                transform_to_pose(T_imu_cam1),
                transform_to_pose(T_imu_cam3)
            ],
            "intrinsics": [
                {
                    "camera_type": "pinhole",
                    "intrinsics": {
                        "fx": cam1_intrinsics['intrinsics'][0],
                        "fy": cam1_intrinsics['intrinsics'][1],
                        "cx": cam1_intrinsics['intrinsics'][2],
                        "cy": cam1_intrinsics['intrinsics'][3]
                    },
                    "resolution": cam1_intrinsics['resolution']
                },
                {
                    "camera_type": "pinhole",
                    "intrinsics": {
                        "fx": cam3_intrinsics['intrinsics'][0],
                        "fy": cam3_intrinsics['intrinsics'][1],
                        "cx": cam3_intrinsics['intrinsics'][2],
                        "cy": cam3_intrinsics['intrinsics'][3]
                    },
                    "resolution": cam3_intrinsics['resolution']
                }
            ],
            "imu_update_rate": imu_params.get('rate', 200.0),
            "gyro_noise_std": imu_params.get('gyr_noise', 0.0001),
            "accel_noise_std": imu_params.get('acc_noise', 0.001),
            "gyro_bias_std": imu_params.get('gyr_walk', 0.000001),
            "accel_bias_std": imu_params.get('acc_walk', 0.00001)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated Basalt calibration: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate VINS-Fusion and Basalt configs from ISEC calibration')
    parser.add_argument('--cams-calib', type=str, required=True,
                       help='Path to cams_calib.yaml')
    parser.add_argument('--cam-imu-calib', type=str, required=True,
                       help='Path to cam2_imu_calib.yaml')
    parser.add_argument('--imu-params', type=str, default=None,
                       help='Path to imu_params.yaml')
    parser.add_argument('--output-dir', type=str, default='./config',
                       help='Output directory for generated configs')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load calibration data
    print("Loading calibration files...")
    cams = load_cams_calib(args.cams_calib)
    T_cam2_imu, imu_params = load_cam_imu_calib(args.cam_imu_calib)
    
    # Load additional IMU params if provided
    if args.imu_params:
        with open(args.imu_params, 'r') as f:
            extra_imu = yaml.safe_load(f)
        imu_params.update({
            'acc_noise': extra_imu.get('accelerometer_noise_density', imu_params.get('acc_noise')),
            'gyr_noise': extra_imu.get('gyroscope_noise_density', imu_params.get('gyr_noise')),
            'acc_walk': extra_imu.get('accelerometer_random_walk', imu_params.get('acc_walk')),
            'gyr_walk': extra_imu.get('gyroscope_random_walk', imu_params.get('gyr_walk')),
        })
    
    # Compute stereo transforms
    print("Computing stereo camera transforms...")
    T_imu_cam1, T_imu_cam3 = compute_stereo_transforms(cams, T_cam2_imu)
    
    print("\nT_imu_cam1:")
    print(T_imu_cam1)
    print("\nT_imu_cam3:")
    print(T_imu_cam3)
    
    # Compute baseline
    baseline = np.linalg.norm(T_imu_cam3[:3, 3] - T_imu_cam1[:3, 3])
    print(f"\nStereo baseline: {baseline:.4f} m")
    
    # Extract camera intrinsics
    cam1_intrinsics = {
        'intrinsics': cams['cam1']['intrinsics'],
        'distortion': cams['cam1']['distortion'],
        'resolution': cams['cam1']['resolution']
    }
    cam3_intrinsics = {
        'intrinsics': cams['cam3']['intrinsics'],
        'distortion': cams['cam3']['distortion'],
        'resolution': cams['cam3']['resolution']
    }
    
    # Generate configs
    print("\nGenerating configuration files...")
    
    vins_dir = output_dir / 'vins_fusion'
    vins_dir.mkdir(exist_ok=True)
    generate_vins_config(
        T_imu_cam1, T_imu_cam3,
        cam1_intrinsics, cam3_intrinsics,
        imu_params,
        vins_dir / 'isec_stereo_imu_corrected.yaml'
    )
    
    basalt_dir = output_dir / 'basalt'
    basalt_dir.mkdir(exist_ok=True)
    generate_basalt_config(
        T_imu_cam1, T_imu_cam3,
        cam1_intrinsics, cam3_intrinsics,
        imu_params,
        basalt_dir / 'isec_calib.json'
    )
    
    print("\nDone! Generated configs in:", output_dir)


if __name__ == '__main__':
    main()
