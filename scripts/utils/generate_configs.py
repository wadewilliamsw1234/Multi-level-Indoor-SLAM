#!/usr/bin/env python3
"""
Generate Algorithm Configurations
==================================

Generates configuration files for all SLAM algorithms from the ISEC dataset
calibration files.

Usage:
    python3 generate_configs.py --calib-dir /data/ISEC --output-dir /config
"""

import argparse
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from calib_converter import (
    load_kalibr_cameras,
    load_camera_imu_calib,
    load_imu_params,
    compute_stereo_baseline,
    convert_to_orbslam3,
    convert_to_vins_fusion,
    convert_to_basalt,
)


def generate_lego_loam_config(output_path: Path):
    """
    Generate LeGO-LOAM configuration for Ouster OS-128.
    """
    config = """# LeGO-LOAM Configuration for ISEC Dataset
# Ouster OS-128 LiDAR Parameters

# Sensor Configuration
N_SCAN: 128
Horizon_SCAN: 1024
ang_res_x: 0.35156
ang_res_y: 0.35156
ang_bottom: 22.5
groundScanInd: 50

# Topics
pointCloudTopic: "/ouster/points"
imuTopic: "/vectornav/imu"

# Loop Closure (disabled for benchmarking)
loopClosureEnableFlag: false
"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(config)


def generate_all_configs(calib_dir: Path, output_dir: Path):
    """
    Generate configuration files for all SLAM algorithms.
    """
    calib_dir = Path(calib_dir)
    output_dir = Path(output_dir)
    
    cams_calib = calib_dir / 'cams_calib.yaml'
    cam2_imu_calib = calib_dir / 'cam2_imu_calib.yaml'
    imu_params_file = calib_dir / 'imu_params.yaml'
    
    if not cams_calib.exists():
        print(f"ERROR: Camera calibration not found: {cams_calib}")
        return False
    
    print(f"Loading calibration from: {calib_dir}")
    
    cameras = load_kalibr_cameras(cams_calib)
    print(f"  Found {len(cameras)} cameras")
    
    baseline = compute_stereo_baseline(cameras, 'cam1', 'cam3')
    print(f"  Stereo baseline (cam1-cam3): {baseline:.4f}m")
    
    T_cam_imu = None
    if cam2_imu_calib.exists():
        T_cam_imu = load_camera_imu_calib(cam2_imu_calib)
        print(f"  Loaded camera-IMU calibration")
    
    imu_params = None
    if imu_params_file.exists():
        imu_params = load_imu_params(imu_params_file)
        print(f"  Loaded IMU parameters (rate: {imu_params.rate_hz}Hz)")
    
    # Create output directories
    (output_dir / 'orb_slam3').mkdir(parents=True, exist_ok=True)
    (output_dir / 'vins_fusion').mkdir(parents=True, exist_ok=True)
    (output_dir / 'basalt').mkdir(parents=True, exist_ok=True)
    (output_dir / 'lego_loam').mkdir(parents=True, exist_ok=True)
    
    # Generate ORB-SLAM3 config
    print("\nGenerating ORB-SLAM3 config...")
    orb_output = output_dir / 'orb_slam3' / 'isec_stereo.yaml'
    convert_to_orbslam3(cameras, 'cam1', 'cam3', orb_output)
    print(f"  Saved: {orb_output}")
    
    # Generate VIO configs if IMU calibration available
    if T_cam_imu is not None and imu_params is not None:
        print("\nGenerating VINS-Fusion config...")
        vins_output = output_dir / 'vins_fusion' / 'isec_stereo_imu.yaml'
        convert_to_vins_fusion(cameras, T_cam_imu, imu_params, 'cam1', 'cam3', vins_output)
        print(f"  Saved: {vins_output}")
        
        print("\nGenerating Basalt config...")
        basalt_output = output_dir / 'basalt' / 'isec_config.json'
        convert_to_basalt(cameras, T_cam_imu, imu_params, 'cam1', 'cam3', basalt_output)
        print(f"  Saved: {basalt_output}")
    
    # Generate LeGO-LOAM config
    print("\nGenerating LeGO-LOAM config...")
    lego_output = output_dir / 'lego_loam' / 'isec_ouster.yaml'
    generate_lego_loam_config(lego_output)
    print(f"  Saved: {lego_output}")
    
    print("\nConfiguration generation complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate SLAM algorithm configs')
    parser.add_argument('--calib-dir', type=Path, default=Path('/data/ISEC'),
                       help='Directory containing calibration files')
    parser.add_argument('--output-dir', type=Path, default=Path('/config'),
                       help='Output directory for configs')
    
    args = parser.parse_args()
    
    success = generate_all_configs(args.calib_dir, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
