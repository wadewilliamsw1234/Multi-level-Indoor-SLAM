#!/usr/bin/env python3
"""
Calibration Converter
=====================

Converts camera and IMU calibration files between different formats:
- Kalibr (input format from ISEC dataset)
- ORB-SLAM3
- VINS-Fusion  
- Basalt (JSON format)

Based on calibration files provided with ISEC dataset:
- cams_calib.yaml: Camera intrinsics and extrinsics (Kalibr format)
- cam2_imu_calib.yaml: Camera 2 to IMU extrinsics
- imu_params.yaml: IMU noise parameters
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_model: str = 'radtan'
    distortion_coeffs: List[float] = field(default_factory=list)
    
    @classmethod
    def from_kalibr(cls, cam_data: dict) -> 'CameraIntrinsics':
        """Parse camera intrinsics from Kalibr format."""
        intrinsics = cam_data['intrinsics']
        resolution = cam_data['resolution']
        distortion = cam_data.get('distortion_coeffs', [0, 0, 0, 0])
        
        return cls(
            fx=intrinsics[0],
            fy=intrinsics[1],
            cx=intrinsics[2],
            cy=intrinsics[3],
            width=resolution[0],
            height=resolution[1],
            distortion_model=cam_data.get('distortion_model', 'radtan'),
            distortion_coeffs=distortion,
        )


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters (transformation matrix)."""
    T: np.ndarray  # 4x4 transformation matrix
    
    @classmethod
    def from_kalibr(cls, cam_data: dict) -> 'CameraExtrinsics':
        """Parse camera extrinsics from Kalibr format."""
        T = np.array(cam_data['T_cn_cnm1'])
        return cls(T=T)
    
    @classmethod
    def identity(cls) -> 'CameraExtrinsics':
        """Return identity transformation."""
        return cls(T=np.eye(4))
    
    @property
    def rotation(self) -> np.ndarray:
        """Get 3x3 rotation matrix."""
        return self.T[:3, :3]
    
    @property
    def translation(self) -> np.ndarray:
        """Get translation vector."""
        return self.T[:3, 3]
    
    def inverse(self) -> 'CameraExtrinsics':
        """Return inverse transformation."""
        T_inv = np.eye(4)
        T_inv[:3, :3] = self.T[:3, :3].T
        T_inv[:3, 3] = -self.T[:3, :3].T @ self.T[:3, 3]
        return CameraExtrinsics(T=T_inv)


@dataclass
class IMUParams:
    """IMU noise parameters."""
    gyro_noise_density: float  # rad/s/sqrt(Hz)
    gyro_random_walk: float    # rad/s^2/sqrt(Hz)
    accel_noise_density: float # m/s^2/sqrt(Hz)
    accel_random_walk: float   # m/s^3/sqrt(Hz)
    rate_hz: float = 200.0
    
    @classmethod
    def from_kalibr(cls, imu_data: dict) -> 'IMUParams':
        """Parse IMU parameters from Kalibr format."""
        return cls(
            gyro_noise_density=imu_data.get('gyroscope_noise_density', 0.0001),
            gyro_random_walk=imu_data.get('gyroscope_random_walk', 0.00001),
            accel_noise_density=imu_data.get('accelerometer_noise_density', 0.001),
            accel_random_walk=imu_data.get('accelerometer_random_walk', 0.0001),
            rate_hz=imu_data.get('update_rate', 200.0),
        )


def load_kalibr_cameras(yaml_path: Path) -> Dict[str, Tuple[CameraIntrinsics, CameraExtrinsics]]:
    """
    Load camera calibration from Kalibr format.
    
    Args:
        yaml_path: Path to Kalibr camera calibration file
        
    Returns:
        Dictionary mapping camera name to (intrinsics, extrinsics) tuple
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    cameras = {}
    for key in sorted(data.keys()):
        if key.startswith('cam'):
            cam_data = data[key]
            intrinsics = CameraIntrinsics.from_kalibr(cam_data)
            
            # First camera has identity extrinsics
            if 'T_cn_cnm1' in cam_data:
                extrinsics = CameraExtrinsics.from_kalibr(cam_data)
            else:
                extrinsics = CameraExtrinsics.identity()
            
            cameras[key] = (intrinsics, extrinsics)
    
    return cameras


def load_camera_imu_calib(yaml_path: Path) -> np.ndarray:
    """
    Load camera-IMU extrinsic calibration.
    
    Args:
        yaml_path: Path to camera-IMU calibration file
        
    Returns:
        4x4 transformation matrix T_cam_imu
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Kalibr stores T_cam_imu
    T = np.array(data['cam0']['T_cam_imu'])
    return T


def load_imu_params(yaml_path: Path) -> IMUParams:
    """
    Load IMU noise parameters.
    
    Args:
        yaml_path: Path to IMU parameters file
        
    Returns:
        IMUParams object
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Handle different possible formats
    if 'imu0' in data:
        imu_data = data['imu0']
    else:
        imu_data = data
    
    return IMUParams.from_kalibr(imu_data)


def compute_stereo_baseline(
    cameras: Dict[str, Tuple[CameraIntrinsics, CameraExtrinsics]],
    left_cam: str,
    right_cam: str,
) -> float:
    """
    Compute baseline distance between stereo cameras.
    
    Args:
        cameras: Camera calibration dictionary
        left_cam: Name of left camera
        right_cam: Name of right camera
        
    Returns:
        Baseline distance in meters
    """
    # Chain transformations from left to right
    # T_right_left = T_right_cam0 * T_cam0_left
    
    # Get camera indices
    left_idx = int(left_cam.replace('cam', ''))
    right_idx = int(right_cam.replace('cam', ''))
    
    # Compute transformation chain
    T_left_cam0 = np.eye(4)
    for i in range(1, left_idx + 1):
        cam_name = f'cam{i}'
        if cam_name in cameras:
            T_left_cam0 = cameras[cam_name][1].T @ T_left_cam0
    
    T_right_cam0 = np.eye(4)
    for i in range(1, right_idx + 1):
        cam_name = f'cam{i}'
        if cam_name in cameras:
            T_right_cam0 = cameras[cam_name][1].T @ T_right_cam0
    
    # T_right_left = T_right_cam0 * inv(T_left_cam0)
    T_left_cam0_inv = np.eye(4)
    T_left_cam0_inv[:3, :3] = T_left_cam0[:3, :3].T
    T_left_cam0_inv[:3, 3] = -T_left_cam0[:3, :3].T @ T_left_cam0[:3, 3]
    
    T_right_left = T_right_cam0 @ T_left_cam0_inv
    
    # Baseline is the translation magnitude
    baseline = np.linalg.norm(T_right_left[:3, 3])
    return baseline


def convert_to_orbslam3(
    cameras: Dict[str, Tuple[CameraIntrinsics, CameraExtrinsics]],
    left_cam: str = 'cam1',
    right_cam: str = 'cam3',
    output_path: Optional[Path] = None,
) -> str:
    """
    Convert calibration to ORB-SLAM3 stereo format.
    
    Args:
        cameras: Camera calibration dictionary
        left_cam: Name of left camera
        right_cam: Name of right camera
        output_path: Path to save output (optional)
        
    Returns:
        ORB-SLAM3 configuration string
    """
    left_intr, _ = cameras[left_cam]
    right_intr, _ = cameras[right_cam]
    baseline = compute_stereo_baseline(cameras, left_cam, right_cam)
    
    # ORB-SLAM3 YAML format
    config = f"""%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters (Stereo Rectified)
#--------------------------------------------------------------------------------------------

# Camera 1 (Left - {left_cam})
Camera1.type: "PinHole"

Camera1.fx: {left_intr.fx}
Camera1.fy: {left_intr.fy}
Camera1.cx: {left_intr.cx}
Camera1.cy: {left_intr.cy}

Camera1.k1: {left_intr.distortion_coeffs[0] if len(left_intr.distortion_coeffs) > 0 else 0.0}
Camera1.k2: {left_intr.distortion_coeffs[1] if len(left_intr.distortion_coeffs) > 1 else 0.0}
Camera1.p1: {left_intr.distortion_coeffs[2] if len(left_intr.distortion_coeffs) > 2 else 0.0}
Camera1.p2: {left_intr.distortion_coeffs[3] if len(left_intr.distortion_coeffs) > 3 else 0.0}

# Camera 2 (Right - {right_cam})
Camera2.type: "PinHole"

Camera2.fx: {right_intr.fx}
Camera2.fy: {right_intr.fy}
Camera2.cx: {right_intr.cx}
Camera2.cy: {right_intr.cy}

Camera2.k1: {right_intr.distortion_coeffs[0] if len(right_intr.distortion_coeffs) > 0 else 0.0}
Camera2.k2: {right_intr.distortion_coeffs[1] if len(right_intr.distortion_coeffs) > 1 else 0.0}
Camera2.p1: {right_intr.distortion_coeffs[2] if len(right_intr.distortion_coeffs) > 2 else 0.0}
Camera2.p2: {right_intr.distortion_coeffs[3] if len(right_intr.distortion_coeffs) > 3 else 0.0}

# Camera resolution
Camera.width: {left_intr.width}
Camera.height: {left_intr.height}

# Camera frames per second
Camera.fps: 20

# Color order of images (0: BGR, 1: RGB)
Camera.RGB: 1

# Stereo baseline times fx
Stereo.ThDepth: 40.0
Stereo.b: {baseline:.6f}

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1500

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0

#--------------------------------------------------------------------------------------------
# Loop Closing Parameters (DISABLED for benchmarking)
#--------------------------------------------------------------------------------------------
LoopClosing.Enabled: 0
"""
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(config)
    
    return config


def convert_to_vins_fusion(
    cameras: Dict[str, Tuple[CameraIntrinsics, CameraExtrinsics]],
    T_cam_imu: np.ndarray,
    imu_params: IMUParams,
    left_cam: str = 'cam1',
    right_cam: str = 'cam3',
    output_path: Optional[Path] = None,
) -> str:
    """
    Convert calibration to VINS-Fusion format.
    
    Args:
        cameras: Camera calibration dictionary
        T_cam_imu: Camera to IMU transformation (for reference camera)
        imu_params: IMU noise parameters
        left_cam: Name of left camera
        right_cam: Name of right camera
        output_path: Path to save output (optional)
        
    Returns:
        VINS-Fusion configuration string
    """
    left_intr, _ = cameras[left_cam]
    right_intr, _ = cameras[right_cam]
    
    # Compute T_body_cam (inverse of T_cam_imu)
    T_body_cam0 = np.eye(4)
    T_body_cam0[:3, :3] = T_cam_imu[:3, :3].T
    T_body_cam0[:3, 3] = -T_cam_imu[:3, :3].T @ T_cam_imu[:3, 3]
    
    # Compute T_body_cam1 (left camera relative to IMU)
    # This requires chaining: T_body_cam1 = T_body_cam0 * T_cam0_cam1
    # Since we're using cam1 (not cam0), we need to compute the chain
    
    baseline = compute_stereo_baseline(cameras, left_cam, right_cam)
    
    def format_matrix(T: np.ndarray) -> str:
        """Format 4x4 matrix for VINS YAML."""
        rows = []
        for row in T:
            rows.append(', '.join(f'{v:.10f}' for v in row))
        return '[' + ',\n         '.join(rows) + ']'
    
    config = f"""%YAML:1.0

#--------------------------------------------------------------------------------------------
# Common Parameters
#--------------------------------------------------------------------------------------------

imu: 1
num_of_cam: 2

imu_topic: "/vectornav/imu"
image0_topic: "/camera_array/{left_cam}/image_raw"
image1_topic: "/camera_array/{right_cam}/image_raw"

output_path: "/results/vins_fusion"

#--------------------------------------------------------------------------------------------
# Camera Parameters
#--------------------------------------------------------------------------------------------

model_type: PINHOLE
camera_name: camera

image_width: {left_intr.width}
image_height: {left_intr.height}

distortion_parameters:
    k1: {left_intr.distortion_coeffs[0] if len(left_intr.distortion_coeffs) > 0 else 0.0}
    k2: {left_intr.distortion_coeffs[1] if len(left_intr.distortion_coeffs) > 1 else 0.0}
    p1: {left_intr.distortion_coeffs[2] if len(left_intr.distortion_coeffs) > 2 else 0.0}
    p2: {left_intr.distortion_coeffs[3] if len(left_intr.distortion_coeffs) > 3 else 0.0}

projection_parameters:
    fx: {left_intr.fx}
    fy: {left_intr.fy}
    cx: {left_intr.cx}
    cy: {left_intr.cy}

#--------------------------------------------------------------------------------------------
# Extrinsic Parameters (Body to Camera)
#--------------------------------------------------------------------------------------------

body_T_cam0: !!opencv-matrix
    rows: 4
    cols: 4
    dt: d
    data: {format_matrix(T_body_cam0)}

body_T_cam1: !!opencv-matrix
    rows: 4
    cols: 4
    dt: d
    data: {format_matrix(T_body_cam0)}  # Approximation - should chain transforms

#--------------------------------------------------------------------------------------------
# IMU Parameters
#--------------------------------------------------------------------------------------------

acc_n: {imu_params.accel_noise_density}
gyr_n: {imu_params.gyro_noise_density}
acc_w: {imu_params.accel_random_walk}
gyr_w: {imu_params.gyro_random_walk}
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
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(config)
    
    return config


def convert_to_basalt(
    cameras: Dict[str, Tuple[CameraIntrinsics, CameraExtrinsics]],
    T_cam_imu: np.ndarray,
    imu_params: IMUParams,
    left_cam: str = 'cam1',
    right_cam: str = 'cam3',
    output_path: Optional[Path] = None,
) -> str:
    """
    Convert calibration to Basalt JSON format.
    
    Args:
        cameras: Camera calibration dictionary
        T_cam_imu: Camera to IMU transformation
        imu_params: IMU noise parameters
        left_cam: Name of left camera
        right_cam: Name of right camera
        output_path: Path to save output (optional)
        
    Returns:
        Basalt configuration JSON string
    """
    left_intr, _ = cameras[left_cam]
    right_intr, _ = cameras[right_cam]
    
    # Basalt expects T_imu_cam
    T_imu_cam = np.eye(4)
    T_imu_cam[:3, :3] = T_cam_imu[:3, :3].T
    T_imu_cam[:3, 3] = -T_cam_imu[:3, :3].T @ T_cam_imu[:3, 3]
    
    def matrix_to_list(T: np.ndarray) -> List[List[float]]:
        """Convert numpy matrix to nested list."""
        return T.tolist()
    
    config = {
        "value0": {
            "T_imu_cam": [
                {
                    "px": float(T_imu_cam[0, 3]),
                    "py": float(T_imu_cam[1, 3]),
                    "pz": float(T_imu_cam[2, 3]),
                    "qx": 0.0,  # TODO: Convert rotation matrix to quaternion
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                },
                {
                    "px": float(T_imu_cam[0, 3]),  # Second camera
                    "py": float(T_imu_cam[1, 3]),
                    "pz": float(T_imu_cam[2, 3]),
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                }
            ],
            "intrinsics": [
                {
                    "camera_type": "pinhole",
                    "intrinsics": {
                        "fx": left_intr.fx,
                        "fy": left_intr.fy,
                        "cx": left_intr.cx,
                        "cy": left_intr.cy,
                    },
                    "resolution": [left_intr.width, left_intr.height],
                },
                {
                    "camera_type": "pinhole",
                    "intrinsics": {
                        "fx": right_intr.fx,
                        "fy": right_intr.fy,
                        "cx": right_intr.cx,
                        "cy": right_intr.cy,
                    },
                    "resolution": [right_intr.width, right_intr.height],
                }
            ],
            "imu_update_rate": imu_params.rate_hz,
            "gyro_noise_std": imu_params.gyro_noise_density,
            "accel_noise_std": imu_params.accel_noise_density,
            "gyro_bias_std": imu_params.gyro_random_walk,
            "accel_bias_std": imu_params.accel_random_walk,
        }
    }
    
    json_str = json.dumps(config, indent=2)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(json_str)
    
    return json_str


def create_sample_kalibr_config() -> str:
    """
    Create a sample Kalibr configuration file for reference.
    Based on ISEC dataset specifications.
    """
    sample = """# Sample Kalibr camera calibration file
# Based on ISEC dataset (cam0-cam4 front-facing array)

cam0:
  camera_model: pinhole
  distortion_model: radtan
  intrinsics: [891.08, 891.36, 368.84, 275.06]
  distortion_coeffs: [-0.2127, 0.1828, -0.0002, 0.0011]
  resolution: [720, 540]

cam1:
  camera_model: pinhole
  distortion_model: radtan
  intrinsics: [893.63, 893.97, 376.95, 266.57]
  distortion_coeffs: [-0.2127, 0.1828, -0.0002, 0.0011]
  resolution: [720, 540]
  T_cn_cnm1:
  - [1.0, 0.0, 0.0, 0.164]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]

cam2:
  camera_model: pinhole
  distortion_model: radtan
  intrinsics: [890.74, 891.04, 365.53, 279.23]
  distortion_coeffs: [-0.2127, 0.1828, -0.0002, 0.0011]
  resolution: [720, 540]
  T_cn_cnm1:
  - [1.0, 0.0, 0.0, 0.164]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]

cam3:
  camera_model: pinhole
  distortion_model: radtan
  intrinsics: [890.41, 890.60, 370.45, 281.40]
  distortion_coeffs: [-0.2127, 0.1828, -0.0002, 0.0011]
  resolution: [720, 540]
  T_cn_cnm1:
  - [1.0, 0.0, 0.0, 0.164]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]

cam4:
  camera_model: pinhole
  distortion_model: radtan
  intrinsics: [891.32, 891.63, 368.46, 272.86]
  distortion_coeffs: [-0.2127, 0.1828, -0.0002, 0.0011]
  resolution: [720, 540]
  T_cn_cnm1:
  - [1.0, 0.0, 0.0, 0.164]
  - [0.0, 1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0, 0.0]
  - [0.0, 0.0, 0.0, 1.0]
"""
    return sample


def main():
    """Command-line interface for calibration conversion."""
    parser = argparse.ArgumentParser(
        description='Convert calibration files between SLAM algorithm formats'
    )
    subparsers = parser.add_subparsers(dest='command', help='Conversion command')
    
    # Kalibr to ORB-SLAM3
    orb_parser = subparsers.add_parser(
        'kalibr-to-orbslam3',
        help='Convert Kalibr to ORB-SLAM3 format'
    )
    orb_parser.add_argument('--input', '-i', type=Path, required=True,
                           help='Input Kalibr camera calibration file')
    orb_parser.add_argument('--output', '-o', type=Path, required=True,
                           help='Output ORB-SLAM3 config file')
    orb_parser.add_argument('--left-cam', default='cam1',
                           help='Left camera name (default: cam1)')
    orb_parser.add_argument('--right-cam', default='cam3',
                           help='Right camera name (default: cam3)')
    
    # Kalibr to VINS-Fusion
    vins_parser = subparsers.add_parser(
        'kalibr-to-vins',
        help='Convert Kalibr to VINS-Fusion format'
    )
    vins_parser.add_argument('--cam-calib', type=Path, required=True,
                            help='Input Kalibr camera calibration file')
    vins_parser.add_argument('--imu-calib', type=Path, required=True,
                            help='Input camera-IMU calibration file')
    vins_parser.add_argument('--imu-params', type=Path,
                            help='Input IMU parameters file')
    vins_parser.add_argument('--output', '-o', type=Path, required=True,
                            help='Output VINS-Fusion config file')
    vins_parser.add_argument('--left-cam', default='cam1',
                            help='Left camera name (default: cam1)')
    vins_parser.add_argument('--right-cam', default='cam3',
                            help='Right camera name (default: cam3)')
    
    # Kalibr to Basalt
    basalt_parser = subparsers.add_parser(
        'kalibr-to-basalt',
        help='Convert Kalibr to Basalt JSON format'
    )
    basalt_parser.add_argument('--cam-calib', type=Path, required=True,
                              help='Input Kalibr camera calibration file')
    basalt_parser.add_argument('--imu-calib', type=Path, required=True,
                              help='Input camera-IMU calibration file')
    basalt_parser.add_argument('--imu-params', type=Path,
                              help='Input IMU parameters file')
    basalt_parser.add_argument('--output', '-o', type=Path, required=True,
                              help='Output Basalt config file')
    basalt_parser.add_argument('--left-cam', default='cam1',
                              help='Left camera name (default: cam1)')
    basalt_parser.add_argument('--right-cam', default='cam3',
                              help='Right camera name (default: cam3)')
    
    # Show sample
    sample_parser = subparsers.add_parser(
        'sample',
        help='Show sample Kalibr calibration file'
    )
    sample_parser.add_argument('--output', '-o', type=Path,
                              help='Output file (optional)')
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show calibration file information'
    )
    info_parser.add_argument('calib_file', type=Path,
                            help='Calibration file to inspect')
    
    args = parser.parse_args()
    
    if args.command == 'kalibr-to-orbslam3':
        print(f"Converting {args.input} to ORB-SLAM3 format...")
        cameras = load_kalibr_cameras(args.input)
        baseline = compute_stereo_baseline(cameras, args.left_cam, args.right_cam)
        print(f"  Stereo baseline ({args.left_cam}-{args.right_cam}): {baseline:.4f}m")
        convert_to_orbslam3(cameras, args.left_cam, args.right_cam, args.output)
        print(f"  Saved to: {args.output}")
    
    elif args.command == 'kalibr-to-vins':
        print(f"Converting to VINS-Fusion format...")
        cameras = load_kalibr_cameras(args.cam_calib)
        T_cam_imu = load_camera_imu_calib(args.imu_calib)
        
        # Load IMU params or use defaults
        if args.imu_params:
            imu_params = load_imu_params(args.imu_params)
        else:
            imu_params = IMUParams(
                gyro_noise_density=0.0001,
                gyro_random_walk=0.00001,
                accel_noise_density=0.001,
                accel_random_walk=0.0001,
            )
        
        convert_to_vins_fusion(
            cameras, T_cam_imu, imu_params,
            args.left_cam, args.right_cam, args.output
        )
        print(f"  Saved to: {args.output}")
    
    elif args.command == 'kalibr-to-basalt':
        print(f"Converting to Basalt format...")
        cameras = load_kalibr_cameras(args.cam_calib)
        T_cam_imu = load_camera_imu_calib(args.imu_calib)
        
        if args.imu_params:
            imu_params = load_imu_params(args.imu_params)
        else:
            imu_params = IMUParams(
                gyro_noise_density=0.0001,
                gyro_random_walk=0.00001,
                accel_noise_density=0.001,
                accel_random_walk=0.0001,
            )
        
        convert_to_basalt(
            cameras, T_cam_imu, imu_params,
            args.left_cam, args.right_cam, args.output
        )
        print(f"  Saved to: {args.output}")
    
    elif args.command == 'sample':
        sample = create_sample_kalibr_config()
        if args.output:
            with open(args.output, 'w') as f:
                f.write(sample)
            print(f"Sample saved to: {args.output}")
        else:
            print(sample)
    
    elif args.command == 'info':
        print(f"Calibration file: {args.calib_file}")
        try:
            cameras = load_kalibr_cameras(args.calib_file)
            print(f"\nFound {len(cameras)} cameras:")
            for name, (intr, extr) in cameras.items():
                print(f"\n  {name}:")
                print(f"    Resolution: {intr.width}x{intr.height}")
                print(f"    Intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, "
                      f"cx={intr.cx:.2f}, cy={intr.cy:.2f}")
                print(f"    Distortion: {intr.distortion_coeffs}")
            
            # Compute stereo baselines
            if len(cameras) >= 2:
                print("\nStereo baselines:")
                cam_names = sorted(cameras.keys())
                for i, c1 in enumerate(cam_names):
                    for c2 in cam_names[i+1:]:
                        try:
                            baseline = compute_stereo_baseline(cameras, c1, c2)
                            print(f"  {c1}-{c2}: {baseline:.4f}m")
                        except:
                            pass
        except Exception as e:
            print(f"Error reading calibration: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
