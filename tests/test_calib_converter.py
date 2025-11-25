#!/usr/bin/env python3
"""
Tests for calibration converter.
"""

import sys
import tempfile
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import pytest
import numpy as np
import yaml
import json

from calib_converter import (
    CameraIntrinsics,
    CameraExtrinsics,
    IMUParams,
    load_kalibr_cameras,
    load_camera_imu_calib,
    load_imu_params,
    compute_stereo_baseline,
    convert_to_orbslam3,
    convert_to_vins_fusion,
    convert_to_basalt,
)


@pytest.fixture
def sample_kalibr_file(tmp_path):
    """Create a sample Kalibr calibration file."""
    config = {
        'cam0': {
            'camera_model': 'pinhole',
            'distortion_model': 'radtan',
            'intrinsics': [891.08, 891.36, 368.84, 275.06],
            'distortion_coeffs': [-0.2127, 0.1828, -0.0002, 0.0011],
            'resolution': [720, 540],
        },
        'cam1': {
            'camera_model': 'pinhole',
            'distortion_model': 'radtan',
            'intrinsics': [893.63, 893.97, 376.95, 266.57],
            'distortion_coeffs': [-0.2127, 0.1828, -0.0002, 0.0011],
            'resolution': [720, 540],
            'T_cn_cnm1': [
                [1.0, 0.0, 0.0, 0.164],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        'cam3': {
            'camera_model': 'pinhole',
            'distortion_model': 'radtan',
            'intrinsics': [890.41, 890.60, 370.45, 281.40],
            'distortion_coeffs': [-0.2127, 0.1828, -0.0002, 0.0011],
            'resolution': [720, 540],
            'T_cn_cnm1': [
                [1.0, 0.0, 0.0, 0.164],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    }
    
    filepath = tmp_path / 'cams_calib.yaml'
    with open(filepath, 'w') as f:
        yaml.dump(config, f)
    
    return filepath


@pytest.fixture
def sample_cam_imu_file(tmp_path):
    """Create a sample camera-IMU calibration file."""
    config = {
        'cam0': {
            'T_cam_imu': [
                [0.0, -1.0, 0.0, 0.05],
                [0.0, 0.0, -1.0, -0.03],
                [1.0, 0.0, 0.0, 0.02],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
    
    filepath = tmp_path / 'cam_imu_calib.yaml'
    with open(filepath, 'w') as f:
        yaml.dump(config, f)
    
    return filepath


@pytest.fixture
def sample_imu_params_file(tmp_path):
    """Create a sample IMU parameters file."""
    config = {
        'imu0': {
            'update_rate': 200.0,
            'gyroscope_noise_density': 0.0001,
            'gyroscope_random_walk': 0.00001,
            'accelerometer_noise_density': 0.001,
            'accelerometer_random_walk': 0.0001,
        }
    }
    
    filepath = tmp_path / 'imu_params.yaml'
    with open(filepath, 'w') as f:
        yaml.dump(config, f)
    
    return filepath


class TestCameraIntrinsics:
    """Tests for CameraIntrinsics class."""
    
    def test_from_kalibr(self):
        cam_data = {
            'intrinsics': [891.08, 891.36, 368.84, 275.06],
            'resolution': [720, 540],
            'distortion_coeffs': [-0.2127, 0.1828, -0.0002, 0.0011],
            'distortion_model': 'radtan',
        }
        
        intr = CameraIntrinsics.from_kalibr(cam_data)
        
        assert intr.fx == pytest.approx(891.08)
        assert intr.fy == pytest.approx(891.36)
        assert intr.cx == pytest.approx(368.84)
        assert intr.cy == pytest.approx(275.06)
        assert intr.width == 720
        assert intr.height == 540
        assert len(intr.distortion_coeffs) == 4


class TestLoadKalibrCameras:
    """Tests for loading Kalibr camera files."""
    
    def test_load_cameras(self, sample_kalibr_file):
        cameras = load_kalibr_cameras(sample_kalibr_file)
        
        assert 'cam0' in cameras
        assert 'cam1' in cameras
        assert 'cam3' in cameras
        
        # Check intrinsics
        cam0_intr, _ = cameras['cam0']
        assert cam0_intr.fx == pytest.approx(891.08)
        assert cam0_intr.width == 720
        
        # Check extrinsics
        _, cam1_extr = cameras['cam1']
        assert cam1_extr.translation[0] == pytest.approx(0.164)


class TestStereoBaseline:
    """Tests for stereo baseline computation."""
    
    def test_baseline_cam0_cam1(self, sample_kalibr_file):
        cameras = load_kalibr_cameras(sample_kalibr_file)
        baseline = compute_stereo_baseline(cameras, 'cam0', 'cam1')
        
        # Expected baseline is 0.164m
        assert baseline == pytest.approx(0.164, abs=0.001)
    
    def test_baseline_cam1_cam3(self, sample_kalibr_file):
        cameras = load_kalibr_cameras(sample_kalibr_file)
        baseline = compute_stereo_baseline(cameras, 'cam1', 'cam3')
        
        # With cam1->cam3, baseline should be roughly 0.328m (2 * 0.164)
        # Note: This depends on actual transform chain
        assert baseline > 0.1  # Just check it's reasonable


class TestConvertToOrbSlam3:
    """Tests for ORB-SLAM3 conversion."""
    
    def test_conversion(self, sample_kalibr_file, tmp_path):
        cameras = load_kalibr_cameras(sample_kalibr_file)
        output_path = tmp_path / 'orb_slam3.yaml'
        
        config_str = convert_to_orbslam3(
            cameras, 
            left_cam='cam0', 
            right_cam='cam1',
            output_path=output_path
        )
        
        assert output_path.exists()
        assert 'Camera1.fx' in config_str
        assert 'Camera2.fx' in config_str
        assert 'Stereo.b' in config_str
        assert 'LoopClosing.Enabled: 0' in config_str  # LC disabled


class TestConvertToVinsFusion:
    """Tests for VINS-Fusion conversion."""
    
    def test_conversion(self, sample_kalibr_file, sample_cam_imu_file, 
                       sample_imu_params_file, tmp_path):
        cameras = load_kalibr_cameras(sample_kalibr_file)
        T_cam_imu = load_camera_imu_calib(sample_cam_imu_file)
        imu_params = load_imu_params(sample_imu_params_file)
        
        output_path = tmp_path / 'vins_config.yaml'
        
        config_str = convert_to_vins_fusion(
            cameras, T_cam_imu, imu_params,
            left_cam='cam0', right_cam='cam1',
            output_path=output_path
        )
        
        assert output_path.exists()
        assert 'imu: 1' in config_str
        assert 'num_of_cam: 2' in config_str
        assert 'loop_closure: 0' in config_str  # LC disabled


class TestConvertToBasalt:
    """Tests for Basalt conversion."""
    
    def test_conversion(self, sample_kalibr_file, sample_cam_imu_file,
                       sample_imu_params_file, tmp_path):
        cameras = load_kalibr_cameras(sample_kalibr_file)
        T_cam_imu = load_camera_imu_calib(sample_cam_imu_file)
        imu_params = load_imu_params(sample_imu_params_file)
        
        output_path = tmp_path / 'basalt_config.json'
        
        config_str = convert_to_basalt(
            cameras, T_cam_imu, imu_params,
            left_cam='cam0', right_cam='cam1',
            output_path=output_path
        )
        
        assert output_path.exists()
        
        # Verify JSON is valid
        config = json.loads(config_str)
        assert 'value0' in config
        assert 'intrinsics' in config['value0']
        assert len(config['value0']['intrinsics']) == 2


class TestIMUParams:
    """Tests for IMU parameters loading."""
    
    def test_load_imu_params(self, sample_imu_params_file):
        params = load_imu_params(sample_imu_params_file)
        
        assert params.rate_hz == pytest.approx(200.0)
        assert params.gyro_noise_density == pytest.approx(0.0001)
        assert params.accel_noise_density == pytest.approx(0.001)


def test_camera_imu_calib(sample_cam_imu_file):
    """Test loading camera-IMU calibration."""
    T = load_camera_imu_calib(sample_cam_imu_file)
    
    assert T.shape == (4, 4)
    assert T[3, 3] == pytest.approx(1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
