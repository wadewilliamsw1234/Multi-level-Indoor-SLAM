#!/usr/bin/env python3
"""
Compute correct camera-IMU transforms for ISEC dataset cam1-cam3 stereo pair.

Calibration chain:
  cam0 --T_cam1_cam0--> cam1 --T_cam2_cam1--> cam2 --T_cam3_cam2--> cam3

From cam2_imu_calib.yaml: T_cam2_imu (cam2 to IMU transform)

To get T_imu_cam1 and T_imu_cam3:
  T_imu_cam2 = inv(T_cam2_imu)
  T_imu_cam1 = T_imu_cam2 @ T_cam2_cam1
  T_imu_cam3 = T_imu_cam2 @ inv(T_cam3_cam2)
"""

import numpy as np
from scipy.spatial.transform import Rotation
import json


def invert_transform(T):
    """Invert a 4x4 homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_to_quat_pos(T):
    """Convert 4x4 transform to position and quaternion (x,y,z,w)."""
    pos = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3])
    quat = rot.as_quat()  # [x, y, z, w]
    return pos, quat


# =============================================================================
# ISEC Calibration Data (from provided files)
# =============================================================================

# T_cam1_cam0: Transform from cam0 to cam1 (from cams_calib.yaml cam1.T_cn_cnm1)
T_cam1_cam0 = np.array([
    [0.9999114767857825, 0.0022435658807767726, 0.013115067831304778, -0.16367451457695553],
    [-0.0022957060646371153, 0.9999895165181838, 0.003961891896486475, 0.000622143811515013],
    [-0.01310604157424733, -0.0039916495178398586, 0.999906144799794, -0.0006036538778557466],
    [0.0, 0.0, 0.0, 1.0]
])

# T_cam2_cam1: Transform from cam1 to cam2 (from cams_calib.yaml cam2.T_cn_cnm1)
T_cam2_cam1 = np.array([
    [0.9998363980211518, -0.0003473714958483863, -0.018084704175858397, -0.1650359869677547],
    [0.00027089450969152524, 0.9999910121661716, -0.004231099506707433, -0.00040226567143760605],
    [0.0180860113969072, 0.0042255082433848064, 0.9998275057587841, -0.0033312692471531807],
    [0.0, 0.0, 0.0, 1.0]
])

# T_cam3_cam2: Transform from cam2 to cam3 (from cams_calib.yaml cam3.T_cn_cnm1)
T_cam3_cam2 = np.array([
    [0.9997871852271749, -0.0024993532867433684, 0.02047773153111626, -0.1633086405209946],
    [0.0024994854749495568, 0.9999968760981375, 1.9139399906633837e-05, -0.0003438129656766549],
    [-0.020477715396814755, 3.204846576235111e-05, 0.999790309087367, 0.00010157880122056058],
    [0.0, 0.0, 0.0, 1.0]
])

# T_cam2_imu: Transform from IMU to cam2 (from cam2_imu_calib.yaml)
T_cam2_imu = np.array([
    [-0.014717448030483915, 0.9998772349793116, -0.005376959512298662, 0.24870122345739343],
    [-0.002915514158615351, 0.005334606124805935, 0.9999815207066001, 0.005432018735669777],
    [0.9998874419156695, 0.014732852664032015, 0.0028366444470543928, -0.05379197879298332],
    [0.0, 0.0, 0.0, 1.0]
])

# Camera intrinsics
cam1_intrinsics = [893.6263545058326, 893.9655105687939, 376.95348001716707, 266.57152598273194]
cam1_distortion = [-0.21272110177039052, 0.18283401892861978, -0.00018083866109219808, 0.0011164116025029272]

cam3_intrinsics = [890.413113214874, 890.5963588964028, 370.45235809833287, 281.40396328476237]
cam3_distortion = [-0.20384682945091906, 0.13909705153511223, 0.0002034498187085838, 7.513834330685657e-05]

# IMU parameters (from imu_params.yaml via cam2_imu_calib.yaml)
imu_params = {
    'acc_noise': 0.0014126598501078217,
    'gyr_noise': 7.77970988215584e-05,
    'acc_walk': 1.9005701759499173e-05,
    'gyr_walk': 3.3171207235534e-07,
    'rate': 200.0
}

# =============================================================================
# Compute Transforms
# =============================================================================

print("=" * 60)
print("ISEC Stereo-IMU Calibration Computation")
print("=" * 60)

# Step 1: Compute T_imu_cam2 = inv(T_cam2_imu)
T_imu_cam2 = invert_transform(T_cam2_imu)
print("\nT_imu_cam2 (IMU to cam2):")
print(T_imu_cam2)

# Step 2: Compute T_imu_cam1 = T_imu_cam2 @ T_cam2_cam1
# This chains: IMU -> cam2 -> cam1
# Wait, T_cam2_cam1 goes cam1 -> cam2, so we need inv(T_cam2_cam1) to go cam2 -> cam1
T_cam1_cam2 = invert_transform(T_cam2_cam1)
T_imu_cam1 = T_imu_cam2 @ T_cam1_cam2

print("\nT_imu_cam1 (IMU to cam1 / left camera):")
print(T_imu_cam1)

# Step 3: Compute T_imu_cam3 = T_imu_cam2 @ T_cam3_cam2
# T_cam3_cam2 goes cam2 -> cam3, which is what we need
T_imu_cam3 = T_imu_cam2 @ T_cam3_cam2

print("\nT_imu_cam3 (IMU to cam3 / right camera):")
print(T_imu_cam3)

# =============================================================================
# Verify: Compute stereo baseline
# =============================================================================

pos_cam1 = T_imu_cam1[:3, 3]
pos_cam3 = T_imu_cam3[:3, 3]
baseline = np.linalg.norm(pos_cam3 - pos_cam1)

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)
print(f"cam1 position in IMU frame: [{pos_cam1[0]:.6f}, {pos_cam1[1]:.6f}, {pos_cam1[2]:.6f}]")
print(f"cam3 position in IMU frame: [{pos_cam3[0]:.6f}, {pos_cam3[1]:.6f}, {pos_cam3[2]:.6f}]")
print(f"Stereo baseline: {baseline:.6f} m")
print(f"Expected baseline: ~0.328 m")
print(f"Baseline error: {abs(baseline - 0.328379) * 1000:.2f} mm")

# =============================================================================
# Generate VINS-Fusion Config
# =============================================================================

print("\n" + "=" * 60)
print("VINS-Fusion Configuration")
print("=" * 60)

vins_config = f"""%YAML:1.0

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

image_width: 720
image_height: 540

distortion_parameters:
    k1: {cam1_distortion[0]}
    k2: {cam1_distortion[1]}
    p1: {cam1_distortion[2]}
    p2: {cam1_distortion[3]}

projection_parameters:
    fx: {cam1_intrinsics[0]}
    fy: {cam1_intrinsics[1]}
    cx: {cam1_intrinsics[2]}
    cy: {cam1_intrinsics[3]}

#--------------------------------------------------------------------------------------------
# Extrinsic Parameters (Body/IMU to Camera)
#--------------------------------------------------------------------------------------------

# T_imu_cam1 (body_T_cam0) - Left camera
body_T_cam0: !!opencv-matrix
    rows: 4
    cols: 4
    dt: d
    data: [{', '.join([f'{x:.10f}' for x in T_imu_cam1.flatten()])}]

# T_imu_cam3 (body_T_cam1) - Right camera  
body_T_cam1: !!opencv-matrix
    rows: 4
    cols: 4
    dt: d
    data: [{', '.join([f'{x:.10f}' for x in T_imu_cam3.flatten()])}]

#--------------------------------------------------------------------------------------------
# IMU Parameters
#--------------------------------------------------------------------------------------------

acc_n: {imu_params['acc_noise']}
gyr_n: {imu_params['gyr_noise']}
acc_w: {imu_params['acc_walk']}
gyr_w: {imu_params['gyr_walk']}
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

print(vins_config)

# Save to file
with open('/tmp/isec_stereo_imu_corrected.yaml', 'w') as f:
    f.write(vins_config)
print(f"\nSaved to: /tmp/isec_stereo_imu_corrected.yaml")

# =============================================================================
# Generate Basalt Config
# =============================================================================

print("\n" + "=" * 60)
print("Basalt Configuration")
print("=" * 60)

pos1, quat1 = transform_to_quat_pos(T_imu_cam1)
pos3, quat3 = transform_to_quat_pos(T_imu_cam3)

basalt_config = {
    "value0": {
        "T_imu_cam": [
            {
                "px": float(pos1[0]),
                "py": float(pos1[1]),
                "pz": float(pos1[2]),
                "qx": float(quat1[0]),
                "qy": float(quat1[1]),
                "qz": float(quat1[2]),
                "qw": float(quat1[3])
            },
            {
                "px": float(pos3[0]),
                "py": float(pos3[1]),
                "pz": float(pos3[2]),
                "qx": float(quat3[0]),
                "qy": float(quat3[1]),
                "qz": float(quat3[2]),
                "qw": float(quat3[3])
            }
        ],
        "intrinsics": [
            {
                "camera_type": "pinhole",
                "intrinsics": {
                    "fx": cam1_intrinsics[0],
                    "fy": cam1_intrinsics[1],
                    "cx": cam1_intrinsics[2],
                    "cy": cam1_intrinsics[3]
                },
                "resolution": [720, 540]
            },
            {
                "camera_type": "pinhole",
                "intrinsics": {
                    "fx": cam3_intrinsics[0],
                    "fy": cam3_intrinsics[1],
                    "cx": cam3_intrinsics[2],
                    "cy": cam3_intrinsics[3]
                },
                "resolution": [720, 540]
            }
        ],
        "imu_update_rate": imu_params['rate'],
        "gyro_noise_std": imu_params['gyr_noise'],
        "accel_noise_std": imu_params['acc_noise'],
        "gyro_bias_std": imu_params['gyr_walk'],
        "accel_bias_std": imu_params['acc_walk']
    }
}

print(json.dumps(basalt_config, indent=2))

# Save to file
with open('/tmp/isec_calib.json', 'w') as f:
    json.dump(basalt_config, f, indent=2)
print(f"\nSaved to: /tmp/isec_calib.json")

print("\n" + "=" * 60)
print("DONE - Copy these files to your config directory")
print("=" * 60)
