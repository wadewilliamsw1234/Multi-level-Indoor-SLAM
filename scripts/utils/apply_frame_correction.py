#!/usr/bin/env python3
"""
Apply NED→FLU frame correction to VINS-Fusion extrinsics.

The VectorNav VN-100 IMU outputs in a Z-DOWN (NED-like) frame,
but VINS-Fusion expects FLU (Forward-Left-Up) with Z pointing up.

Transformation: R_flu_ned = diag(1, -1, -1)

For body_T_cam: T_corrected = R_flu_ned @ T_original
"""

import numpy as np

# Frame transformation: NED → FLU
# This flips Y and Z axes
R_flu_ned = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
], dtype=np.float64)


def apply_frame_correction(T_original):
    """
    Apply NED→FLU frame correction to a 4x4 transform.
    
    Args:
        T_original: 4x4 homogeneous transform in NED body frame
        
    Returns:
        T_corrected: 4x4 homogeneous transform in FLU body frame
    """
    T = np.array(T_original).reshape(4, 4)
    
    R_old = T[:3, :3]
    t_old = T[:3, 3]
    
    # Transform rotation and translation
    R_new = R_flu_ned @ R_old
    t_new = R_flu_ned @ t_old
    
    # Build corrected transform
    T_corrected = np.eye(4)
    T_corrected[:3, :3] = R_new
    T_corrected[:3, 3] = t_new
    
    return T_corrected


def format_opencv_matrix(T, name, indent=3):
    """Format 4x4 matrix for YAML opencv-matrix format."""
    data = T.flatten()
    data_str = ', '.join([f'{x:.10f}' for x in data])
    
    indent_str = ' ' * indent
    return f"""{name}: !!opencv-matrix
{indent_str}rows: 4
{indent_str}cols: 4
{indent_str}dt: d
{indent_str}data: [{data_str}]"""


# Original transforms from isec_stereo_imu_config.yaml
# These were computed in VectorNav (NED-like) frame

body_T_cam0_original = np.array([
    -0.0327966961, -0.0071500981, 0.9994364676, 0.0553758892,
    0.9994453607, 0.0055430833, 0.0328366438, -0.0828510910,
    -0.0057747448, 0.9999590743, 0.0069643376, -0.0044696646,
    0.0, 0.0, 0.0, 1.0
]).reshape(4, 4)

body_T_cam1_original = np.array([
    -0.0351970137, -0.0028466761, 0.9993763389, 0.0599680647,
    0.9993760849, 0.0028360152, 0.0352050830, -0.4111960895,
    -0.0029344639, 0.9999919267, 0.0027450807, -0.0034074877,
    0.0, 0.0, 0.0, 1.0
]).reshape(4, 4)


if __name__ == '__main__':
    print("=" * 70)
    print("NED → FLU Frame Correction for VINS-Fusion Extrinsics")
    print("=" * 70)
    
    print("\nOriginal body_T_cam0 (NED frame):")
    print(body_T_cam0_original)
    
    print("\nOriginal body_T_cam1 (NED frame):")
    print(body_T_cam1_original)
    
    # Apply corrections
    body_T_cam0_corrected = apply_frame_correction(body_T_cam0_original)
    body_T_cam1_corrected = apply_frame_correction(body_T_cam1_original)
    
    print("\n" + "=" * 70)
    print("Corrected Transforms (FLU frame)")
    print("=" * 70)
    
    print("\nCorrected body_T_cam0 (FLU frame):")
    print(body_T_cam0_corrected)
    
    print("\nCorrected body_T_cam1 (FLU frame):")
    print(body_T_cam1_corrected)
    
    # Verify baseline is preserved
    t0 = body_T_cam0_corrected[:3, 3]
    t1 = body_T_cam1_corrected[:3, 3]
    baseline = np.linalg.norm(t1 - t0)
    print(f"\nStereo baseline: {baseline:.6f} m (should be ~0.328m)")
    
    # Generate YAML output
    print("\n" + "=" * 70)
    print("YAML Configuration (copy to config file)")
    print("=" * 70 + "\n")
    
    print("# Extrinsic parameter between IMU and Camera (FLU frame corrected)")
    print("estimate_extrinsic: 0\n")
    print("# T_imu_cam0 (body_T_cam0) - cam1 in ISEC naming - FLU CORRECTED")
    print(format_opencv_matrix(body_T_cam0_corrected, "body_T_cam0"))
    print()
    print("# T_imu_cam1 (body_T_cam1) - cam3 in ISEC naming - FLU CORRECTED")
    print(format_opencv_matrix(body_T_cam1_corrected, "body_T_cam1"))








