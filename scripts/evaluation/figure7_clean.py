#!/usr/bin/env python3
"""
Figure 7: Clean 5th Floor Trajectory Comparison
Proper timestamp-based alignment, no region circles
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Update these paths to match your setup
BASE = Path.home() / 'Dev' / 'ros1' / 'slam-benchmark'
TRAJ = BASE / 'results' / 'trajectories'
FIGURES = BASE / 'results' / 'figures'

def load_traj_with_time(filepath):
    """Load TUM trajectory file, return timestamps and Nx3 positions"""
    times = []
    positions = []
    with open(filepath) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                times.append(float(parts[0]))
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(times), np.array(positions)

def compute_length(traj):
    return np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))

def associate_by_timestamp(times_src, pos_src, times_tgt, pos_tgt, max_diff=0.1):
    """Associate trajectories by timestamp matching"""
    matched_src = []
    matched_tgt = []
    
    for i, t_src in enumerate(times_src):
        diffs = np.abs(times_tgt - t_src)
        j = np.argmin(diffs)
        if diffs[j] < max_diff:
            matched_src.append(pos_src[i])
            matched_tgt.append(pos_tgt[j])
    
    return np.array(matched_src), np.array(matched_tgt)

def align_sim3_umeyama(source, target):
    """Sim(3) alignment using Umeyama algorithm"""
    n = source.shape[0]
    
    mu_src = np.mean(source, axis=0)
    mu_tgt = np.mean(target, axis=0)
    
    src_c = source - mu_src
    tgt_c = target - mu_tgt
    
    var_src = np.sum(src_c**2) / n
    cov = (tgt_c.T @ src_c) / n
    
    U, D, Vt = np.linalg.svd(cov)
    
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / var_src
    t = mu_tgt - scale * (R @ mu_src)
    
    return scale, R, t

def align_se3_umeyama(source, target):
    """SE(3) alignment using Umeyama algorithm (scale=1)"""
    n = source.shape[0]
    
    mu_src = np.mean(source, axis=0)
    mu_tgt = np.mean(target, axis=0)
    
    src_c = source - mu_src
    tgt_c = target - mu_tgt
    
    cov = (tgt_c.T @ src_c) / n
    
    U, D, Vt = np.linalg.svd(cov)
    
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    
    R = U @ S @ Vt
    t = mu_tgt - R @ mu_src
    
    return R, t

def apply_sim3(traj, scale, R, t):
    return scale * (traj @ R.T) + t

def apply_se3(traj, R, t):
    return traj @ R.T + t

def main():
    print("Loading trajectories with timestamps...")
    
    t_lego, lego = load_traj_with_time(TRAJ / 'lego_loam' / '5th_floor.txt')
    t_orb, orb = load_traj_with_time(TRAJ / 'orb_slam3' / '5th_floor.txt')
    t_droid, droid = load_traj_with_time(TRAJ / 'droid_slam' / '5th_floor_stereo.txt')
    
    print(f"  LeGO-LOAM: {len(lego)} poses, {compute_length(lego):.1f}m")
    print(f"    Time: {t_lego[0]:.1f} - {t_lego[-1]:.1f}")
    print(f"  ORB-SLAM3: {len(orb)} poses, {compute_length(orb):.1f}m")
    print(f"    Time: {t_orb[0]:.1f} - {t_orb[-1]:.1f}")
    print(f"  DROID-SLAM: {len(droid)} poses, {compute_length(droid):.1f}m (raw)")
    print(f"    Time: {t_droid[0]:.1f} - {t_droid[-1]:.1f}")
    
    print("\nAssociating by timestamp...")
    
    orb_matched, lego_matched_orb = associate_by_timestamp(t_orb, orb, t_lego, lego)
    droid_matched, lego_matched_droid = associate_by_timestamp(t_droid, droid, t_lego, lego)
    
    print(f"  ORB-SLAM3: {len(orb_matched)} matched poses")
    print(f"  DROID-SLAM: {len(droid_matched)} matched poses")
    
    print("\nAligning...")
    
    # Align ORB-SLAM3 (SE3)
    R_orb, t_orb_align = align_se3_umeyama(orb_matched, lego_matched_orb)
    orb_aligned = apply_se3(orb, R_orb, t_orb_align)
    
    # Align DROID-SLAM (Sim3)
    scale_droid, R_droid, t_droid_align = align_sim3_umeyama(droid_matched, lego_matched_droid)
    droid_aligned = apply_sim3(droid, scale_droid, R_droid, t_droid_align)
    
    print(f"\nResults:")
    print(f"  LeGO-LOAM: {compute_length(lego):.1f}m")
    print(f"  ORB-SLAM3: {compute_length(orb_aligned):.1f}m")
    print(f"  DROID-SLAM: {compute_length(droid_aligned):.1f}m (scale={scale_droid:.3f})")
    
    # ATE
    orb_aligned_matched = apply_se3(orb_matched, R_orb, t_orb_align)
    droid_aligned_matched = apply_sim3(droid_matched, scale_droid, R_droid, t_droid_align)
    
    orb_ate = np.sqrt(np.mean(np.sum((orb_aligned_matched - lego_matched_orb)**2, axis=1)))
    droid_ate = np.sqrt(np.mean(np.sum((droid_aligned_matched - lego_matched_droid)**2, axis=1)))
    
    print(f"\nATE RMSE: ORB-SLAM3={orb_ate:.2f}m, DROID-SLAM={droid_ate:.2f}m")
    
    # ========== Clean Figure 7 ==========
    fig, ax = plt.subplots(figsize=(8, 12))
    
    # Plot X vs Z for top-down view
    ax.plot(lego[:, 0], lego[:, 2], 
            color='black', linewidth=1.5, label='LeGO-LOAM', zorder=3)
    
    ax.plot(orb_aligned[:, 0], orb_aligned[:, 2], 
            color='red', linewidth=1.2, label='ORB-SLAM3', alpha=0.8, zorder=2)
    
    ax.plot(droid_aligned[:, 0], droid_aligned[:, 2], 
            color='green', linewidth=1.2, label='DROID-SLAM', alpha=0.8, zorder=2)
    
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 15)
    ax.set_ylim(-25, 40)
    
    plt.tight_layout()
    plt.savefig(FIGURES / 'figure_7_clean.png', dpi=200, bbox_inches='tight')
    plt.savefig(FIGURES / 'figure_7_clean.pdf', bbox_inches='tight')
    print(f"\nSaved to {FIGURES}")
    plt.close()

if __name__ == '__main__':
    main()