#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def plot_trajectory_2d(traj_file, output_file=None):
    """Plot top-down (XY) view of trajectory."""
    data = np.loadtxt(traj_file)
    positions = data[:, 1:4]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1, alpha=0.6, label='Trajectory')
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
    
    # Mark every 100th pose
    every_n = max(1, len(positions) // 20)
    ax.plot(positions[::every_n, 0], positions[::every_n, 1], 'k.', markersize=3, alpha=0.3)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'ORB-SLAM3 Trajectory: {Path(traj_file).stem}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot_trajectory_2d.py <trajectory.txt> [output.png]")
        sys.exit(1)
    
    traj_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_trajectory_2d(traj_file, output_file)
