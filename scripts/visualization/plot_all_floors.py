#!/usr/bin/env python3
"""
Plot all LeGO-LOAM floor trajectories in a grid layout.
Matches paper Figure 7 orientation.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("/home/wadewilliams/Dev/ros1/slam-benchmark/results")
TRAJ_DIR = RESULTS_DIR / "trajectories" / "lego_loam"
FIG_DIR = RESULTS_DIR / "figures"

# Paper values for reference
PAPER_VALUES = {
    "5th_floor": {"length_m": 187, "ate_m": 0.395},
    "1st_floor": {"length_m": 65, "ate_m": 0.256},
    "4th_floor": {"length_m": 66, "ate_m": 0.789},
    "2nd_floor": {"length_m": 128, "ate_m": 0.286},
}

FLOOR_COLORS = {
    "5th_floor": "purple",
    "1st_floor": "blue", 
    "4th_floor": "green",
    "2nd_floor": "orange",
}


def load_trajectory(traj_path):
    """Load TUM format trajectory."""
    poses = []
    with open(traj_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                poses.append([float(p) for p in parts])
    return np.array(poses) if poses else None


def compute_stats(poses):
    """Compute trajectory statistics."""
    if poses is None or len(poses) < 2:
        return None
    
    # Length
    diffs = np.diff(poses[:, 1:4], axis=0)
    length = np.sum(np.linalg.norm(diffs, axis=1))
    
    # Drift
    drift = np.linalg.norm(poses[-1, 1:4] - poses[0, 1:4])
    
    # Duration
    duration = poses[-1, 0] - poses[0, 0]
    
    return {
        "length": length,
        "drift": drift,
        "duration": duration,
        "poses": len(poses),
    }


def plot_single_floor(ax, poses, floor_name, color):
    """Plot single floor trajectory."""
    if poses is None:
        ax.text(0.5, 0.5, f'{floor_name}\n(no data)', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Transform to match paper orientation (x → -x for top-down view)
    x = -poses[:, 1]  # Negate x
    z = poses[:, 3]   # Use z as forward direction
    
    ax.plot(x, z, color=color, linewidth=1.5, label='LeGO-LOAM')
    ax.plot(x[0], z[0], 'go', markersize=8, label='Start')
    ax.plot(x[-1], z[-1], 'ro', markersize=8, label='End')
    
    stats = compute_stats(poses)
    paper = PAPER_VALUES.get(floor_name, {})
    
    # Title with stats
    title = f"{floor_name.replace('_', ' ').title()}\n"
    title += f"L={stats['length']:.0f}m (paper: {paper.get('length_m', '?')}m)\n"
    title += f"Drift={stats['drift']:.2f}m (paper: {paper.get('ate_m', '?')}m)"
    
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    floors = ["5th_floor", "1st_floor", "4th_floor", "2nd_floor"]
    
    # Load all trajectories
    trajectories = {}
    for floor in floors:
        traj_path = TRAJ_DIR / f"{floor}.txt"
        if traj_path.exists():
            trajectories[floor] = load_trajectory(traj_path)
            print(f"Loaded {floor}: {len(trajectories[floor])} poses")
        else:
            trajectories[floor] = None
            print(f"Missing: {floor}")
    
    # Create 2x2 grid plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    for idx, floor in enumerate(floors):
        plot_single_floor(axes[idx], trajectories[floor], floor, FLOOR_COLORS[floor])
    
    fig.suptitle('LeGO-LOAM Trajectories - All Floors\n(Matching Paper Figure 7 Orientation)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = FIG_DIR / "lego_loam_all_floors.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    # Also create individual plots
    for floor in floors:
        if trajectories[floor] is not None:
            fig2, ax2 = plt.subplots(figsize=(10, 10))
            plot_single_floor(ax2, trajectories[floor], floor, FLOOR_COLORS[floor])
            ax2.legend(loc='upper right')
            
            output_path2 = FIG_DIR / f"lego_loam_{floor}.png"
            plt.savefig(output_path2, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path2}")
            plt.close(fig2)
    
    # Show main plot
    plt.show()


if __name__ == "__main__":
    main()
