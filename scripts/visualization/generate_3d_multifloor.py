#!/usr/bin/env python3
"""
Generate 3D Multi-Floor Visualization
Shows all floors stacked vertically using LeGO-LOAM trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
TRAJ_DIR = BASE_DIR / 'results' / 'trajectories'
FIG_DIR = BASE_DIR / 'results' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

def load_trajectory(filepath):
    """Load TUM format trajectory"""
    data = np.loadtxt(filepath)
    return data[:, 1:4]

def main():
    # Floor heights (approximate, for visualization)
    floor_heights = {
        '5th_floor': 15.0,  # ~3m per floor from ground
        '4th_floor': 12.0,
        '2nd_floor': 6.0,
        '1st_floor': 3.0,
    }
    
    floor_colors = {
        '5th_floor': 'tab:blue',
        '4th_floor': 'tab:orange', 
        '2nd_floor': 'tab:green',
        '1st_floor': 'tab:red',
    }
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Load and plot each floor
    for floor, height in floor_heights.items():
        traj_file = TRAJ_DIR / 'lego_loam' / f'{floor}.txt'
        if traj_file.exists():
            pos = load_trajectory(traj_file)
            
            # Offset to center trajectories
            pos_centered = pos - pos.mean(axis=0)
            
            # Plot at floor height
            ax.plot(pos_centered[:, 0], pos_centered[:, 1], 
                   np.full(len(pos), height),
                   label=floor.replace('_', ' ').title(),
                   color=floor_colors[floor],
                   linewidth=2,
                   alpha=0.8)
            
            # Mark start point
            ax.scatter(pos_centered[0, 0], pos_centered[0, 1], height,
                      c=floor_colors[floor], s=100, marker='o',
                      edgecolors='white', linewidths=2, zorder=5)
            
            print(f"Loaded {floor}: {len(pos)} poses")
    
    # Add floor planes (semi-transparent)
    for floor, height in floor_heights.items():
        xx, yy = np.meshgrid(np.linspace(-30, 30, 2), np.linspace(-30, 30, 2))
        zz = np.full_like(xx, height)
        ax.plot_surface(xx, yy, zz, alpha=0.1, color=floor_colors[floor])
    
    # Formatting
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Floor Height (m)', fontsize=11)
    ax.set_title('ISEC Building Multi-Floor Trajectories\n(LeGO-LOAM Ground Truth)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    
    # Save
    output_path = FIG_DIR / 'trajectory_3d_multifloor.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {output_path}")
    
    # Save from different angle
    ax.view_init(elev=5, azim=0)  # Side view
    plt.savefig(FIG_DIR / 'trajectory_3d_multifloor_side.png', dpi=150, bbox_inches='tight')
    print(f"✅ Side view saved to: {FIG_DIR / 'trajectory_3d_multifloor_side.png'}")
    
    plt.close()
    
    # Also create a figure showing the perceptual aliasing problem
    # (how similar floor layouts look from above)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    for idx, (floor, height) in enumerate(floor_heights.items()):
        ax = axes[idx // 2, idx % 2]
        traj_file = TRAJ_DIR / 'lego_loam' / f'{floor}.txt'
        if traj_file.exists():
            pos = load_trajectory(traj_file)
            pos_centered = pos - pos.mean(axis=0)
            
            ax.plot(pos_centered[:, 0], pos_centered[:, 1],
                   color=floor_colors[floor], linewidth=2)
            ax.scatter(pos_centered[0, 0], pos_centered[0, 1],
                      c=floor_colors[floor], s=100, marker='o',
                      edgecolors='white', linewidths=2, zorder=5)
            
            ax.set_title(floor.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-35, 35)
            ax.set_ylim(-35, 35)
    
    fig.suptitle('Perceptual Aliasing: Similar Floor Layouts in ISEC Building\n'
                '(5th, 4th, and 2nd floors have nearly identical layouts)',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'perceptual_aliasing_floors.png', dpi=150, bbox_inches='tight')
    print(f"✅ Perceptual aliasing figure saved to: {FIG_DIR / 'perceptual_aliasing_floors.png'}")
    
    plt.close()

if __name__ == '__main__':
    main()
