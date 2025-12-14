#!/usr/bin/env python3
"""
Generate Figure 7: 5th Floor Trajectory Comparison (Top-down XY view)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
TRAJ_DIR = BASE_DIR / 'results' / 'trajectories'
FIG_DIR = BASE_DIR / 'results' / 'figures'
FIG_DIR.mkdir(exist_ok=True)

def load_trajectory(filepath):
    """Load TUM format trajectory, return positions"""
    data = np.loadtxt(filepath)
    return data[:, 1:4]  # x, y, z

def main():
    # Load trajectories
    trajectories = {}
    
    algos = {
        'lego_loam': ('LeGO-LOAM (Ground Truth)', 'black', 2.5, '-'),
        'orb_slam3': ('ORB-SLAM3', 'tab:red', 1.5, '-'),
        'droid_slam': ('DROID-SLAM', 'tab:blue', 1.5, '-'),
    }
    
    for algo, (label, color, lw, ls) in algos.items():
        traj_file = TRAJ_DIR / algo / '5th_floor.txt'
        if traj_file.exists():
            trajectories[algo] = {
                'pos': load_trajectory(traj_file),
                'label': label,
                'color': color,
                'lw': lw,
                'ls': ls
            }
            print(f"Loaded {algo}: {len(trajectories[algo]['pos'])} poses")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each trajectory
    for algo, data in trajectories.items():
        pos = data['pos']
        ax.plot(pos[:, 0], pos[:, 1], 
                label=data['label'], 
                color=data['color'],
                linewidth=data['lw'],
                linestyle=data['ls'],
                alpha=0.8)
        
        # Mark start and end
        ax.scatter(pos[0, 0], pos[0, 1], c=data['color'], s=100, marker='o', zorder=5, edgecolors='white', linewidths=2)
        ax.scatter(pos[-1, 0], pos[-1, 1], c=data['color'], s=100, marker='s', zorder=5, edgecolors='white', linewidths=2)
    
    # Highlight challenging regions (approximate based on trajectory shape)
    # Region A: Dynamic content area
    # Region B: Featureless corridor during tight turn
    
    # Find interesting regions from LeGO-LOAM trajectory
    if 'lego_loam' in trajectories:
        pos = trajectories['lego_loam']['pos']
        
        # Region A - around the middle where jagged artifacts appear (estimate)
        region_a_center = (pos[len(pos)//4, 0], pos[len(pos)//4, 1])
        
        # Region B - at the far end (corridor turnaround)
        max_x_idx = np.argmax(pos[:, 0])
        region_b_center = (pos[max_x_idx, 0], pos[max_x_idx, 1])
        
        # Draw region circles
        circle_a = plt.Circle(region_a_center, 3, fill=False, color='orange', 
                             linestyle='--', linewidth=2, label='Region A: Dynamic content')
        circle_b = plt.Circle(region_b_center, 3, fill=False, color='purple',
                             linestyle='--', linewidth=2, label='Region B: Featureless corridor')
        ax.add_patch(circle_a)
        ax.add_patch(circle_b)
        
        # Add labels
        ax.annotate('A', region_a_center, fontsize=14, fontweight='bold', 
                   ha='center', va='center', color='orange')
        ax.annotate('B', region_b_center, fontsize=14, fontweight='bold',
                   ha='center', va='center', color='purple')
    
    # Formatting
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Figure 7: 5th Floor Trajectory Comparison\n(Top-down XY view)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add legend for markers
    ax.scatter([], [], c='gray', s=100, marker='o', label='Start', edgecolors='white', linewidths=2)
    ax.scatter([], [], c='gray', s=100, marker='s', label='End', edgecolors='white', linewidths=2)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = FIG_DIR / 'figure_7_5th_floor.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {output_path}")
    
    # Also save PDF
    plt.savefig(FIG_DIR / 'figure_7_5th_floor.pdf', bbox_inches='tight')
    print(f"✅ PDF saved to: {FIG_DIR / 'figure_7_5th_floor.pdf'}")
    
    plt.close()

if __name__ == '__main__':
    main()
