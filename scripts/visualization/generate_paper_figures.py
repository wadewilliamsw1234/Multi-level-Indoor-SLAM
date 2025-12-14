#!/usr/bin/env python3
"""
Generate Figure 6 from the paper: Perceptual Aliasing Visualization

Figure 6a: Basalt trajectory on full ISEC sequence WITHOUT loop closure
  - 3D plot showing separate floors stacked vertically
  - Each floor colored differently
  - Shows correct vertical separation

Figure 6b: Basalt trajectory WITH loop closure enabled  
  - Same view as 6a
  - Green line segments showing incorrect loop closure constraints between floors
  - Floors incorrectly merged due to visual similarity (perceptual aliasing)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse


# Floor colors matching paper style
FLOOR_COLORS = {
    '5th_floor': '#1f77b4',  # Blue
    '4th_floor': '#ff7f0e',  # Orange  
    '3rd_floor': '#2ca02c',  # Green
    '2nd_floor': '#d62728',  # Red
    '1st_floor': '#9467bd',  # Purple
    'transit': '#7f7f7f',    # Gray
}

# Expected floor heights (meters) - ISEC building
FLOOR_HEIGHTS = {
    '1st_floor': 0.0,
    '2nd_floor': 4.5,
    '3rd_floor': 9.0,
    '4th_floor': 13.5,
    '5th_floor': 18.0,
}


def load_trajectory(filepath):
    """Load trajectory in TUM format."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]
    
    return timestamps, positions, quaternions


def segment_by_floor(timestamps, positions, floor_transitions=None):
    """
    Segment trajectory by floor based on Z-height or provided transitions.
    
    Returns dict mapping floor names to (timestamps, positions) tuples.
    """
    if floor_transitions is None:
        # Automatic segmentation based on Z-height
        z = positions[:, 2]
        segments = {}
        
        # Simple threshold-based segmentation
        for floor_name, height in FLOOR_HEIGHTS.items():
            mask = np.abs(z - height) < 2.0  # Within 2m of floor height
            if np.any(mask):
                segments[floor_name] = (timestamps[mask], positions[mask])
        
        return segments
    else:
        # Use provided transition indices
        segments = {}
        for floor_name, (start_idx, end_idx) in floor_transitions.items():
            segments[floor_name] = (
                timestamps[start_idx:end_idx],
                positions[start_idx:end_idx]
            )
        return segments


def detect_loop_closures(positions_with_lc, positions_no_lc, threshold=5.0):
    """
    Detect where loop closures caused trajectory changes.
    
    Compares trajectories with and without loop closure to find
    where significant corrections were made (indicating loop closure).
    
    Returns list of (idx1, idx2) tuples indicating loop closure connections.
    """
    loop_closures = []
    
    # Find points where trajectories diverge significantly
    if len(positions_with_lc) != len(positions_no_lc):
        # Different lengths - can't directly compare
        return loop_closures
    
    diff = np.linalg.norm(positions_with_lc - positions_no_lc, axis=1)
    
    # Find discontinuities in the difference
    diff_grad = np.gradient(diff)
    jumps = np.where(np.abs(diff_grad) > threshold)[0]
    
    # Group jumps into loop closure events
    for i, jump_idx in enumerate(jumps):
        # Look for similar positions that might be incorrectly matched
        pos_at_jump = positions_with_lc[jump_idx]
        
        # Find distant positions that are now close (false positives)
        distances = np.linalg.norm(positions_with_lc - pos_at_jump, axis=1)
        close_but_far = np.where(
            (distances < 3.0) &  # Close in corrected trajectory
            (np.abs(np.arange(len(distances)) - jump_idx) > 100)  # Far in time
        )[0]
        
        for match_idx in close_but_far:
            loop_closures.append((jump_idx, match_idx))
    
    return loop_closures


def plot_figure_6(traj_no_lc_path, traj_with_lc_path=None, output_path=None,
                  floor_segments=None):
    """
    Generate Figure 6 from the paper.
    
    Args:
        traj_no_lc_path: Path to trajectory without loop closure
        traj_with_lc_path: Path to trajectory with loop closure (optional)
        output_path: Output file path
        floor_segments: Dict mapping floor names to (start_idx, end_idx)
    """
    # Load trajectories
    ts_no_lc, pos_no_lc, _ = load_trajectory(traj_no_lc_path)
    
    has_lc_comparison = traj_with_lc_path is not None
    if has_lc_comparison:
        ts_with_lc, pos_with_lc, _ = load_trajectory(traj_with_lc_path)
    
    # Segment by floor
    segments_no_lc = segment_by_floor(ts_no_lc, pos_no_lc, floor_segments)
    
    # Create figure
    if has_lc_comparison:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                                  subplot_kw={'projection': '3d'})
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                                subplot_kw={'projection': '3d'})
        axes = [ax]
    
    # Figure 6a: Without loop closure
    ax = axes[0]
    
    for floor_name, (ts, pos) in segments_no_lc.items():
        color = FLOOR_COLORS.get(floor_name, '#333333')
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
               color=color, label=floor_name, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('(a) Without Loop Closure', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    # Set equal aspect ratio
    max_range = np.max([
        np.max(pos_no_lc[:, 0]) - np.min(pos_no_lc[:, 0]),
        np.max(pos_no_lc[:, 1]) - np.min(pos_no_lc[:, 1]),
        np.max(pos_no_lc[:, 2]) - np.min(pos_no_lc[:, 2])
    ]) / 2.0
    
    mid_x = (np.max(pos_no_lc[:, 0]) + np.min(pos_no_lc[:, 0])) / 2
    mid_y = (np.max(pos_no_lc[:, 1]) + np.min(pos_no_lc[:, 1])) / 2
    mid_z = (np.max(pos_no_lc[:, 2]) + np.min(pos_no_lc[:, 2])) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=-60)
    
    # Figure 6b: With loop closure (showing incorrect constraints)
    if has_lc_comparison:
        ax = axes[1]
        
        # Segment the with-LC trajectory
        segments_with_lc = segment_by_floor(ts_with_lc, pos_with_lc, floor_segments)
        
        for floor_name, (ts, pos) in segments_with_lc.items():
            color = FLOOR_COLORS.get(floor_name, '#333333')
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                   color=color, label=floor_name, linewidth=1.5, alpha=0.8)
        
        # Detect and draw incorrect loop closures
        loop_closures = detect_loop_closures(pos_with_lc, pos_no_lc)
        
        for idx1, idx2 in loop_closures:
            ax.plot([pos_with_lc[idx1, 0], pos_with_lc[idx2, 0]],
                   [pos_with_lc[idx1, 1], pos_with_lc[idx2, 1]],
                   [pos_with_lc[idx1, 2], pos_with_lc[idx2, 2]],
                   'g-', linewidth=2, alpha=0.7)
        
        # Add legend entry for loop closures
        if loop_closures:
            ax.plot([], [], 'g-', linewidth=2, label='Incorrect Loop Closures')
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('(b) With Loop Closure (Perceptual Aliasing)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.view_init(elev=25, azim=-60)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    plt.show()
    
    return fig


def plot_figure_7_floor_comparison(trajectories_dict, output_path=None, 
                                    floor='5th_floor'):
    """
    Generate Figure 7: Floor trajectory comparison showing all algorithms.
    
    Args:
        trajectories_dict: Dict mapping algorithm names to trajectory file paths
        output_path: Output file path
        floor: Which floor to visualize
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Algorithm colors matching paper style
    algo_colors = {
        'LeGO-LOAM': '#1f77b4',
        'ORB-SLAM3': '#d62728',
        'DROID-SLAM': '#ff7f0e',
        'VINS-Fusion': '#2ca02c',
        'Basalt': '#9467bd',
        'SVO': '#8c564b',
        'SVO-inertial': '#e377c2',
    }
    
    all_positions = []
    
    for algo_name, traj_path in trajectories_dict.items():
        try:
            _, positions, _ = load_trajectory(traj_path)
            color = algo_colors.get(algo_name, '#333333')
            
            # Plot XY (top-down view)
            ax.plot(positions[:, 0], positions[:, 1],
                   color=color, label=algo_name, linewidth=1.5, alpha=0.8)
            
            all_positions.append(positions)
        except Exception as e:
            print(f"Warning: Could not load {algo_name}: {e}")
    
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_title(f'{floor.replace("_", " ").title()} - Algorithm Comparison',
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add region annotations (A and B from paper)
    # These should be adjusted based on actual trajectory coordinates
    if all_positions:
        combined = np.vstack(all_positions)
        x_range = np.max(combined[:, 0]) - np.min(combined[:, 0])
        y_range = np.max(combined[:, 1]) - np.min(combined[:, 1])
        
        # Region A: Dynamic content area (example coordinates)
        # Region B: Featureless corridor (example coordinates)
        # These need to be calibrated to actual ISEC data
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Generate paper figures for SLAM evaluation')
    
    subparsers = parser.add_subparsers(dest='command', help='Figure to generate')
    
    # Figure 6 subcommand
    fig6_parser = subparsers.add_parser('figure6', 
        help='Generate Figure 6 (perceptual aliasing)')
    fig6_parser.add_argument('--no-lc', type=str, required=True,
                            help='Trajectory without loop closure')
    fig6_parser.add_argument('--with-lc', type=str, default=None,
                            help='Trajectory with loop closure')
    fig6_parser.add_argument('--output', type=str, 
                            default='figure_6_perceptual_aliasing.png',
                            help='Output file path')
    
    # Figure 7 subcommand
    fig7_parser = subparsers.add_parser('figure7',
        help='Generate Figure 7 (floor comparison)')
    fig7_parser.add_argument('--trajectories-dir', type=str, required=True,
                            help='Directory containing trajectory files')
    fig7_parser.add_argument('--floor', type=str, default='5th_floor',
                            help='Floor to visualize')
    fig7_parser.add_argument('--output', type=str,
                            default='figure_7_floor_comparison.png',
                            help='Output file path')
    
    args = parser.parse_args()
    
    if args.command == 'figure6':
        plot_figure_6(args.no_lc, args.with_lc, args.output)
        
    elif args.command == 'figure7':
        traj_dir = Path(args.trajectories_dir)
        
        trajectories = {}
        algo_dirs = ['lego_loam', 'orb_slam3', 'droid_slam', 'vins_fusion', 'basalt']
        algo_names = ['LeGO-LOAM', 'ORB-SLAM3', 'DROID-SLAM', 'VINS-Fusion', 'Basalt']
        
        for algo_dir, algo_name in zip(algo_dirs, algo_names):
            traj_file = traj_dir / algo_dir / f"{args.floor}.txt"
            if traj_file.exists():
                trajectories[algo_name] = str(traj_file)
        
        if trajectories:
            plot_figure_7_floor_comparison(trajectories, args.output, args.floor)
        else:
            print(f"No trajectories found in {traj_dir}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
