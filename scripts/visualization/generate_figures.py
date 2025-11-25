#!/usr/bin/env python3
"""
Figure Generation Script
========================

Generates figures from the paper:
- Figure 6: Perceptual aliasing visualization (Basalt with/without loop closure)
- Figure 7: 5th floor trajectory comparison
- Additional 3D trajectory visualizations
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Plotting imports
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Color scheme for algorithms
ALGORITHM_COLORS = {
    'orb_slam3': '#E41A1C',      # Red
    'svo': '#377EB8',             # Blue
    'svo_inertial': '#4DAF4A',    # Green
    'vins_fusion': '#984EA3',     # Purple
    'basalt': '#FF7F00',          # Orange
    'basalt_lc': '#FFFF33',       # Yellow
    'droid_slam': '#A65628',      # Brown
    'lego_loam': '#F781BF',       # Pink
    'mcslam': '#999999',          # Gray
}

# Floor colors for multi-floor visualization
FLOOR_COLORS = {
    '5th_floor': '#1f77b4',
    '4th_floor': '#ff7f0e', 
    '3rd_floor': '#2ca02c',
    '2nd_floor': '#d62728',
    '1st_floor': '#9467bd',
}


def load_trajectory(filepath: Path) -> Optional[np.ndarray]:
    """
    Load trajectory from TUM format file.
    
    Returns:
        Nx8 array: [timestamp, tx, ty, tz, qx, qy, qz, qw]
    """
    if not filepath.exists():
        return None
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                values = [float(v) for v in line.split()]
                if len(values) >= 8:
                    data.append(values[:8])
    
    return np.array(data) if data else None


def load_all_trajectories(
    results_dir: Path,
    sequences: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all trajectories from results directory.
    
    Returns:
        {algorithm: {sequence: trajectory_array}}
    """
    trajectories = {}
    traj_dir = results_dir / 'trajectories'
    
    if not traj_dir.exists():
        return trajectories
    
    for algo_dir in traj_dir.iterdir():
        if algo_dir.is_dir():
            algorithm = algo_dir.name
            trajectories[algorithm] = {}
            
            for traj_file in algo_dir.glob('*.txt'):
                sequence = traj_file.stem
                if sequences is None or sequence in sequences:
                    traj = load_trajectory(traj_file)
                    if traj is not None:
                        trajectories[algorithm][sequence] = traj
    
    return trajectories


def plot_figure_6(
    basalt_no_lc: np.ndarray,
    basalt_with_lc: np.ndarray,
    floor_segments: Optional[Dict[str, Tuple[int, int]]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Generate Figure 6: Perceptual aliasing demonstration.
    
    Shows Basalt trajectory on full ISEC sequence:
    (a) Without loop closure - floors correctly separated
    (b) With loop closure - floors incorrectly merged
    
    Args:
        basalt_no_lc: Trajectory without loop closure
        basalt_with_lc: Trajectory with loop closure
        floor_segments: Dict mapping floor name to (start_idx, end_idx)
        output_path: Path to save figure
        show: Whether to display figure
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})
    
    # Default floor segments if not provided
    if floor_segments is None:
        # Approximate based on typical trajectory
        n = len(basalt_no_lc)
        floor_segments = {
            '5th_floor': (0, int(n * 0.28)),
            '1st_floor': (int(n * 0.28), int(n * 0.36)),
            '4th_floor': (int(n * 0.50), int(n * 0.58)),
            '2nd_floor': (int(n * 0.66), int(n * 0.83)),
        }
    
    def plot_trajectory_3d(ax, traj, title, show_edges=False):
        """Plot 3D trajectory with floor coloring."""
        for floor, (start, end) in floor_segments.items():
            if start < len(traj) and end <= len(traj):
                segment = traj[start:end]
                color = FLOOR_COLORS.get(floor, 'gray')
                ax.plot(
                    segment[:, 1], segment[:, 2], segment[:, 3],
                    c=color, label=floor, linewidth=1.5
                )
        
        # Plot transit segments in black
        sorted_floors = sorted(floor_segments.items(), key=lambda x: x[1][0])
        for i in range(len(sorted_floors) - 1):
            _, (_, end1) = sorted_floors[i]
            _, (start2, _) = sorted_floors[i + 1]
            if end1 < start2 and start2 < len(traj):
                transit = traj[end1:start2]
                ax.plot(
                    transit[:, 1], transit[:, 2], transit[:, 3],
                    c='black', linewidth=0.5, alpha=0.5
                )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.legend(loc='upper left', fontsize=8)
    
    # Figure 6a: Without loop closure
    plot_trajectory_3d(axes[0], basalt_no_lc, '(a) Without Loop Closure')
    
    # Figure 6b: With loop closure
    plot_trajectory_3d(axes[1], basalt_with_lc, '(b) With Loop Closure')
    
    # Add incorrect loop closure edges (green lines between floors)
    # This would require actual loop closure constraint data
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 6 to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_figure_7(
    trajectories: Dict[str, np.ndarray],
    ground_truth: Optional[np.ndarray] = None,
    regions: Optional[Dict[str, Tuple[float, float]]] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Generate Figure 7: 5th floor trajectory comparison.
    
    Top-down (XY) view of 5th floor with all algorithms overlaid.
    Highlights regions A (dynamic content) and B (featureless corridor).
    
    Args:
        trajectories: Dict mapping algorithm name to trajectory array
        ground_truth: Optional ground truth trajectory
        regions: Dict with region names and (x, y) coordinates
        output_path: Path to save figure
        show: Whether to display figure
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot ground truth first (if available)
    if ground_truth is not None:
        ax.plot(
            ground_truth[:, 1], ground_truth[:, 2],
            'k--', linewidth=2, label='Ground Truth', alpha=0.5
        )
    
    # Plot each algorithm's trajectory
    for algo_name, traj in trajectories.items():
        color = ALGORITHM_COLORS.get(algo_name, 'gray')
        display_name = algo_name.replace('_', '-').upper()
        ax.plot(
            traj[:, 1], traj[:, 2],
            c=color, linewidth=1.5, label=display_name
        )
    
    # Highlight regions
    if regions is None:
        # Default regions (approximate)
        regions = {
            'A': (5, 15),   # Dynamic content area
            'B': (-5, 25),  # Featureless corridor
        }
    
    for region_name, (rx, ry) in regions.items():
        circle = plt.Circle(
            (rx, ry), radius=3,
            fill=False, color='black', linestyle='--', linewidth=2
        )
        ax.add_patch(circle)
        ax.annotate(
            region_name, (rx, ry),
            fontsize=14, fontweight='bold',
            ha='center', va='center'
        )
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('5th Floor Trajectory Comparison', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 7 to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_3d_interactive(
    trajectories: Dict[str, Dict[str, np.ndarray]],
    output_path: Optional[Path] = None,
):
    """
    Generate interactive 3D trajectory visualization using Plotly.
    
    Args:
        trajectories: {algorithm: {sequence: trajectory}}
        output_path: Path to save HTML file
    """
    if not HAS_PLOTLY:
        print("Plotly not available")
        return
    
    fig = go.Figure()
    
    for algorithm, sequences in trajectories.items():
        color = ALGORITHM_COLORS.get(algorithm, 'gray')
        
        for sequence, traj in sequences.items():
            display_name = f"{algorithm}/{sequence}"
            
            fig.add_trace(go.Scatter3d(
                x=traj[:, 1],
                y=traj[:, 2],
                z=traj[:, 3],
                mode='lines',
                name=display_name,
                line=dict(color=color, width=2),
            ))
    
    fig.update_layout(
        title='SLAM Trajectory Comparison (3D)',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Saved interactive 3D plot to {output_path}")
    
    return fig


def plot_all_floors_overview(
    trajectories: Dict[str, Dict[str, np.ndarray]],
    output_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Generate overview figure with all floor trajectories.
    
    Creates a 2x2 grid with each floor's trajectories.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    
    floors = ['5th_floor', '1st_floor', '4th_floor', '2nd_floor']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, floor in enumerate(floors):
        ax = axes[idx]
        
        for algorithm, sequences in trajectories.items():
            if floor in sequences:
                traj = sequences[floor]
                color = ALGORITHM_COLORS.get(algorithm, 'gray')
                display_name = algorithm.replace('_', '-')
                ax.plot(
                    traj[:, 1], traj[:, 2],
                    c=color, linewidth=1.5, label=display_name
                )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(floor.replace('_', ' ').title())
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved overview figure to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Generate trajectory figures')
    
    parser.add_argument('--results-dir', type=Path, default=Path('/results'),
                       help='Results directory')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for figures')
    parser.add_argument('--figure', choices=['6', '7', 'overview', '3d', 'all'],
                       default='all', help='Which figure to generate')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'],
                       default='png', help='Output format')
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or args.results_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trajectories
    trajectories = load_all_trajectories(args.results_dir)
    
    if not trajectories:
        print("No trajectories found in results directory.")
        return
    
    print(f"Found trajectories for: {list(trajectories.keys())}")
    
    # Generate requested figures
    if args.figure in ['6', 'all']:
        # Figure 6: Basalt perceptual aliasing
        if 'basalt' in trajectories and 'full_sequence' in trajectories.get('basalt', {}):
            basalt_no_lc = trajectories['basalt'].get('full_sequence')
            basalt_with_lc = trajectories.get('basalt_lc', {}).get('full_sequence')
            
            if basalt_no_lc is not None:
                if basalt_with_lc is None:
                    basalt_with_lc = basalt_no_lc  # Use same if LC version not available
                
                plot_figure_6(
                    basalt_no_lc, basalt_with_lc,
                    output_path=output_dir / f'figure_6_perceptual_aliasing.{args.format}',
                    show=args.show
                )
        else:
            print("Basalt full sequence not found, skipping Figure 6")
    
    if args.figure in ['7', 'all']:
        # Figure 7: 5th floor comparison
        floor_trajs = {}
        for algo, seqs in trajectories.items():
            if '5th_floor' in seqs:
                floor_trajs[algo] = seqs['5th_floor']
        
        if floor_trajs:
            gt = trajectories.get('lego_loam', {}).get('5th_floor')
            plot_figure_7(
                floor_trajs,
                ground_truth=gt,
                output_path=output_dir / f'figure_7_5th_floor_comparison.{args.format}',
                show=args.show
            )
        else:
            print("No 5th floor trajectories found, skipping Figure 7")
    
    if args.figure in ['overview', 'all']:
        plot_all_floors_overview(
            trajectories,
            output_path=output_dir / f'all_floors_overview.{args.format}',
            show=args.show
        )
    
    if args.figure in ['3d', 'all']:
        plot_3d_interactive(
            trajectories,
            output_path=output_dir / 'trajectory_3d_interactive.html'
        )
    
    print(f"\nFigures saved to: {output_dir}")


if __name__ == '__main__':
    main()
