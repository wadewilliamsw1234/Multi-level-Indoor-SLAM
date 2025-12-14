#!/usr/bin/env python3
"""
Generate all figures for SLAM evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
TRAJ_DIR = BASE_DIR / 'results' / 'trajectories'
METRICS_DIR = BASE_DIR / 'results' / 'metrics'
FIGURES_DIR = BASE_DIR / 'results' / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

COLORS = {
    'lego_loam': 'black',
    'orb_slam3': 'red',
    'basalt': 'blue',
    'droid_slam': 'green',
}

LABELS = {
    'lego_loam': 'LeGO-LOAM',
    'orb_slam3': 'ORB-SLAM3',
    'basalt': 'Basalt',
    'droid_slam': 'DROID-SLAM',
}

FLOORS = ['5th_floor', '1st_floor', '4th_floor', '2nd_floor']

def load_trajectory(algo, floor):
    """Load trajectory from file"""
    traj_dir = TRAJ_DIR / algo
    
    if algo == 'droid_slam':
        candidates = [f'{floor}_stereo.txt', f'{floor}.txt']
        for c in candidates:
            if (traj_dir / c).exists():
                data = np.loadtxt(traj_dir / c, comments='#')
                return data[:, 1:4]  # Return positions only
    
    traj_file = traj_dir / f'{floor}.txt'
    if traj_file.exists():
        data = np.loadtxt(traj_file, comments='#')
        return data[:, 1:4]
    
    return None

def load_evaluation_results():
    """Load comprehensive evaluation results"""
    results_file = METRICS_DIR / 'comprehensive_evaluation.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

# ============================================================================
# FIGURE 1: 2D Trajectory Comparison (Paper Figure 7 style)
# ============================================================================

def plot_floor_trajectories_2d(floor, save=True):
    """Plot 2D trajectory comparison for a single floor"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for algo in ['lego_loam', 'orb_slam3', 'basalt', 'droid_slam']:
        positions = load_trajectory(algo, floor)
        if positions is not None:
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=COLORS[algo], label=LABELS[algo], 
                   linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'{floor.replace("_", " ").title()} - Trajectory Comparison', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Add problem region annotations for 5th floor
    if floor == '5th_floor':
        # These would need to be adjusted based on actual trajectory coordinates
        ax.annotate('A', xy=(0.25, 0.75), xycoords='axes fraction', 
                   fontsize=16, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))
        ax.annotate('B', xy=(0.6, 0.4), xycoords='axes fraction',
                   fontsize=16, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    if save:
        output_file = FIGURES_DIR / f'trajectory_2d_{floor}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close()

# ============================================================================
# FIGURE 2: All Floors 2D Grid
# ============================================================================

def plot_all_floors_grid():
    """Plot all floors in a 2x2 grid"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, floor in enumerate(FLOORS):
        ax = axes[idx]
        
        for algo in ['lego_loam', 'orb_slam3', 'basalt', 'droid_slam']:
            positions = load_trajectory(algo, floor)
            if positions is not None:
                ax.plot(positions[:, 0], positions[:, 1],
                       color=COLORS[algo], label=LABELS[algo],
                       linewidth=1.2, alpha=0.8)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(floor.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=8)
    
    plt.suptitle('ISEC Dataset - Trajectory Comparison Across All Floors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = FIGURES_DIR / 'all_floors_grid.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# FIGURE 3: 3D Multi-Floor Visualization
# ============================================================================

def plot_3d_trajectory(algo):
    """Plot 3D trajectory showing all floors with vertical separation"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    floor_heights = {'5th_floor': 16, '4th_floor': 12, '2nd_floor': 4, '1st_floor': 0}
    floor_colors = {'5th_floor': 'red', '4th_floor': 'orange', '2nd_floor': 'green', '1st_floor': 'blue'}
    
    for floor in FLOORS:
        positions = load_trajectory(algo, floor)
        if positions is not None:
            # Use actual Z or offset by floor height
            z = positions[:, 2] + floor_heights[floor]
            ax.plot(positions[:, 0], positions[:, 1], z,
                   color=floor_colors[floor], label=floor.replace('_', ' ').title(),
                   linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'{LABELS[algo]} - Multi-Floor 3D Trajectory', fontsize=14)
    ax.legend()
    
    output_file = FIGURES_DIR / f'trajectory_3d_{algo}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# FIGURE 4: Error Accumulation Plot
# ============================================================================

def plot_error_accumulation(floor):
    """Plot error vs trajectory progress for all algorithms"""
    results = load_evaluation_results()
    if results is None:
        print("  No evaluation results found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for algo in ['orb_slam3', 'basalt', 'droid_slam']:
        if algo in results and floor in results[algo]:
            r = results[algo][floor]
            if 'ate_errors' in r:
                errors = np.array(r['ate_errors'])
                progress = np.linspace(0, 100, len(errors))
                ax.plot(progress, errors, color=COLORS[algo], 
                       label=LABELS[algo], linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Trajectory Progress (%)', fontsize=12)
    ax.set_ylabel('Error vs LeGO-LOAM (m)', fontsize=12)
    ax.set_title(f'{floor.replace("_", " ").title()} - Error Accumulation', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add problem region shading
    if floor == '5th_floor':
        ax.axvspan(25, 40, alpha=0.2, color='red', label='Region A (dynamic)')
        ax.axvspan(55, 70, alpha=0.2, color='orange', label='Region B (featureless)')
    
    plt.tight_layout()
    
    output_file = FIGURES_DIR / f'error_accumulation_{floor}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# FIGURE 5: Segment Performance Heatmap
# ============================================================================

def plot_segment_heatmap(floor):
    """Plot segment-wise error as heatmap"""
    results = load_evaluation_results()
    if results is None:
        return
    
    algos = ['orb_slam3', 'basalt', 'droid_slam']
    num_segments = 10
    
    # Build data matrix
    data = np.zeros((len(algos), num_segments))
    
    for i, algo in enumerate(algos):
        if algo in results and floor in results[algo]:
            r = results[algo][floor]
            if 'segments' in r:
                for seg in r['segments']:
                    seg_idx = seg['segment'] - 1
                    if seg_idx < num_segments:
                        data[i, seg_idx] = seg['rmse']
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(range(num_segments))
    ax.set_xticklabels([f'{i*10}-{(i+1)*10}%' for i in range(num_segments)])
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels([LABELS[a] for a in algos])
    
    ax.set_xlabel('Trajectory Segment', fontsize=12)
    ax.set_title(f'{floor.replace("_", " ").title()} - Segment-wise RMSE (m)', fontsize=14)
    
    # Add values to cells
    for i in range(len(algos)):
        for j in range(num_segments):
            if data[i, j] > 0:
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im, ax=ax, label='RMSE (m)')
    plt.tight_layout()
    
    output_file = FIGURES_DIR / f'segment_heatmap_{floor}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# FIGURE 6: Bar Chart Comparison with Paper
# ============================================================================

def plot_paper_comparison():
    """Bar chart comparing our results to paper results"""
    results = load_evaluation_results()
    if results is None:
        return
    
    paper_results = {
        'orb_slam3': {'5th_floor': 0.516, '1st_floor': 0.949, '4th_floor': 0.483, '2nd_floor': 0.310},
        'droid_slam': {'5th_floor': 0.441, '1st_floor': 0.666, '4th_floor': 0.112, '2nd_floor': 0.214},
        'lego_loam': {'5th_floor': 0.395, '1st_floor': 0.256, '4th_floor': 0.789, '2nd_floor': 0.286},
        'basalt': {'5th_floor': 1.214, '1st_floor': 4.043, '4th_floor': 1.809, '2nd_floor': 3.054},
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, floor in enumerate(FLOORS):
        ax = axes[idx]
        
        algos = ['lego_loam', 'orb_slam3', 'basalt', 'droid_slam']
        x = np.arange(len(algos))
        width = 0.35
        
        ours = []
        paper = []
        
        for algo in algos:
            # Our results - endpoint drift
            if algo in results and floor in results[algo]:
                ours.append(results[algo][floor].get('endpoint_drift', 0))
            else:
                ours.append(0)
            
            # Paper results
            paper.append(paper_results.get(algo, {}).get(floor, 0))
        
        bars1 = ax.bar(x - width/2, ours, width, label='Ours', color='steelblue')
        bars2 = ax.bar(x + width/2, paper, width, label='Paper', color='coral')
        
        ax.set_ylabel('Endpoint Drift (m)')
        ax.set_title(floor.replace('_', ' ').title())
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[a] for a in algos], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Endpoint Drift Comparison: Our Results vs Paper', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = FIGURES_DIR / 'paper_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# FIGURE 7: RPE Distribution Box Plot
# ============================================================================

def plot_rpe_boxplot():
    """Box plot of RPE distribution across algorithms"""
    results = load_evaluation_results()
    if results is None:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data = []
    labels = []
    
    for algo in ['lego_loam', 'orb_slam3', 'basalt', 'droid_slam']:
        algo_rpes = []
        for floor in FLOORS:
            if algo in results and floor in results[algo]:
                r = results[algo][floor]
                if 'rpe_1m' in r:
                    algo_rpes.append(r['rpe_1m'].get('rmse', 0))
        if algo_rpes:
            data.append(algo_rpes)
            labels.append(LABELS[algo])
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    colors = [COLORS[a] for a in ['lego_loam', 'orb_slam3', 'basalt', 'droid_slam'] if LABELS[a] in labels]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('RPE RMSE (1m segments) %', fontsize=12)
    ax.set_title('Relative Pose Error Distribution Across All Floors', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = FIGURES_DIR / 'rpe_boxplot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def generate_all_figures():
    """Generate all figures"""
    print("=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    print("\n[1] 2D Trajectory plots per floor...")
    for floor in FLOORS:
        plot_floor_trajectories_2d(floor)
    
    print("\n[2] All floors grid...")
    plot_all_floors_grid()
    
    print("\n[3] 3D trajectories...")
    for algo in ['lego_loam', 'basalt']:
        plot_3d_trajectory(algo)
    
    print("\n[4] Error accumulation plots...")
    for floor in FLOORS:
        plot_error_accumulation(floor)
    
    print("\n[5] Segment heatmaps...")
    for floor in FLOORS:
        plot_segment_heatmap(floor)
    
    print("\n[6] Paper comparison...")
    plot_paper_comparison()
    
    print("\n[7] RPE boxplot...")
    plot_rpe_boxplot()
    
    print("\n" + "=" * 70)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    generate_all_figures()
