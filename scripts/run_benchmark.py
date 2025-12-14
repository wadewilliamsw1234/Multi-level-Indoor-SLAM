#!/usr/bin/env python3
"""
Master SLAM Benchmarking Script

Runs all SLAM algorithms on the ISEC dataset and generates paper figures.
Reproduces Table IV and Figure 6 from Kaveti et al. IEEE CASE 2023.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json


# Configuration
ALGORITHMS = {
    'lego_loam': {
        'name': 'LeGO-LOAM',
        'docker_image': 'slam-benchmark/lego-loam:latest',
        'requires_gpu': False,
        'modality': 'lidar',
    },
    'orb_slam3': {
        'name': 'ORB-SLAM3',
        'docker_image': 'slam-benchmark/orb-slam3:latest',
        'requires_gpu': False,
        'modality': 'visual',
    },
    'droid_slam': {
        'name': 'DROID-SLAM',
        'docker_image': 'slam-benchmark/droid-slam:latest',
        'requires_gpu': True,
        'modality': 'deep_learning',
    },
    'vins_fusion': {
        'name': 'VINS-Fusion',
        'docker_image': 'slam-benchmark/vins-fusion:latest',
        'requires_gpu': False,
        'modality': 'visual_inertial',
    },
    'basalt': {
        'name': 'Basalt',
        'docker_image': 'slam-benchmark/basalt:latest',
        'requires_gpu': False,
        'modality': 'visual_inertial',
    },
}

FLOORS = ['5th_floor', '1st_floor', '4th_floor', '2nd_floor']
FULL_SEQUENCE = 'full_sequence'


def run_docker_algorithm(algorithm, floor, data_dir, results_dir, config_dir):
    """Run a SLAM algorithm in Docker container."""
    algo_config = ALGORITHMS[algorithm]
    
    # Build docker command
    cmd = ['docker', 'run', '--rm']
    
    # Add GPU support if needed
    if algo_config['requires_gpu']:
        cmd.extend(['--gpus', 'all'])
    
    # Mount volumes
    cmd.extend([
        '-v', f'{data_dir}:/data/ISEC:ro',
        '-v', f'{results_dir}:/results',
        '-v', f'{config_dir}:/config:ro',
    ])
    
    # Add image and command
    cmd.append(algo_config['docker_image'])
    
    # Algorithm-specific commands
    if algorithm == 'lego_loam':
        cmd.extend(['bash', '-c', 
            f'source /opt/ros/melodic/setup.bash && '
            f'source /catkin_ws/devel/setup.bash && '
            f'/scripts/run_lego_loam.sh {floor}'])
    
    elif algorithm == 'orb_slam3':
        cmd.extend(['bash', '-c',
            f'source /opt/ros/noetic/setup.bash && '
            f'source /catkin_ws/devel/setup.bash && '
            f'/scripts/run_orb_slam3.sh {floor}'])
    
    elif algorithm == 'droid_slam':
        cmd.extend(['python3', '/scripts/droid_slam/run_droid_slam_stereo.py',
                   '--floor', floor])
    
    elif algorithm == 'vins_fusion':
        cmd.extend(['python3', '/scripts/vins_fusion/run_vins_fusion.py',
                   '--floor', floor])
    
    elif algorithm == 'basalt':
        cmd.extend(['bash', '/scripts/basalt/run_basalt.sh', floor])
    
    print(f"\n{'='*60}")
    print(f"Running {algo_config['name']} on {floor}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {algorithm} on {floor}: {e}")
        return False


def build_docker_images(algorithms=None):
    """Build Docker images for specified algorithms."""
    if algorithms is None:
        algorithms = list(ALGORITHMS.keys())
    
    docker_dir = Path(__file__).parent.parent / 'docker'
    
    for algo in algorithms:
        dockerfile = docker_dir / f'Dockerfile.{algo.replace("_", "-")}'
        if dockerfile.exists():
            image_name = ALGORITHMS[algo]['docker_image']
            print(f"Building {image_name}...")
            
            cmd = ['docker', 'build', '-t', image_name, '-f', str(dockerfile), '.']
            subprocess.run(cmd, check=True)


def check_trajectories(results_dir, algorithms=None, floors=None):
    """Check which trajectories already exist."""
    if algorithms is None:
        algorithms = list(ALGORITHMS.keys())
    if floors is None:
        floors = FLOORS
    
    existing = {}
    missing = {}
    
    results_path = Path(results_dir)
    
    for algo in algorithms:
        existing[algo] = []
        missing[algo] = []
        
        for floor in floors:
            traj_file = results_path / 'trajectories' / algo / f'{floor}.txt'
            if traj_file.exists() and traj_file.stat().st_size > 0:
                existing[algo].append(floor)
            else:
                missing[algo].append(floor)
    
    return existing, missing


def run_evaluation(results_dir):
    """Run evaluation to compute metrics and generate Table IV."""
    from scripts.evaluation.evaluate_trajectories import evaluate_all_trajectories
    
    print("\n" + "="*60)
    print("Running Evaluation")
    print("="*60)
    
    results = evaluate_all_trajectories(
        results_dir=results_dir,
        ground_truth_algo='lego_loam',
        output_file=Path(results_dir) / 'metrics' / 'table_iv.csv'
    )
    
    return results


def generate_figures(results_dir):
    """Generate paper figures."""
    from scripts.visualization.generate_paper_figures import (
        plot_figure_6, plot_figure_7_floor_comparison
    )
    
    results_path = Path(results_dir)
    figures_dir = results_path / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Figures")
    print("="*60)
    
    # Figure 6: Perceptual aliasing (requires Basalt with and without LC)
    basalt_no_lc = results_path / 'trajectories' / 'basalt' / 'full_sequence.txt'
    basalt_with_lc = results_path / 'trajectories' / 'basalt' / 'full_sequence_with_lc.txt'
    
    if basalt_no_lc.exists():
        print("Generating Figure 6...")
        plot_figure_6(
            str(basalt_no_lc),
            str(basalt_with_lc) if basalt_with_lc.exists() else None,
            str(figures_dir / 'figure_6_perceptual_aliasing.png')
        )
    else:
        print("Warning: Basalt trajectories not found, skipping Figure 6")
    
    # Figure 7: Floor comparison
    for floor in FLOORS:
        trajectories = {}
        traj_dir = results_path / 'trajectories'
        
        for algo, config in ALGORITHMS.items():
            traj_file = traj_dir / algo / f'{floor}.txt'
            if traj_file.exists():
                trajectories[config['name']] = str(traj_file)
        
        if len(trajectories) >= 2:
            print(f"Generating Figure 7 for {floor}...")
            plot_figure_7_floor_comparison(
                trajectories,
                str(figures_dir / f'figure_7_{floor}_comparison.png'),
                floor
            )


def main():
    parser = argparse.ArgumentParser(
        description='SLAM Benchmarking Suite for ISEC Dataset')
    
    parser.add_argument('--data-dir', type=str, 
                       default='/home/wadewilliams/Dev/shared/datasets/ISEC',
                       help='ISEC dataset directory')
    parser.add_argument('--results-dir', type=str,
                       default='./results',
                       help='Results output directory')
    parser.add_argument('--config-dir', type=str,
                       default='./config',
                       help='Configuration directory')
    
    parser.add_argument('--algorithms', type=str, nargs='+',
                       choices=list(ALGORITHMS.keys()),
                       default=None,
                       help='Algorithms to run (default: all)')
    parser.add_argument('--floors', type=str, nargs='+',
                       default=None,
                       help='Floors to process (default: all)')
    
    parser.add_argument('--build', action='store_true',
                       help='Build Docker images before running')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip algorithms/floors that already have results')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only run evaluation (skip SLAM processing)')
    parser.add_argument('--figures-only', action='store_true',
                       help='Only generate figures')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    config_dir = Path(args.config_dir).resolve()
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    algorithms = args.algorithms or list(ALGORITHMS.keys())
    floors = args.floors or FLOORS
    
    print(f"SLAM Benchmarking Suite")
    print(f"="*60)
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Floors: {', '.join(floors)}")
    print(f"="*60)
    
    # Build Docker images if requested
    if args.build:
        print("\nBuilding Docker images...")
        build_docker_images(algorithms)
    
    # Check existing results
    existing, missing = check_trajectories(results_dir, algorithms, floors)
    
    print("\nExisting trajectories:")
    for algo, floors_done in existing.items():
        if floors_done:
            print(f"  {algo}: {', '.join(floors_done)}")
    
    print("\nMissing trajectories:")
    for algo, floors_missing in missing.items():
        if floors_missing:
            print(f"  {algo}: {', '.join(floors_missing)}")
    
    # Run SLAM algorithms
    if not args.evaluate_only and not args.figures_only:
        for algo in algorithms:
            floors_to_run = missing[algo] if args.skip_existing else floors
            
            for floor in floors_to_run:
                if floor in floors:  # Only run requested floors
                    success = run_docker_algorithm(
                        algo, floor, 
                        str(data_dir), str(results_dir), str(config_dir)
                    )
                    
                    if not success:
                        print(f"Warning: {algo} failed on {floor}")
    
    # Run evaluation
    if not args.figures_only:
        try:
            results = run_evaluation(str(results_dir))
            
            # Save results summary
            summary_file = results_dir / 'metrics' / 'benchmark_summary.json'
            with open(summary_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'algorithms': algorithms,
                    'floors': floors,
                    'results': results
                }, f, indent=2)
            
            print(f"\nResults saved to {summary_file}")
        except Exception as e:
            print(f"Evaluation failed: {e}")
    
    # Generate figures
    try:
        generate_figures(str(results_dir))
    except Exception as e:
        print(f"Figure generation failed: {e}")
    
    print("\n" + "="*60)
    print("Benchmarking Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
