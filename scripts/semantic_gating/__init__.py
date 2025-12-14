"""
Semantic Gating Module for Multi-Floor SLAM

Provides floor detection from IMU data and semantic loop closure gating
to prevent perceptual aliasing in buildings with similar floor layouts.

Components:
- floor_detector: IMU-based elevator detection and floor labeling
- loop_closure_gate: Semantic filtering of loop closure candidates
- semantic_gating_pipeline: Complete processing pipeline
- orb_slam3_integration: Post-processing for ORB-SLAM3 trajectories
- droid_slam_integration: Post-processing for DROID-SLAM trajectories
- lego_loam_integration: Post-processing for LeGO-LOAM trajectories
- lidar_floor_tracker: LiDAR-based floor detection using ground plane
- place_recognition: Foundation model VPR (MixVPR, SALAD, AnyLoc, CricaVPR)
- geometric_verification: LightGlue/SuperGlue/LoFTR feature matching

Author: Wade Williams
Course: EECE5554 Robotic Sensing and Navigation
"""

from .floor_detector import IMUFloorDetector, ElevatorEvent, load_imu_from_bag
from .loop_closure_gate import (
    SemanticLoopClosureGate,
    LoopClosureCandidate,
    ContextualPriorFactor
)
from .semantic_gating_pipeline import SemanticGatingPipeline
from .orb_slam3_integration import ORBSlam3SemanticIntegration
from .droid_slam_integration import DroidSlamSemanticIntegration
from .lego_loam_integration import LegoLoamSemanticIntegration
from .lidar_floor_tracker import LiDARFloorTracker, MultiModalFloorDetector, FloorEstimate
from .place_recognition import (
    MixVPR,
    SALAD,
    AnyLoc,
    CricaVPR,
    SemanticPlaceRecognition,
    PlaceMatch,
    PlaceDescriptor
)
from .geometric_verification import (
    LightGlue,
    SuperGlue,
    LoFTR,
    GeometricVerifier,
    SemanticGeometricVerifier,
    MatchResult
)

__all__ = [
    # Floor detection
    'IMUFloorDetector',
    'ElevatorEvent',
    'load_imu_from_bag',
    'LiDARFloorTracker',
    'MultiModalFloorDetector',
    'FloorEstimate',
    # Loop closure gating
    'SemanticLoopClosureGate',
    'LoopClosureCandidate',
    'ContextualPriorFactor',
    # Pipeline
    'SemanticGatingPipeline',
    # Algorithm integrations
    'ORBSlam3SemanticIntegration',
    'DroidSlamSemanticIntegration',
    'LegoLoamSemanticIntegration',
    # Place recognition
    'MixVPR',
    'SALAD',
    'AnyLoc',
    'CricaVPR',
    'SemanticPlaceRecognition',
    'PlaceMatch',
    'PlaceDescriptor',
    # Geometric verification
    'LightGlue',
    'SuperGlue',
    'LoFTR',
    'GeometricVerifier',
    'SemanticGeometricVerifier',
    'MatchResult',
]

__version__ = '1.4.0'
