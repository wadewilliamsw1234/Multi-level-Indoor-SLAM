"""
Semantic Loop Closure Gating

Filters loop closure candidates based on floor labels to prevent
perceptual aliasing errors in multi-floor environments.

Author: Wade Williams
Course: EECE5554 Robotic Sensing and Navigation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class LoopClosureCandidate:
    """Represents a potential loop closure"""
    query_idx: int          # Index of query keyframe
    match_idx: int          # Index of matched keyframe
    similarity_score: float # Visual similarity (e.g., DBoW2 score)
    query_floor: int        # Floor label of query
    match_floor: int        # Floor label of match
    is_valid: bool = True   # Whether it passes gating
    rejection_reason: str = ""


class SemanticLoopClosureGate:
    """
    Gate loop closure candidates based on semantic constraints.
    
    Primary constraint: Floor consistency
    - Reject candidates where query and match are on different floors
    - This prevents perceptual aliasing in buildings with similar floor layouts
    
    Future extensions:
    - Room type consistency (office vs corridor vs lobby)
    - Object presence verification
    - Temporal consistency (time since last visit)
    """
    
    def __init__(self, 
                 floor_labels: np.ndarray,
                 strict_mode: bool = True):
        """
        Args:
            floor_labels: Array of floor labels for each keyframe
            strict_mode: If True, reject all cross-floor candidates
                        If False, only reject if floors differ by > 1
        """
        self.floor_labels = floor_labels
        self.strict_mode = strict_mode
        self.stats = {
            'total_candidates': 0,
            'accepted': 0,
            'rejected_cross_floor': 0,
            'rejected_other': 0
        }
    
    def gate_candidate(self, 
                       query_idx: int, 
                       match_idx: int,
                       similarity_score: float = 0.0) -> LoopClosureCandidate:
        """
        Check if a single loop closure candidate should be accepted.
        
        Args:
            query_idx: Index of query keyframe
            match_idx: Index of candidate match keyframe
            similarity_score: Visual similarity score
            
        Returns:
            LoopClosureCandidate with is_valid set appropriately
        """
        query_floor = self.floor_labels[query_idx]
        match_floor = self.floor_labels[match_idx]
        
        candidate = LoopClosureCandidate(
            query_idx=query_idx,
            match_idx=match_idx,
            similarity_score=similarity_score,
            query_floor=query_floor,
            match_floor=match_floor
        )
        
        self.stats['total_candidates'] += 1
        
        # Floor consistency check
        floor_diff = abs(query_floor - match_floor)
        
        if self.strict_mode and floor_diff > 0:
            candidate.is_valid = False
            candidate.rejection_reason = f"Cross-floor: {query_floor} vs {match_floor}"
            self.stats['rejected_cross_floor'] += 1
        elif not self.strict_mode and floor_diff > 1:
            candidate.is_valid = False
            candidate.rejection_reason = f"Floor diff > 1: {query_floor} vs {match_floor}"
            self.stats['rejected_cross_floor'] += 1
        else:
            candidate.is_valid = True
            self.stats['accepted'] += 1
        
        return candidate
    
    def gate_candidates(self, 
                        candidates: List[Tuple[int, int, float]]) -> Tuple[List, List]:
        """
        Filter a batch of loop closure candidates.
        
        Args:
            candidates: List of (query_idx, match_idx, score) tuples
            
        Returns:
            Tuple of (valid_candidates, rejected_candidates)
        """
        valid = []
        rejected = []
        
        for query_idx, match_idx, score in candidates:
            result = self.gate_candidate(query_idx, match_idx, score)
            if result.is_valid:
                valid.append(result)
            else:
                rejected.append(result)
        
        return valid, rejected
    
    def get_stats(self) -> Dict:
        """Return gating statistics"""
        total = self.stats['total_candidates']
        if total > 0:
            self.stats['acceptance_rate'] = self.stats['accepted'] / total
            self.stats['rejection_rate'] = 1 - self.stats['acceptance_rate']
        return self.stats
    
    def print_summary(self):
        """Print summary of gating results"""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("LOOP CLOSURE GATING SUMMARY")
        print("=" * 50)
        print(f"Total candidates:      {stats['total_candidates']}")
        print(f"Accepted:              {stats['accepted']}")
        print(f"Rejected (cross-floor): {stats['rejected_cross_floor']}")
        if stats['total_candidates'] > 0:
            print(f"Acceptance rate:       {stats['acceptance_rate']:.1%}")
            print(f"Perceptual aliasing prevented: {stats['rejected_cross_floor']}")
        print("=" * 50)


class ContextualPriorFactor:
    """
    Creates factor graph constraints based on contextual priors.
    
    For use with GTSAM or similar factor graph libraries.
    """
    
    def __init__(self, floor_labels: np.ndarray):
        self.floor_labels = floor_labels
    
    def create_floor_constraint(self, 
                                 pose_idx: int,
                                 floor_height: float = 3.0) -> Dict:
        """
        Create a soft constraint that poses on the same floor
        should have similar z-values.
        
        Args:
            pose_idx: Index of the pose
            floor_height: Assumed height per floor (meters)
            
        Returns:
            Dict describing the factor for GTSAM
        """
        floor = self.floor_labels[pose_idx]
        expected_z = floor * floor_height
        
        return {
            'type': 'floor_prior',
            'pose_idx': pose_idx,
            'floor': floor,
            'expected_z': expected_z,
            'noise_model': 'diagonal',
            'sigma_z': 0.5  # Allow 0.5m variance within floor
        }
    
    def create_elevator_transition_factor(self,
                                          pose_before: int,
                                          pose_after: int,
                                          direction: str,
                                          floor_height: float = 3.0) -> Dict:
        """
        Create a factor constraining the z-change during elevator transit.
        
        Args:
            pose_before: Pose index before elevator
            pose_after: Pose index after elevator
            direction: 'up' or 'down'
            floor_height: Expected height change
            
        Returns:
            Dict describing the between factor
        """
        expected_dz = floor_height if direction == 'up' else -floor_height
        
        return {
            'type': 'elevator_transition',
            'pose_before': pose_before,
            'pose_after': pose_after,
            'expected_dz': expected_dz,
            'noise_model': 'diagonal',
            'sigma_dz': 0.3  # Elevator height uncertainty
        }


def integrate_with_orbslam3(floor_labels: np.ndarray,
                            keyframe_times: np.ndarray) -> str:
    """
    Generate C++ code snippet for integrating with ORB-SLAM3's LoopClosing.
    
    This would be added to ORB-SLAM3's src/LoopClosing.cc
    """
    code = '''
// Add to LoopClosing.cc - DetectLoop() function
// After DBoW2 candidate retrieval, before geometric verification

bool LoopClosing::CheckFloorConsistency(KeyFrame* pKF, KeyFrame* pKFcandidate)
{
    // Get floor labels (stored in KeyFrame during tracking)
    int queryFloor = pKF->mnFloorLabel;
    int matchFloor = pKFcandidate->mnFloorLabel;
    
    // Strict mode: reject any cross-floor candidates
    if (queryFloor != matchFloor)
    {
        // Log rejected candidate for analysis
        VLOG(1) << "Loop closure rejected: Floor " << queryFloor 
                << " vs Floor " << matchFloor;
        return false;
    }
    
    return true;
}

// Modify DetectLoop() to call this before ComputeSim3()
vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

// Filter by floor consistency
vector<KeyFrame*> vpValidCandidates;
for(KeyFrame* pKF : vpCandidateKFs)
{
    if(CheckFloorConsistency(mpCurrentKF, pKF))
        vpValidCandidates.push_back(pKF);
}

// Continue with geometric verification on filtered candidates
'''
    return code


if __name__ == '__main__':
    print("Semantic Loop Closure Gating - Demo")
    print("=" * 50)
    
    # Simulate floor labels for ISEC sequence
    # 5th floor (0-5000), elevator, 1st floor (5000-7000), etc.
    n_keyframes = 10000
    floor_labels = np.zeros(n_keyframes, dtype=int)
    floor_labels[0:5000] = 5      # 5th floor
    floor_labels[5000:7000] = 1   # 1st floor
    floor_labels[7000:8500] = 4   # 4th floor
    floor_labels[8500:10000] = 2  # 2nd floor
    
    # Create gating object
    gate = SemanticLoopClosureGate(floor_labels, strict_mode=True)
    
    # Simulate loop closure candidates (some valid, some cross-floor)
    candidates = [
        (100, 4500, 0.85),    # Same floor (5th) - should accept
        (200, 5500, 0.92),    # Cross-floor (5th vs 1st) - should reject!
        (5100, 6800, 0.88),   # Same floor (1st) - should accept
        (300, 7200, 0.91),    # Cross-floor (5th vs 4th) - should reject!
        (7100, 8200, 0.87),   # Same floor (4th) - should accept
        (400, 9000, 0.93),    # Cross-floor (5th vs 2nd) - should reject!
        (4000, 4200, 0.80),   # Same floor (5th) - should accept
    ]
    
    valid, rejected = gate.gate_candidates(candidates)
    
    print("\nProcessed candidates:")
    print("-" * 50)
    for c in valid:
        print(f"  ✓ VALID: ({c.query_idx}, {c.match_idx}) "
              f"Floor {c.query_floor} <-> {c.match_floor}")
    for c in rejected:
        print(f"  ✗ REJECT: ({c.query_idx}, {c.match_idx}) "
              f"{c.rejection_reason}")
    
    gate.print_summary()
    
    print("\n" + "=" * 50)
    print("ORB-SLAM3 Integration Code:")
    print("=" * 50)
    print(integrate_with_orbslam3(floor_labels, None)[:500] + "...")
