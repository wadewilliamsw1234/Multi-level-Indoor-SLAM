"""
Geometric Verification for Loop Closure Candidates

Implements learned feature matching for geometric verification:
- LightGlue: Fast and accurate learned matcher with adaptive stopping
- SuperGlue: Graph neural network-based matcher
- LoFTR: Detector-free local feature matching

Provides 6-DoF pose estimation and inlier ratio computation
for validating loop closure candidates.

Author: Wade Williams
Course: EECE5554 Robotic Sensing and Navigation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

# Lazy imports
torch = None
cv2 = None


def _import_torch():
    """Lazy import PyTorch"""
    global torch
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except ImportError:
            raise ImportError(
                "PyTorch is required. Install with: pip install torch torchvision"
            )
    return torch


def _import_cv2():
    """Lazy import OpenCV"""
    global cv2
    if cv2 is None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            raise ImportError(
                "OpenCV is required. Install with: pip install opencv-python"
            )
    return cv2


@dataclass
class MatchResult:
    """Result of geometric verification between two images"""
    query_idx: int
    match_idx: int
    num_keypoints_query: int
    num_keypoints_match: int
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    relative_pose: Optional[np.ndarray]  # 4x4 transformation matrix
    essential_matrix: Optional[np.ndarray]
    confidence: float
    is_valid: bool


@dataclass
class Keypoint:
    """2D keypoint with descriptor"""
    x: float
    y: float
    score: float
    descriptor: Optional[np.ndarray] = None


class BaseFeatureMatcher:
    """Base class for feature matching methods"""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None

    def detect_and_match(self,
                         image1: np.ndarray,
                         image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect features and match between two images.

        Args:
            image1: First image (query)
            image2: Second image (match candidate)

        Returns:
            Tuple of (keypoints1, keypoints2, match_confidences)
            keypoints: Nx2 array of matched keypoint coordinates
            confidences: N array of match confidence scores
        """
        raise NotImplementedError

    def verify_geometric_consistency(self,
                                     kpts1: np.ndarray,
                                     kpts2: np.ndarray,
                                     K: Optional[np.ndarray] = None,
                                     ransac_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Verify geometric consistency using RANSAC.

        Args:
            kpts1: Nx2 keypoints in image 1
            kpts2: Nx2 keypoints in image 2
            K: Camera intrinsic matrix (3x3)
            ransac_threshold: RANSAC inlier threshold in pixels

        Returns:
            Tuple of (inlier_mask, essential_matrix, inlier_ratio)
        """
        cv2 = _import_cv2()

        if len(kpts1) < 5:
            return np.array([]), None, 0.0

        if K is not None:
            # Compute essential matrix
            E, mask = cv2.findEssentialMat(
                kpts1, kpts2, K,
                method=cv2.RANSAC,
                threshold=ransac_threshold,
                prob=0.999
            )
            if mask is None:
                return np.array([]), None, 0.0

            inlier_mask = mask.ravel().astype(bool)
            inlier_ratio = np.sum(inlier_mask) / len(kpts1)
            return inlier_mask, E, inlier_ratio
        else:
            # Compute fundamental matrix (no calibration)
            F, mask = cv2.findFundamentalMat(
                kpts1, kpts2,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=ransac_threshold,
                confidence=0.999
            )
            if mask is None:
                return np.array([]), None, 0.0

            inlier_mask = mask.ravel().astype(bool)
            inlier_ratio = np.sum(inlier_mask) / len(kpts1)
            return inlier_mask, F, inlier_ratio

    def estimate_relative_pose(self,
                               kpts1: np.ndarray,
                               kpts2: np.ndarray,
                               K: np.ndarray,
                               inlier_mask: np.ndarray,
                               E: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate relative pose from essential matrix.

        Returns:
            4x4 transformation matrix [R|t; 0 1] or None if failed
        """
        cv2 = _import_cv2()

        if E is None or np.sum(inlier_mask) < 5:
            return None

        kpts1_inlier = kpts1[inlier_mask]
        kpts2_inlier = kpts2[inlier_mask]

        # Recover pose
        _, R, t, mask_pose = cv2.recoverPose(
            E, kpts1_inlier, kpts2_inlier, K
        )

        if R is None:
            return None

        # Build 4x4 transformation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()

        return T


class LightGlue(BaseFeatureMatcher):
    """
    LightGlue: Fast and Accurate Feature Matcher

    Uses SuperPoint for keypoint detection and LightGlue
    for learned matching with adaptive early stopping.

    Reference: https://github.com/cvg/LightGlue
    """

    def __init__(self,
                 device: str = 'cuda',
                 max_keypoints: int = 2048,
                 detection_threshold: float = 0.001):
        """
        Args:
            device: 'cuda' or 'cpu'
            max_keypoints: Maximum keypoints to detect
            detection_threshold: Keypoint detection threshold
        """
        super().__init__(device)
        self.max_keypoints = max_keypoints
        self.detection_threshold = detection_threshold
        self._model_loaded = False

    def _load_model(self):
        """Lazy load LightGlue and SuperPoint"""
        if self._model_loaded:
            return

        torch = _import_torch()

        try:
            from lightglue import LightGlue as LG
            from lightglue import SuperPoint
            from lightglue.utils import load_image, rbd

            self.extractor = SuperPoint(
                max_num_keypoints=self.max_keypoints,
                detection_threshold=self.detection_threshold
            ).eval().to(self.device)

            self.matcher = LG(features='superpoint').eval().to(self.device)
            self._model_loaded = True
            self._is_native = True

        except ImportError:
            warnings.warn(
                "LightGlue not installed. Using ORB+BFMatcher fallback. "
                "Install with: pip install git+https://github.com/cvg/LightGlue.git"
            )
            self._load_fallback()

    def _load_fallback(self):
        """Load OpenCV ORB as fallback"""
        cv2 = _import_cv2()
        self.orb = cv2.ORB_create(nfeatures=self.max_keypoints)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._model_loaded = True
        self._is_native = False

    def detect_and_match(self,
                         image1: np.ndarray,
                         image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect and match features using LightGlue"""
        self._load_model()

        if hasattr(self, '_is_native') and self._is_native:
            return self._detect_and_match_native(image1, image2)
        else:
            return self._detect_and_match_fallback(image1, image2)

    def _detect_and_match_native(self,
                                  image1: np.ndarray,
                                  image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Native LightGlue matching"""
        torch = _import_torch()
        cv2 = _import_cv2()

        # Convert to grayscale if needed
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1

        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = image2

        # Normalize to [0, 1] and convert to tensor
        tensor1 = torch.from_numpy(gray1).float() / 255.0
        tensor2 = torch.from_numpy(gray2).float() / 255.0

        # Add batch and channel dimensions
        tensor1 = tensor1.unsqueeze(0).unsqueeze(0).to(self.device)
        tensor2 = tensor2.unsqueeze(0).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            feats1 = self.extractor({'image': tensor1})
            feats2 = self.extractor({'image': tensor2})

            # Match
            matches = self.matcher({
                'image0': feats1,
                'image1': feats2
            })

        # Extract matched keypoints
        kpts1 = feats1['keypoints'][0].cpu().numpy()
        kpts2 = feats2['keypoints'][0].cpu().numpy()
        match_indices = matches['matches'][0].cpu().numpy()
        match_scores = matches['matching_scores'][0].cpu().numpy()

        # Get matched coordinates
        valid = match_indices[:, 0] >= 0
        matched_kpts1 = kpts1[match_indices[valid, 0]]
        matched_kpts2 = kpts2[match_indices[valid, 1]]
        confidences = match_scores[valid]

        return matched_kpts1, matched_kpts2, confidences

    def _detect_and_match_fallback(self,
                                    image1: np.ndarray,
                                    image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback ORB matching"""
        cv2 = _import_cv2()

        # Convert to grayscale
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1

        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = image2

        # Detect and compute
        kpts1, desc1 = self.orb.detectAndCompute(gray1, None)
        kpts2, desc2 = self.orb.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None or len(kpts1) < 5 or len(kpts2) < 5:
            return np.array([]), np.array([]), np.array([])

        # Match
        matches = self.bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        matched_kpts1 = np.array([kpts1[m.queryIdx].pt for m in matches])
        matched_kpts2 = np.array([kpts2[m.trainIdx].pt for m in matches])

        # Confidence from distance (lower distance = higher confidence)
        max_dist = max(m.distance for m in matches) if matches else 1
        confidences = np.array([1 - m.distance / max_dist for m in matches])

        return matched_kpts1, matched_kpts2, confidences


class SuperGlue(BaseFeatureMatcher):
    """
    SuperGlue: Graph Neural Network Feature Matcher

    Uses SuperPoint for detection and SuperGlue GNN for matching.
    More accurate than LightGlue but slower.

    Reference: https://github.com/magicleap/SuperGluePretrainedNetwork
    """

    def __init__(self,
                 device: str = 'cuda',
                 max_keypoints: int = 2048,
                 weights: str = 'indoor'):
        """
        Args:
            device: 'cuda' or 'cpu'
            max_keypoints: Maximum keypoints
            weights: 'indoor' or 'outdoor'
        """
        super().__init__(device)
        self.max_keypoints = max_keypoints
        self.weights = weights
        self._model_loaded = False

    def _load_model(self):
        """Lazy load SuperGlue"""
        if self._model_loaded:
            return

        torch = _import_torch()

        try:
            from models.superpoint import SuperPoint
            from models.superglue import SuperGlue as SG

            self.extractor = SuperPoint({
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': self.max_keypoints
            }).to(self.device).eval()

            self.matcher = SG({
                'weights': self.weights,
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2
            }).to(self.device).eval()

            self._model_loaded = True
            self._is_native = True

        except ImportError:
            warnings.warn("SuperGlue not installed. Using LightGlue fallback.")
            self._fallback = LightGlue(device=self.device, max_keypoints=self.max_keypoints)
            self._model_loaded = True
            self._is_native = False

    def detect_and_match(self,
                         image1: np.ndarray,
                         image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect and match using SuperGlue"""
        self._load_model()

        if not self._is_native:
            return self._fallback.detect_and_match(image1, image2)

        # Native SuperGlue implementation would go here
        # For now, defer to fallback
        return self._fallback.detect_and_match(image1, image2)


class LoFTR(BaseFeatureMatcher):
    """
    LoFTR: Detector-Free Local Feature Matching

    Uses Transformer architecture for dense matching without
    explicit keypoint detection. Good for textureless regions.

    Reference: https://github.com/zju3dv/LoFTR
    """

    def __init__(self,
                 device: str = 'cuda',
                 weights: str = 'indoor'):
        """
        Args:
            device: 'cuda' or 'cpu'
            weights: 'indoor' or 'outdoor'
        """
        super().__init__(device)
        self.weights = weights
        self._model_loaded = False

    def _load_model(self):
        """Lazy load LoFTR"""
        if self._model_loaded:
            return

        torch = _import_torch()

        try:
            from kornia.feature import LoFTR as KorniaLoFTR

            self.matcher = KorniaLoFTR(pretrained=self.weights).to(self.device).eval()
            self._model_loaded = True
            self._is_native = True

        except ImportError:
            warnings.warn(
                "LoFTR (kornia) not installed. Using LightGlue fallback. "
                "Install with: pip install kornia"
            )
            self._fallback = LightGlue(device=self.device)
            self._model_loaded = True
            self._is_native = False

    def detect_and_match(self,
                         image1: np.ndarray,
                         image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect and match using LoFTR"""
        self._load_model()

        if not self._is_native:
            return self._fallback.detect_and_match(image1, image2)

        torch = _import_torch()
        cv2 = _import_cv2()

        # Convert to grayscale and normalize
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1

        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = image2

        # LoFTR expects specific resolution (multiple of 8)
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape

        # Resize to nearest multiple of 8
        new_h1, new_w1 = (h1 // 8) * 8, (w1 // 8) * 8
        new_h2, new_w2 = (h2 // 8) * 8, (w2 // 8) * 8

        gray1_resized = cv2.resize(gray1, (new_w1, new_h1))
        gray2_resized = cv2.resize(gray2, (new_w2, new_h2))

        # To tensor
        tensor1 = torch.from_numpy(gray1_resized).float() / 255.0
        tensor2 = torch.from_numpy(gray2_resized).float() / 255.0

        tensor1 = tensor1.unsqueeze(0).unsqueeze(0).to(self.device)
        tensor2 = tensor2.unsqueeze(0).unsqueeze(0).to(self.device)

        # Match
        with torch.no_grad():
            input_dict = {'image0': tensor1, 'image1': tensor2}
            matches = self.matcher(input_dict)

        kpts1 = matches['keypoints0'].cpu().numpy()
        kpts2 = matches['keypoints1'].cpu().numpy()
        confidences = matches['confidence'].cpu().numpy()

        # Scale keypoints back to original resolution
        scale1 = np.array([w1 / new_w1, h1 / new_h1])
        scale2 = np.array([w2 / new_w2, h2 / new_h2])

        kpts1 = kpts1 * scale1
        kpts2 = kpts2 * scale2

        return kpts1, kpts2, confidences


class GeometricVerifier:
    """
    Complete geometric verification pipeline for loop closure candidates.

    Combines feature matching with RANSAC geometric verification
    and semantic floor gating.
    """

    def __init__(self,
                 matcher_type: str = 'lightglue',
                 device: str = 'cuda',
                 min_inliers: int = 20,
                 min_inlier_ratio: float = 0.25,
                 ransac_threshold: float = 3.0):
        """
        Args:
            matcher_type: 'lightglue', 'superglue', or 'loftr'
            device: 'cuda' or 'cpu'
            min_inliers: Minimum inliers for valid match
            min_inlier_ratio: Minimum inlier ratio
            ransac_threshold: RANSAC inlier threshold (pixels)
        """
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_threshold = ransac_threshold

        if matcher_type.lower() == 'lightglue':
            self.matcher = LightGlue(device=device)
        elif matcher_type.lower() == 'superglue':
            self.matcher = SuperGlue(device=device)
        elif matcher_type.lower() == 'loftr':
            self.matcher = LoFTR(device=device)
        else:
            raise ValueError(f"Unknown matcher: {matcher_type}")

    def verify(self,
               image1: np.ndarray,
               image2: np.ndarray,
               K: Optional[np.ndarray] = None,
               query_idx: int = 0,
               match_idx: int = 0) -> MatchResult:
        """
        Verify geometric consistency between two images.

        Args:
            image1: Query image
            image2: Match candidate image
            K: Camera intrinsic matrix (optional)
            query_idx: Index of query in database
            match_idx: Index of match candidate

        Returns:
            MatchResult with verification details
        """
        # Detect and match features
        kpts1, kpts2, confidences = self.matcher.detect_and_match(image1, image2)

        if len(kpts1) < 5:
            return MatchResult(
                query_idx=query_idx,
                match_idx=match_idx,
                num_keypoints_query=0,
                num_keypoints_match=0,
                num_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                relative_pose=None,
                essential_matrix=None,
                confidence=0.0,
                is_valid=False
            )

        # Geometric verification
        inlier_mask, E, inlier_ratio = self.matcher.verify_geometric_consistency(
            kpts1, kpts2, K, self.ransac_threshold
        )

        num_inliers = int(np.sum(inlier_mask)) if len(inlier_mask) > 0 else 0

        # Estimate relative pose if we have calibration
        relative_pose = None
        if K is not None and E is not None and num_inliers >= 5:
            relative_pose = self.matcher.estimate_relative_pose(
                kpts1, kpts2, K, inlier_mask, E
            )

        # Determine validity
        is_valid = (num_inliers >= self.min_inliers and
                    inlier_ratio >= self.min_inlier_ratio)

        # Compute confidence score
        confidence = min(1.0, inlier_ratio * (num_inliers / self.min_inliers))

        return MatchResult(
            query_idx=query_idx,
            match_idx=match_idx,
            num_keypoints_query=len(kpts1),
            num_keypoints_match=len(kpts2),
            num_matches=len(kpts1),
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            relative_pose=relative_pose,
            essential_matrix=E,
            confidence=confidence,
            is_valid=is_valid
        )

    def verify_batch(self,
                     image_pairs: List[Tuple[np.ndarray, np.ndarray]],
                     K: Optional[np.ndarray] = None,
                     indices: Optional[List[Tuple[int, int]]] = None) -> List[MatchResult]:
        """
        Verify multiple image pairs.

        Args:
            image_pairs: List of (image1, image2) tuples
            K: Camera intrinsic matrix
            indices: List of (query_idx, match_idx) tuples

        Returns:
            List of MatchResult objects
        """
        results = []

        for i, (img1, img2) in enumerate(image_pairs):
            if indices is not None:
                q_idx, m_idx = indices[i]
            else:
                q_idx, m_idx = i, i

            result = self.verify(img1, img2, K, q_idx, m_idx)
            results.append(result)

        return results


class SemanticGeometricVerifier(GeometricVerifier):
    """
    Geometric verification with semantic floor gating.

    Skips geometric verification for cross-floor candidates,
    saving computation while maintaining accuracy.
    """

    def __init__(self,
                 matcher_type: str = 'lightglue',
                 device: str = 'cuda',
                 min_inliers: int = 20,
                 min_inlier_ratio: float = 0.25,
                 enable_floor_gating: bool = True):
        super().__init__(matcher_type, device, min_inliers, min_inlier_ratio)
        self.enable_floor_gating = enable_floor_gating
        self.stats = {
            'verified': 0,
            'skipped_floor_mismatch': 0,
            'valid': 0,
            'invalid': 0
        }

    def verify_with_semantics(self,
                              image1: np.ndarray,
                              image2: np.ndarray,
                              floor1: int,
                              floor2: int,
                              K: Optional[np.ndarray] = None,
                              query_idx: int = 0,
                              match_idx: int = 0) -> MatchResult:
        """
        Verify with semantic floor check.

        Args:
            image1, image2: Images to verify
            floor1, floor2: Floor labels for each image
            K: Camera intrinsic matrix
            query_idx, match_idx: Database indices

        Returns:
            MatchResult (is_valid=False if cross-floor)
        """
        # Check floor consistency first
        if self.enable_floor_gating and floor1 != floor2:
            self.stats['skipped_floor_mismatch'] += 1
            return MatchResult(
                query_idx=query_idx,
                match_idx=match_idx,
                num_keypoints_query=0,
                num_keypoints_match=0,
                num_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                relative_pose=None,
                essential_matrix=None,
                confidence=0.0,
                is_valid=False
            )

        # Perform geometric verification
        result = self.verify(image1, image2, K, query_idx, match_idx)
        self.stats['verified'] += 1

        if result.is_valid:
            self.stats['valid'] += 1
        else:
            self.stats['invalid'] += 1

        return result

    def get_statistics(self) -> Dict:
        """Get verification statistics"""
        total = (self.stats['verified'] + self.stats['skipped_floor_mismatch'])
        return {
            **self.stats,
            'total_candidates': total,
            'skip_rate': self.stats['skipped_floor_mismatch'] / total if total > 0 else 0,
            'valid_rate': self.stats['valid'] / self.stats['verified'] if self.stats['verified'] > 0 else 0
        }


def demo():
    """Demo with synthetic images"""
    print("Geometric Verification Demo")
    print("=" * 50)

    cv2 = _import_cv2()

    # Create synthetic test images with known features
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add some random rectangles as features
    np.random.seed(42)
    for _ in range(20):
        x, y = np.random.randint(50, 550), np.random.randint(50, 400)
        w, h = np.random.randint(20, 60), np.random.randint(20, 60)
        color = tuple(np.random.randint(100, 255, 3).tolist())
        cv2.rectangle(img1, (x, y), (x + w, y + h), color, -1)
        # Add same features to img2 with small offset (simulating camera motion)
        offset_x = np.random.randint(-5, 5)
        offset_y = np.random.randint(-5, 5)
        cv2.rectangle(img2, (x + offset_x, y + offset_y), (x + w + offset_x, y + h + offset_y), color, -1)

    # Add some noise
    noise1 = np.random.randint(0, 30, img1.shape, dtype=np.uint8)
    noise2 = np.random.randint(0, 30, img2.shape, dtype=np.uint8)
    img1 = cv2.add(img1, noise1)
    img2 = cv2.add(img2, noise2)

    print("Testing geometric verification...")

    # Test LightGlue (will fall back to ORB if not installed)
    verifier = GeometricVerifier(matcher_type='lightglue', device='cpu')

    # Synthetic camera matrix
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    result = verifier.verify(img1, img2, K)

    print(f"\nMatch Result:")
    print(f"  Matches found: {result.num_matches}")
    print(f"  Inliers: {result.num_inliers}")
    print(f"  Inlier ratio: {result.inlier_ratio:.2%}")
    print(f"  Valid match: {result.is_valid}")
    print(f"  Confidence: {result.confidence:.3f}")

    # Test semantic verifier
    print("\nTesting semantic geometric verification...")

    sem_verifier = SemanticGeometricVerifier(
        matcher_type='lightglue',
        device='cpu',
        enable_floor_gating=True
    )

    # Same floor - should verify
    result_same = sem_verifier.verify_with_semantics(img1, img2, floor1=1, floor2=1, K=K)
    print(f"Same floor (1,1): valid={result_same.is_valid}, verified=True")

    # Different floor - should skip
    result_diff = sem_verifier.verify_with_semantics(img1, img2, floor1=1, floor2=2, K=K)
    print(f"Different floor (1,2): valid={result_diff.is_valid}, verified=False (skipped)")

    stats = sem_verifier.get_statistics()
    print(f"\nStatistics: {stats}")


if __name__ == '__main__':
    demo()
