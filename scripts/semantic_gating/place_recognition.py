"""
Foundation Model Visual Place Recognition

Implements state-of-the-art Visual Place Recognition (VPR) using foundation models:
- MixVPR: Aggregated feature mixing for robust place recognition
- SALAD: Sinkhorn Algorithm for Locally-Aware Distributions
- AnyLoc: Universal visual place recognition via foundation models
- CricaVPR: Cross-image correlation-aware VPR for perceptual aliasing robustness

These methods provide semantically-aware place matching that is more robust
to viewpoint and appearance changes than classical bag-of-words approaches.

CricaVPR (CVPR 2024) explicitly addresses perceptual aliasing by learning
viewpoint/condition-invariant features that distinguish similar-but-different places.

Author: Wade Williams
Course: EECE5554 Robotic Sensing and Navigation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

# Lazy imports to avoid hard dependencies
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
                "PyTorch is required for place recognition. "
                "Install with: pip install torch torchvision"
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
                "OpenCV is required for image loading. "
                "Install with: pip install opencv-python"
            )
    return cv2


@dataclass
class PlaceMatch:
    """Represents a place recognition match between two images"""
    query_idx: int
    match_idx: int
    similarity: float
    query_timestamp: Optional[float] = None
    match_timestamp: Optional[float] = None
    is_valid: bool = True  # Can be invalidated by semantic gating


@dataclass
class PlaceDescriptor:
    """Global place descriptor for an image"""
    timestamp: float
    descriptor: np.ndarray
    image_path: Optional[str] = None
    floor_label: Optional[int] = None


class BasePlaceRecognition:
    """Base class for VPR methods"""

    def __init__(self,
                 descriptor_dim: int = 4096,
                 device: str = 'cuda'):
        """
        Args:
            descriptor_dim: Output descriptor dimensionality
            device: 'cuda' or 'cpu'
        """
        self.descriptor_dim = descriptor_dim
        self.device = device
        self.model = None
        self.descriptors: List[PlaceDescriptor] = []

    def extract_descriptor(self, image: np.ndarray) -> np.ndarray:
        """Extract global descriptor from image"""
        raise NotImplementedError

    def add_image(self,
                  image: np.ndarray,
                  timestamp: float,
                  floor_label: Optional[int] = None,
                  image_path: Optional[str] = None) -> PlaceDescriptor:
        """Add image to database"""
        descriptor = self.extract_descriptor(image)
        place_desc = PlaceDescriptor(
            timestamp=timestamp,
            descriptor=descriptor,
            image_path=image_path,
            floor_label=floor_label
        )
        self.descriptors.append(place_desc)
        return place_desc

    def query(self,
              image: np.ndarray,
              timestamp: Optional[float] = None,
              k: int = 5,
              min_time_gap: float = 10.0) -> List[PlaceMatch]:
        """
        Query database for similar places.

        Args:
            image: Query image
            timestamp: Query timestamp (for temporal filtering)
            k: Number of top matches to return
            min_time_gap: Minimum time between query and match

        Returns:
            List of PlaceMatch objects
        """
        if len(self.descriptors) == 0:
            return []

        query_desc = self.extract_descriptor(image)

        # Compute similarities
        db_descriptors = np.vstack([d.descriptor for d in self.descriptors])
        similarities = self._compute_similarity(query_desc, db_descriptors)

        # Filter by time gap if timestamp provided
        if timestamp is not None:
            for i, desc in enumerate(self.descriptors):
                if abs(desc.timestamp - timestamp) < min_time_gap:
                    similarities[i] = -np.inf

        # Get top-k matches
        top_k_idx = np.argsort(similarities)[::-1][:k]

        matches = []
        for idx in top_k_idx:
            if similarities[idx] > -np.inf:
                matches.append(PlaceMatch(
                    query_idx=len(self.descriptors),  # Query would be next index
                    match_idx=idx,
                    similarity=float(similarities[idx]),
                    query_timestamp=timestamp,
                    match_timestamp=self.descriptors[idx].timestamp
                ))

        return matches

    def _compute_similarity(self,
                            query: np.ndarray,
                            database: np.ndarray) -> np.ndarray:
        """Compute cosine similarity"""
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        db_norms = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
        return np.dot(db_norms, query_norm)

    def build_descriptor_matrix(self) -> np.ndarray:
        """Build matrix of all descriptors for batch operations"""
        if len(self.descriptors) == 0:
            return np.array([])
        return np.vstack([d.descriptor for d in self.descriptors])

    def compute_all_pairwise_similarities(self) -> np.ndarray:
        """Compute NxN similarity matrix"""
        desc_matrix = self.build_descriptor_matrix()
        if len(desc_matrix) == 0:
            return np.array([])

        # Normalize
        norms = np.linalg.norm(desc_matrix, axis=1, keepdims=True)
        desc_matrix = desc_matrix / (norms + 1e-8)

        # Compute pairwise similarities
        return np.dot(desc_matrix, desc_matrix.T)


class MixVPR(BasePlaceRecognition):
    """
    MixVPR: Feature Mixing for Visual Place Recognition

    Uses a CNN backbone (ResNet/EfficientNet) with feature aggregation
    via MLP-Mixer blocks for robust global descriptors.

    Reference: https://github.com/amaralibey/MixVPR
    """

    def __init__(self,
                 backbone: str = 'resnet50',
                 descriptor_dim: int = 4096,
                 device: str = 'cuda',
                 pretrained_path: Optional[str] = None):
        """
        Args:
            backbone: CNN backbone ('resnet50', 'efficientnet_b0', etc.)
            descriptor_dim: Output descriptor dimension
            device: 'cuda' or 'cpu'
            pretrained_path: Path to pretrained MixVPR weights
        """
        super().__init__(descriptor_dim, device)
        self.backbone_name = backbone
        self.pretrained_path = pretrained_path
        self._model_loaded = False

    def _load_model(self):
        """Lazy load the model"""
        if self._model_loaded:
            return

        torch = _import_torch()

        try:
            # Try to import MixVPR
            from mixvpr import MixVPRModel
            self.model = MixVPRModel(
                backbone=self.backbone_name,
                out_dim=self.descriptor_dim
            )
            if self.pretrained_path:
                state_dict = torch.load(self.pretrained_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True

        except ImportError:
            warnings.warn(
                "MixVPR not installed. Using ResNet50 features as fallback. "
                "For full MixVPR: pip install mixvpr or clone from GitHub."
            )
            self._load_fallback_model()

    def _load_fallback_model(self):
        """Load ResNet50 as fallback feature extractor"""
        torch = _import_torch()
        import torchvision.models as models
        import torchvision.transforms as transforms

        # Use ResNet50 with global average pooling
        resnet = models.resnet50(pretrained=True)
        # Remove final FC layer
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self._model_loaded = True
        self._is_fallback = True

    def extract_descriptor(self, image: np.ndarray) -> np.ndarray:
        """Extract MixVPR descriptor from image"""
        self._load_model()
        torch = _import_torch()

        # Preprocess image
        if hasattr(self, '_is_fallback') and self._is_fallback:
            # Fallback preprocessing
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        else:
            # MixVPR preprocessing
            tensor = self._preprocess(image)

        # Extract features
        with torch.no_grad():
            features = self.model(tensor)

        # Flatten and normalize
        descriptor = features.cpu().numpy().flatten()

        # If dimension doesn't match, project or pad
        if len(descriptor) != self.descriptor_dim:
            if len(descriptor) > self.descriptor_dim:
                descriptor = descriptor[:self.descriptor_dim]
            else:
                descriptor = np.pad(descriptor, (0, self.descriptor_dim - len(descriptor)))

        return descriptor

    def _preprocess(self, image: np.ndarray) -> 'torch.Tensor':
        """Preprocess image for MixVPR"""
        torch = _import_torch()
        cv2 = _import_cv2()

        # Resize to expected input size
        image = cv2.resize(image, (320, 320))

        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # To tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)


class SALAD(BasePlaceRecognition):
    """
    SALAD: Sinkhorn Algorithm for Locally-Aware Distributions

    Uses optimal transport for aggregating local features into
    a global descriptor that preserves spatial structure.

    Reference: https://github.com/serizba/salad
    """

    def __init__(self,
                 descriptor_dim: int = 8448,
                 device: str = 'cuda',
                 pretrained_path: Optional[str] = None):
        super().__init__(descriptor_dim, device)
        self.pretrained_path = pretrained_path
        self._model_loaded = False

    def _load_model(self):
        """Lazy load SALAD model"""
        if self._model_loaded:
            return

        torch = _import_torch()

        try:
            from salad import SALAD as SALADModel
            self.model = SALADModel(out_dim=self.descriptor_dim)
            if self.pretrained_path:
                state_dict = torch.load(self.pretrained_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True

        except ImportError:
            warnings.warn(
                "SALAD not installed. Using MixVPR fallback. "
                "For SALAD: pip install salad-vpr"
            )
            # Fall back to MixVPR
            self._fallback = MixVPR(descriptor_dim=self.descriptor_dim, device=self.device)
            self._model_loaded = True
            self._is_fallback = True

    def extract_descriptor(self, image: np.ndarray) -> np.ndarray:
        """Extract SALAD descriptor"""
        self._load_model()

        if hasattr(self, '_is_fallback') and self._is_fallback:
            return self._fallback.extract_descriptor(image)

        torch = _import_torch()
        tensor = self._preprocess(image)

        with torch.no_grad():
            descriptor = self.model(tensor)

        return descriptor.cpu().numpy().flatten()

    def _preprocess(self, image: np.ndarray) -> 'torch.Tensor':
        """Preprocess for SALAD"""
        torch = _import_torch()
        cv2 = _import_cv2()

        image = cv2.resize(image, (480, 640))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)


class AnyLoc(BasePlaceRecognition):
    """
    AnyLoc: Universal Visual Place Recognition

    Uses DINOv2 foundation model features with VLAD aggregation
    for domain-agnostic place recognition.

    Reference: https://github.com/AnyLoc/AnyLoc
    """

    def __init__(self,
                 backbone: str = 'dinov2_vitb14',
                 descriptor_dim: int = 49152,
                 device: str = 'cuda',
                 num_clusters: int = 64):
        """
        Args:
            backbone: DINOv2 backbone variant
            descriptor_dim: Output dimension (depends on backbone and clusters)
            device: 'cuda' or 'cpu'
            num_clusters: Number of VLAD clusters
        """
        super().__init__(descriptor_dim, device)
        self.backbone_name = backbone
        self.num_clusters = num_clusters
        self._model_loaded = False

    def _load_model(self):
        """Lazy load DINOv2 backbone"""
        if self._model_loaded:
            return

        torch = _import_torch()

        try:
            # Load DINOv2 from torch hub
            self.model = torch.hub.load(
                'facebookresearch/dinov2',
                self.backbone_name,
                pretrained=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            # Initialize VLAD clusters (would normally be trained on data)
            self.vlad_clusters = None
            self._model_loaded = True

        except Exception as e:
            warnings.warn(f"Failed to load DINOv2: {e}. Using MixVPR fallback.")
            self._fallback = MixVPR(descriptor_dim=4096, device=self.device)
            self._model_loaded = True
            self._is_fallback = True

    def extract_descriptor(self, image: np.ndarray) -> np.ndarray:
        """Extract AnyLoc descriptor using DINOv2 + VLAD"""
        self._load_model()

        if hasattr(self, '_is_fallback') and self._is_fallback:
            return self._fallback.extract_descriptor(image)

        torch = _import_torch()
        tensor = self._preprocess(image)

        with torch.no_grad():
            # Get intermediate features
            features = self.model.get_intermediate_layers(tensor, n=1)[0]

            # Remove CLS token
            patch_features = features[:, 1:, :]  # [B, N, D]

            # Simple global average pooling (full VLAD would use clustering)
            descriptor = patch_features.mean(dim=1)

        return descriptor.cpu().numpy().flatten()

    def _preprocess(self, image: np.ndarray) -> 'torch.Tensor':
        """Preprocess for DINOv2"""
        torch = _import_torch()
        cv2 = _import_cv2()

        # DINOv2 expects 518x518 for ViT-B/14
        image = cv2.resize(image, (518, 518))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)


class CricaVPR(BasePlaceRecognition):
    """
    CricaVPR: Cross-Image Correlation-Aware Visual Place Recognition

    CVPR 2024 - Explicitly addresses perceptual aliasing through cross-image
    correlation-aware representation learning. During training, it harvests
    information from other images in the batch to learn viewpoint/condition-
    invariant features that distinguish similar-but-different places.

    Key Innovation: Uses cross-attention between query and database images
    to learn discriminative features that are aware of potential confusers.

    This is the MOST RELEVANT VPR method for multi-floor SLAM because it
    directly tackles the perceptual aliasing problem.

    Reference: https://github.com/Lu-Feng/CricaVPR
    Paper: "CricaVPR: Cross-image Correlation-aware Representation Learning
           for Visual Place Recognition" (CVPR 2024)
    """

    def __init__(self,
                 backbone: str = 'dinov2_vitb14',
                 descriptor_dim: int = 10752,
                 device: str = 'cuda',
                 pretrained_path: Optional[str] = None,
                 use_reranking: bool = True):
        """
        Args:
            backbone: DINOv2 backbone variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14')
            descriptor_dim: Output descriptor dimension (depends on backbone)
            device: 'cuda' or 'cpu'
            pretrained_path: Path to pretrained CricaVPR weights
            use_reranking: Enable cross-correlation reranking for improved accuracy
        """
        super().__init__(descriptor_dim, device)
        self.backbone_name = backbone
        self.pretrained_path = pretrained_path
        self.use_reranking = use_reranking
        self._model_loaded = False

        # Cross-correlation cache for reranking
        self._feature_cache = {}

    def _load_model(self):
        """Lazy load CricaVPR model"""
        if self._model_loaded:
            return

        torch = _import_torch()

        try:
            # Try to import CricaVPR
            from cricavpr import CricaVPRModel
            self.model = CricaVPRModel(
                backbone=self.backbone_name,
                agg_config={'type': 'gem', 'p': 3}
            )
            if self.pretrained_path:
                state_dict = torch.load(self.pretrained_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            self._native_model = True

        except ImportError:
            warnings.warn(
                "CricaVPR not installed. Using DINOv2 with GeM pooling as approximation. "
                "For full CricaVPR: git clone https://github.com/Lu-Feng/CricaVPR"
            )
            self._load_dinov2_fallback()

    def _load_dinov2_fallback(self):
        """Load DINOv2 with GeM pooling as CricaVPR approximation"""
        torch = _import_torch()

        try:
            # Load DINOv2 backbone
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2',
                self.backbone_name,
                pretrained=True
            )
            self.backbone = self.backbone.to(self.device)
            self.backbone.eval()

            # Get feature dimension based on backbone
            if 'vits' in self.backbone_name:
                self.feat_dim = 384
            elif 'vitb' in self.backbone_name:
                self.feat_dim = 768
            elif 'vitl' in self.backbone_name:
                self.feat_dim = 1024
            else:
                self.feat_dim = 768

            self._model_loaded = True
            self._native_model = False

        except Exception as e:
            warnings.warn(f"Failed to load DINOv2: {e}. Using MixVPR fallback.")
            self._fallback = MixVPR(descriptor_dim=4096, device=self.device)
            self._model_loaded = True
            self._is_fallback = True

    def extract_descriptor(self, image: np.ndarray) -> np.ndarray:
        """
        Extract CricaVPR descriptor from image.

        For the full model, this extracts correlation-aware features.
        For the fallback, this uses DINOv2 + GeM pooling.
        """
        self._load_model()

        if hasattr(self, '_is_fallback') and self._is_fallback:
            return self._fallback.extract_descriptor(image)

        torch = _import_torch()
        tensor = self._preprocess(image)

        with torch.no_grad():
            if hasattr(self, '_native_model') and self._native_model:
                # Native CricaVPR extraction
                descriptor = self.model.extract_global_descriptor(tensor)
            else:
                # DINOv2 fallback with GeM pooling
                features = self.backbone.get_intermediate_layers(tensor, n=1)[0]
                # Remove CLS token and reshape
                patch_features = features[:, 1:, :]  # [B, N, D]

                # Generalized Mean (GeM) Pooling - better than average for VPR
                p = 3.0  # GeM power parameter
                gem_features = (patch_features.clamp(min=1e-6).pow(p).mean(dim=1)).pow(1.0/p)
                descriptor = gem_features

        return descriptor.cpu().numpy().flatten()

    def extract_local_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract local features for cross-correlation reranking.

        Returns patch-level features that can be used for spatial verification.
        """
        self._load_model()
        torch = _import_torch()

        if hasattr(self, '_is_fallback') and self._is_fallback:
            return np.zeros((1, 256, 768))  # Dummy features

        tensor = self._preprocess(image)

        with torch.no_grad():
            if hasattr(self, '_native_model') and self._native_model:
                local_feats = self.model.extract_local_features(tensor)
            else:
                # DINOv2 patch features
                features = self.backbone.get_intermediate_layers(tensor, n=1)[0]
                local_feats = features[:, 1:, :]  # Remove CLS token

        return local_feats.cpu().numpy()

    def compute_cross_correlation_score(self,
                                        query_features: np.ndarray,
                                        match_features: np.ndarray) -> float:
        """
        Compute cross-correlation score between query and match local features.

        This is the KEY INNOVATION of CricaVPR - using spatial correlation
        to distinguish visually similar but geometrically different places.

        Args:
            query_features: Local features from query image [N, D]
            match_features: Local features from match candidate [M, D]

        Returns:
            Cross-correlation score (higher = more likely same place)
        """
        torch = _import_torch()

        # Convert to tensors
        q = torch.from_numpy(query_features).float()
        m = torch.from_numpy(match_features).float()

        # Handle batch dimension
        if q.dim() == 3:
            q = q.squeeze(0)
        if m.dim() == 3:
            m = m.squeeze(0)

        # L2 normalize
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        m = m / (m.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute correlation matrix
        correlation = torch.mm(q, m.t())  # [N, M]

        # Bidirectional matching score
        # For each query patch, find best match; for each match patch, find best query
        q_to_m_scores = correlation.max(dim=1)[0]  # Best match for each query patch
        m_to_q_scores = correlation.max(dim=0)[0]  # Best query for each match patch

        # Combined score (geometric mean of mutual matches)
        score = (q_to_m_scores.mean() * m_to_q_scores.mean()).sqrt()

        return float(score)

    def rerank_candidates(self,
                          query_idx: int,
                          candidates: List[Tuple[int, float]],
                          top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Rerank candidates using cross-correlation.

        This is critical for perceptual aliasing - initial global descriptor
        matching may rank similar-looking but different places highly.
        Cross-correlation reranking uses local spatial structure to disambiguate.

        Args:
            query_idx: Index of query image
            candidates: List of (match_idx, global_similarity) tuples
            top_k: Number of top candidates to keep after reranking

        Returns:
            Reranked list of (match_idx, combined_score) tuples
        """
        if not self.use_reranking:
            return candidates[:top_k]

        # Get query local features (from cache if available)
        if query_idx not in self._feature_cache:
            return candidates[:top_k]  # Can't rerank without features

        query_feats = self._feature_cache[query_idx]

        reranked = []
        for match_idx, global_sim in candidates:
            if match_idx in self._feature_cache:
                match_feats = self._feature_cache[match_idx]
                cross_corr = self.compute_cross_correlation_score(query_feats, match_feats)
                # Combined score: weighted combination of global and local
                combined = 0.5 * global_sim + 0.5 * cross_corr
            else:
                combined = global_sim  # Fall back to global only

            reranked.append((match_idx, combined))

        # Sort by combined score
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]

    def add_image(self,
                  image: np.ndarray,
                  timestamp: float,
                  floor_label: Optional[int] = None,
                  image_path: Optional[str] = None) -> PlaceDescriptor:
        """Override to cache local features for reranking"""
        descriptor = self.extract_descriptor(image)
        place_desc = PlaceDescriptor(
            timestamp=timestamp,
            descriptor=descriptor,
            image_path=image_path,
            floor_label=floor_label
        )
        self.descriptors.append(place_desc)

        # Cache local features for potential reranking
        if self.use_reranking:
            idx = len(self.descriptors) - 1
            self._feature_cache[idx] = self.extract_local_features(image)

        return place_desc

    def _preprocess(self, image: np.ndarray) -> 'torch.Tensor':
        """Preprocess image for DINOv2/CricaVPR"""
        torch = _import_torch()
        cv2 = _import_cv2()

        # CricaVPR uses 322x322 input (divisible by 14 for ViT patch size)
        image = cv2.resize(image, (322, 322))

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ImageNet normalization (standard for DINOv2)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)


class SemanticPlaceRecognition:
    """
    Semantic-aware Place Recognition with Floor Gating

    Combines foundation model VPR with semantic floor constraints
    to prevent cross-floor false matches.
    """

    def __init__(self,
                 vpr_method: str = 'mixvpr',
                 device: str = 'cuda',
                 similarity_threshold: float = 0.5,
                 min_time_gap: float = 10.0):
        """
        Args:
            vpr_method: VPR method ('mixvpr', 'salad', 'anyloc')
            device: 'cuda' or 'cpu'
            similarity_threshold: Minimum similarity for valid match
            min_time_gap: Minimum temporal gap between query and match
        """
        self.similarity_threshold = similarity_threshold
        self.min_time_gap = min_time_gap

        # Initialize VPR method
        if vpr_method.lower() == 'mixvpr':
            self.vpr = MixVPR(device=device)
        elif vpr_method.lower() == 'salad':
            self.vpr = SALAD(device=device)
        elif vpr_method.lower() == 'anyloc':
            self.vpr = AnyLoc(device=device)
        elif vpr_method.lower() == 'cricavpr':
            # CricaVPR - BEST for perceptual aliasing in multi-floor environments
            self.vpr = CricaVPR(device=device, use_reranking=True)
        else:
            raise ValueError(f"Unknown VPR method: {vpr_method}. "
                           f"Available: mixvpr, salad, anyloc, cricavpr")

    def add_image(self,
                  image: np.ndarray,
                  timestamp: float,
                  floor_label: int,
                  image_path: Optional[str] = None) -> PlaceDescriptor:
        """Add image with floor label to database"""
        return self.vpr.add_image(image, timestamp, floor_label, image_path)

    def find_loop_closures(self,
                           enable_floor_gating: bool = True,
                           k: int = 10) -> List[PlaceMatch]:
        """
        Find all valid loop closure candidates in database.

        Args:
            enable_floor_gating: If True, reject cross-floor matches
            k: Number of candidates per query

        Returns:
            List of PlaceMatch objects
        """
        if len(self.vpr.descriptors) < 2:
            return []

        # Compute all pairwise similarities
        sim_matrix = self.vpr.compute_all_pairwise_similarities()
        n = len(self.vpr.descriptors)

        all_matches = []

        for i in range(n):
            # Get timestamps and floor labels
            query_time = self.vpr.descriptors[i].timestamp
            query_floor = self.vpr.descriptors[i].floor_label

            # Find top-k matches for this query
            similarities = sim_matrix[i].copy()

            # Mask out self and temporal neighbors
            for j in range(n):
                match_time = self.vpr.descriptors[j].timestamp
                if abs(match_time - query_time) < self.min_time_gap:
                    similarities[j] = -np.inf

            # Get top candidates
            top_k = np.argsort(similarities)[::-1][:k]

            for j in top_k:
                if similarities[j] < self.similarity_threshold:
                    continue

                match_floor = self.vpr.descriptors[j].floor_label

                # Check floor consistency
                is_valid = True
                if enable_floor_gating and query_floor is not None and match_floor is not None:
                    is_valid = (query_floor == match_floor)

                match = PlaceMatch(
                    query_idx=i,
                    match_idx=j,
                    similarity=float(similarities[j]),
                    query_timestamp=query_time,
                    match_timestamp=self.vpr.descriptors[j].timestamp,
                    is_valid=is_valid
                )
                all_matches.append(match)

        return all_matches

    def get_statistics(self, matches: List[PlaceMatch]) -> Dict:
        """Compute statistics on place recognition matches"""
        if not matches:
            return {
                'total_matches': 0,
                'valid_matches': 0,
                'rejected_matches': 0,
                'rejection_rate': 0.0
            }

        valid = sum(1 for m in matches if m.is_valid)
        rejected = sum(1 for m in matches if not m.is_valid)

        return {
            'total_matches': len(matches),
            'valid_matches': valid,
            'rejected_matches': rejected,
            'rejection_rate': rejected / len(matches) if matches else 0.0,
            'mean_similarity': np.mean([m.similarity for m in matches]),
            'mean_valid_similarity': np.mean([m.similarity for m in matches if m.is_valid]) if valid > 0 else 0.0
        }


def process_image_sequence(image_dir: Union[str, Path],
                           timestamps: np.ndarray,
                           floor_labels: np.ndarray,
                           vpr_method: str = 'mixvpr',
                           device: str = 'cuda') -> Tuple[SemanticPlaceRecognition, List[PlaceMatch]]:
    """
    Process an image sequence for place recognition.

    Args:
        image_dir: Directory containing images
        timestamps: Array of timestamps
        floor_labels: Array of floor labels
        vpr_method: VPR method to use
        device: 'cuda' or 'cpu'

    Returns:
        Tuple of (SemanticPlaceRecognition instance, list of matches)
    """
    cv2 = _import_cv2()

    image_dir = Path(image_dir)
    spr = SemanticPlaceRecognition(vpr_method=vpr_method, device=device)

    # Get sorted image files
    image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))

    if len(image_files) != len(timestamps):
        warnings.warn(
            f"Number of images ({len(image_files)}) != timestamps ({len(timestamps)}). "
            "Using minimum of both."
        )

    n = min(len(image_files), len(timestamps), len(floor_labels))

    print(f"Processing {n} images with {vpr_method}...")

    for i in range(n):
        image = cv2.imread(str(image_files[i]))
        if image is None:
            warnings.warn(f"Failed to load image: {image_files[i]}")
            continue

        spr.add_image(
            image=image,
            timestamp=timestamps[i],
            floor_label=int(floor_labels[i]),
            image_path=str(image_files[i])
        )

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n} images")

    print("Finding loop closure candidates...")
    matches = spr.find_loop_closures(enable_floor_gating=True)

    return spr, matches


def demo():
    """Demo with synthetic data"""
    print("Place Recognition Demo")
    print("=" * 50)

    # Create synthetic descriptors (no actual images needed for demo)
    spr = SemanticPlaceRecognition(vpr_method='mixvpr', device='cpu')

    # Simulate adding places from two floors
    n_places = 20

    print(f"Adding {n_places} synthetic places...")

    for i in range(n_places):
        # Create random descriptor (simulating extracted features)
        fake_descriptor = np.random.randn(4096).astype(np.float32)

        # Alternate floors
        floor = 1 if i < n_places // 2 else 2

        # Create fake PlaceDescriptor
        desc = PlaceDescriptor(
            timestamp=float(i) * 2.0,  # 2 second intervals
            descriptor=fake_descriptor,
            floor_label=floor
        )
        spr.vpr.descriptors.append(desc)

    print(f"Finding loop closures...")

    # Find matches without gating
    matches_no_gate = spr.find_loop_closures(enable_floor_gating=False)
    stats_no_gate = spr.get_statistics(matches_no_gate)

    # Find matches with gating
    matches_gated = spr.find_loop_closures(enable_floor_gating=True)
    stats_gated = spr.get_statistics(matches_gated)

    print("\nWithout Floor Gating:")
    print(f"  Total matches: {stats_no_gate['total_matches']}")
    print(f"  Mean similarity: {stats_no_gate.get('mean_similarity', 0):.3f}")

    print("\nWith Floor Gating:")
    print(f"  Valid matches: {stats_gated['valid_matches']}")
    print(f"  Rejected (cross-floor): {stats_gated['rejected_matches']}")
    print(f"  Rejection rate: {stats_gated['rejection_rate']:.1%}")


if __name__ == '__main__':
    demo()
