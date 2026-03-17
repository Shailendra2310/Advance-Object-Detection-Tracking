"""
Re-Identification (Re-ID) module.
Extracts appearance embeddings and matches objects across camera views or
after occlusion using cosine similarity.
"""

import cv2
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("reid")


class ReIDFeatureExtractor:
    """
    Lightweight Re-ID feature extractor.
    Uses a colour + HOG histogram approach as a dependency-free baseline.
    Replace with OSNet / FastReID for production-grade accuracy.
    """

    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        self._try_load_deep_model()

    def _try_load_deep_model(self):
        """Attempt to load a deep Re-ID model (torchreid / OSNet)."""
        try:
            import torchreid
            self.extractor = torchreid.utils.FeatureExtractor(
                model_name="osnet_x0_25",
                device="cuda"
            )
            self._use_deep = True
            logger.info("Re-ID: Using OSNet deep feature extractor.")
        except Exception:
            self.extractor = None
            self._use_deep = False
            logger.info("Re-ID: Using colour-histogram baseline extractor.")

    def extract(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Extract normalised feature vector for the crop defined by bbox."""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.feature_dim)

        if self._use_deep:
            return self._deep_extract(crop)
        return self._histogram_extract(crop)

    def _deep_extract(self, crop: np.ndarray) -> np.ndarray:
        import torch
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        feat = self.extractor([crop_rgb])
        v = feat[0].cpu().numpy()
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _histogram_extract(self, crop: np.ndarray) -> np.ndarray:
        """Combine colour + HOG histogram into a fixed-length descriptor."""
        resized = cv2.resize(crop, (64, 128))
        hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Colour histogram – H:32 bins, S:32 bins per body strip
        feats = []
        for strip in np.array_split(hsv, 4, axis=0):
            h_hist = cv2.calcHist([strip], [0], None, [32], [0, 180]).flatten()
            s_hist = cv2.calcHist([strip], [1], None, [32], [0, 256]).flatten()
            feats.extend(h_hist)
            feats.extend(s_hist)

        # Simple gradient magnitude
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag  = cv2.magnitude(gx, gy)
        g_hist = np.histogram(mag.flatten(), bins=64, range=(0, 255))[0].astype(float)
        feats.extend(g_hist)

        feat = np.array(feats, dtype=np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat


class ReIDGallery:
    """
    Manages a gallery of known identities and performs matching.
    """

    def __init__(self, similarity_threshold: float = 0.55):
        self.threshold = similarity_threshold
        self.gallery: dict[int, np.ndarray] = {}   # reid_id -> feature vector
        self._next_id = 1

    def update(self, track_id: int, feature: np.ndarray):
        """Add / update feature for a tracked object."""
        if track_id in self.gallery:
            # Exponential moving average
            self.gallery[track_id] = 0.7 * self.gallery[track_id] + 0.3 * feature
        else:
            self.gallery[track_id] = feature.copy()

    def match(self, feature: np.ndarray) -> int | None:
        """Return the best-matching gallery ID, or None if below threshold."""
        if not self.gallery:
            return None

        best_sim, best_id = -1, None
        for gid, gfeat in self.gallery.items():
            sim = self._cosine_similarity(feature, gfeat)
            if sim > best_sim:
                best_sim, best_id = sim, gid

        return best_id if best_sim >= self.threshold else None

    def register(self, feature: np.ndarray) -> int:
        """Register a new identity and return its ID."""
        new_id = self._next_id
        self._next_id += 1
        self.gallery[new_id] = feature.copy()
        return new_id

    def get_or_register(self, feature: np.ndarray) -> tuple[int, bool]:
        """
        Try to match; register as new if no match found.
        Returns (reid_id, is_new).
        """
        matched = self.match(feature)
        if matched is not None:
            self.update(matched, feature)
            return matched, False
        new_id = self.register(feature)
        return new_id, True

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
