"""
DeepSORT-based multi-object tracker.
Falls back to a simple IoU tracker if deep_sort_realtime is not installed.
"""

import numpy as np
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("tracker")


class Track:
    """Represents a tracked object."""
    def __init__(self, track_id: int, bbox: list, class_name: str, confidence: float):
        self.track_id   = track_id
        self.bbox       = bbox          # [x1, y1, x2, y2]
        self.class_name = class_name
        self.confidence = confidence
        self.history    = []            # list of centre points
        self.age        = 0             # frames since first seen
        self.update(bbox, confidence)

    def update(self, bbox, confidence):
        self.bbox       = bbox
        self.confidence = confidence
        self.age       += 1
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.history.append((cx, cy))
        if len(self.history) > 60:      # keep last 60 points
            self.history.pop(0)

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]


class Tracker:
    """
    Wraps DeepSORT (deep_sort_realtime).
    Auto-falls-back to IoU-only tracking when the library is absent.
    """

    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self._ds = DeepSort(
                max_age=self.settings.max_track_age,
                n_init=self.settings.min_hits,
                nn_budget=100,
                embedder="mobilenet",
                half=True,
                bgr=True
            )
            self._use_deepsort = True
            logger.info("Using DeepSORT tracker.")
        except ImportError:
            self._ds = None
            self._use_deepsort = False
            logger.warning("deep_sort_realtime not found – falling back to IoU tracker.")

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections: list, frame: np.ndarray) -> list[Track]:
        """
        Update tracker with current-frame detections.

        Args:
            detections: list of core.detector.Detection objects
            frame:      BGR image

        Returns:
            list of active Track objects
        """
        if self._use_deepsort:
            return self._update_deepsort(detections, frame)
        return self._update_iou(detections)

    def get_track(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)

    # ── DeepSORT ─────────────────────────────────────────────────────────────

    def _update_deepsort(self, detections, frame) -> list[Track]:
        raw = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            raw.append(([x1, y1, x2 - x1, y2 - y1], d.confidence, d.class_name))

        ds_tracks = self._ds.update_tracks(raw, frame=frame)
        active: list[Track] = []

        for t in ds_tracks:
            if not t.is_confirmed():
                continue
            tid  = t.track_id
            ltrb = t.to_ltrb()
            bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
            cls  = t.get_det_class() or "object"
            conf = t.get_det_conf() or 0.0

            if tid not in self._tracks:
                self._tracks[tid] = Track(tid, bbox, cls, conf)
            else:
                self._tracks[tid].update(bbox, conf)
            active.append(self._tracks[tid])

        return active

    # ── Fallback IoU tracker ──────────────────────────────────────────────────

    def _update_iou(self, detections) -> list[Track]:
        if not detections:
            return []

        unmatched_dets = list(range(len(detections)))
        active_ids     = list(self._tracks.keys())
        matched_ids    = set()

        # Match by IoU
        for det_idx in list(unmatched_dets):
            det = detections[det_idx]
            best_iou, best_tid = 0, None
            for tid in active_ids:
                if tid in matched_ids:
                    continue
                iou = self._iou(det.bbox, self._tracks[tid].bbox)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_iou > 0.3 and best_tid is not None:
                self._tracks[best_tid].update(det.bbox, det.confidence)
                matched_ids.add(best_tid)
                unmatched_dets.remove(det_idx)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(tid, det.bbox, det.class_name, det.confidence)

        # Remove stale tracks
        stale = [tid for tid, t in self._tracks.items()
                 if tid not in matched_ids and t.age > self.settings.max_track_age]
        for tid in stale:
            del self._tracks[tid]

        return list(self._tracks.values())

    @staticmethod
    def _iou(boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / float(areaA + areaB - inter)
