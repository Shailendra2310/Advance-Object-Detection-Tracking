"""
Smart Surveillance System
──────────────────────────
Features:
  • Person detection & multi-object tracking
  • Person Re-ID across camera views
  • Loitering detection (dwell-time threshold)
  • Fall detection (aspect-ratio + velocity heuristic)
  • Crowd density heatmap
"""

import cv2
import numpy as np
import time
from collections import defaultdict

from core.detector   import Detector
from core.tracker    import Tracker
from core.reid       import ReIDFeatureExtractor, ReIDGallery
from core.video_io   import VideoReader, VideoWriter, FPSCounter
from config.settings import Settings
from utils.display   import print_info, print_warning, print_alert, print_stat, print_section
from utils.logger    import setup_logger

logger = setup_logger("surveillance")


class LoiteringDetector:
    """Alert when a person stays in roughly the same region for too long."""

    def __init__(self, time_threshold: float = 60.0, movement_threshold: int = 50):
        self.time_threshold     = time_threshold      # seconds
        self.movement_threshold = movement_threshold  # pixels – below = "stationary"
        self._entry_time: dict[int, float]  = {}
        self._entry_pos:  dict[int, tuple]  = {}
        self._alerted:    set               = set()

    def update(self, track_id: int, center: tuple) -> bool:
        """Return True on first loitering alert for this track."""
        now = time.time()
        if track_id not in self._entry_time:
            self._entry_time[track_id] = now
            self._entry_pos[track_id]  = center
            return False

        # Reset if person has moved significantly
        dist = np.hypot(center[0] - self._entry_pos[track_id][0],
                        center[1] - self._entry_pos[track_id][1])
        if dist > self.movement_threshold:
            self._entry_time[track_id] = now
            self._entry_pos[track_id]  = center
            self._alerted.discard(track_id)
            return False

        dwell = now - self._entry_time[track_id]
        if dwell >= self.time_threshold and track_id not in self._alerted:
            self._alerted.add(track_id)
            return True

        return False

    def get_dwell_time(self, track_id: int) -> float:
        if track_id not in self._entry_time:
            return 0.0
        return time.time() - self._entry_time[track_id]

    def remove(self, track_id: int):
        self._entry_time.pop(track_id, None)
        self._entry_pos.pop(track_id, None)
        self._alerted.discard(track_id)


class FallDetector:
    """
    Detects falls using two signals:
      1. Bounding-box aspect ratio (width / height > threshold → horizontal)
      2. Sudden downward velocity spike
    """

    def __init__(self, ar_threshold: float = 0.75, velocity_threshold: int = 40):
        self.ar_threshold       = ar_threshold
        self.velocity_threshold = velocity_threshold
        self._prev_cy: dict[int, float] = {}
        self._fall_frames: dict[int, int] = defaultdict(int)
        self._alerted: set = set()

    def update(self, track_id: int, bbox: list) -> bool:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = max(y2 - y1, 1)
        ar = w / h

        cy = (y1 + y2) / 2
        velocity = 0
        if track_id in self._prev_cy:
            velocity = cy - self._prev_cy[track_id]
        self._prev_cy[track_id] = cy

        is_fallen = ar > self.ar_threshold and velocity >= 0
        if is_fallen:
            self._fall_frames[track_id] += 1
        else:
            self._fall_frames[track_id] = max(0, self._fall_frames[track_id] - 1)

        # Require 3 consecutive "fallen" frames to reduce false positives
        if self._fall_frames[track_id] >= 3 and track_id not in self._alerted:
            self._alerted.add(track_id)
            return True

        if self._fall_frames[track_id] == 0:
            self._alerted.discard(track_id)

        return False

    def is_fallen(self, track_id: int) -> bool:
        return self._fall_frames.get(track_id, 0) >= 3


class CrowdHeatmap:
    """
    Accumulates person-centre positions and renders a Gaussian heat map.
    """

    def __init__(self, frame_shape: tuple):
        h, w = frame_shape[:2]
        self._map   = np.zeros((h, w), dtype=np.float32)
        self._decay = 0.97      # Slowly fade old data
        self._sigma = 30

    def update(self, centers: list[tuple]):
        self._map *= self._decay
        for cx, cy in centers:
            cv2.circle(self._map, (cx, cy), self._sigma, 1.0, -1)
            # Slight Gaussian blur for smoothness
        if centers:
            self._map = cv2.GaussianBlur(self._map, (0, 0), self._sigma * 0.4)
        self._map = np.clip(self._map, 0, 1)

    def overlay(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        norm = (self._map * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        mask = norm > 10
        out = frame.copy()
        out[mask] = cv2.addWeighted(frame, 1 - alpha, coloured, alpha, 0)[mask]
        return out

    def get_density(self, region: tuple | None = None) -> float:
        """Return mean intensity (0–1) as a crowd density indicator."""
        return float(np.mean(self._map))


class SurveillanceSystem:
    """Main Surveillance pipeline."""

    def __init__(self, source, save_output: bool = False,
                 show_display: bool = True, conf_threshold: float = 0.4):
        self.settings     = Settings()
        self.settings.confidence_threshold = conf_threshold
        self.show_display = show_display
        self.save_output  = save_output
        self.source       = source
        self._show_heatmap = False

        print_section("Initialising Smart Surveillance System")
        self.detector   = Detector(self.settings)
        self.tracker    = Tracker(self.settings)
        self.extractor  = ReIDFeatureExtractor()
        self.gallery    = ReIDGallery(self.settings.reid_similarity_threshold)
        self.fps_ctr    = FPSCounter()
        self.loitering  = LoiteringDetector(self.settings.loitering_time_threshold)
        self.fall_det   = FallDetector(self.settings.fall_aspect_ratio_threshold)

        # State
        self._reid_map:    dict[int, int]   = {}
        self._loiterers:   set              = set()
        self._fallen:      set              = set()
        self._person_count = 0

        print_info("Surveillance system ready.")

    def run(self):
        reader   = VideoReader(self.source, self.settings)
        writer   = VideoWriter(self.source, "surveillance", self.settings) if self.save_output else None
        heatmap  = CrowdHeatmap((reader.height, reader.width))

        frame_num  = 0
        start_time = time.time()
        print_info("Processing video… Press Q to stop | H to toggle heatmap.\n")

        try:
            while True:
                ret, frame = reader.read()
                if not ret:
                    break

                frame_num += 1
                fps = self.fps_ctr.tick()

                # ── Detection ────────────────────────────────────────────────
                detections = self.detector.detect_persons(frame)
                self._person_count = len(detections)

                # ── Tracking ─────────────────────────────────────────────────
                tracks = self.tracker.update(detections, frame)
                centers = []

                for track in tracks:
                    tid = track.track_id
                    centers.append(track.center)

                    # Re-ID
                    if frame_num % 10 == 0:
                        feat = self.extractor.extract(frame, track.bbox)
                        reid_id, is_new = self.gallery.get_or_register(feat)
                        self._reid_map[tid] = reid_id

                    # Loitering
                    if self.loitering.update(tid, track.center):
                        self._loiterers.add(tid)
                        print_alert(f"LOITERING DETECTED! Track ID:{tid}  "
                                    f"Re-ID:{self._reid_map.get(tid,'?')}  "
                                    f"Dwell:{self.loitering.get_dwell_time(tid):.0f}s")

                    # Fall
                    if self.fall_det.update(tid, track.bbox):
                        self._fallen.add(tid)
                        print_alert(f"FALL DETECTED! Track ID:{tid}  Re-ID:{self._reid_map.get(tid,'?')}")

                # Crowd alert
                if len(tracks) >= self.settings.crowd_density_threshold:
                    if frame_num % 30 == 0:
                        print_warning(f"CROWD ALERT – {len(tracks)} persons detected in frame!")

                # Heatmap update
                heatmap.update(centers)

                # ── Draw ─────────────────────────────────────────────────────
                annotated = self._draw(frame, tracks, fps, len(tracks))
                if self._show_heatmap:
                    annotated = heatmap.overlay(annotated, self.settings.heatmap_alpha)

                if writer:
                    writer.write(annotated)

                if self.show_display:
                    cv2.imshow("Smart Vision – Surveillance Mode", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("h"):
                        self._show_heatmap = not self._show_heatmap
                        print_info(f"Heatmap {'ON' if self._show_heatmap else 'OFF'}")
                    elif key == ord("s"):
                        snap = f"output/snapshot_surveillance_{frame_num}.jpg"
                        cv2.imwrite(snap, annotated)
                        print_info(f"Snapshot saved: {snap}")

                if frame_num % 60 == 0:
                    self._print_stats(tracks, fps, time.time() - start_time)

        finally:
            reader.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self._print_summary(frame_num, time.time() - start_time)

    def _draw(self, frame, tracks, fps, person_count):
        out = frame.copy()
        S   = self.settings

        for track in tracks:
            tid = track.track_id
            x1, y1, x2, y2 = track.bbox

            # Choose colour based on alert state
            if tid in self._fallen:
                color = (0, 0, 255)     # Red – fall
            elif tid in self._loiterers:
                color = (0, 165, 255)   # Orange – loitering
            else:
                color = S.COLOR_NORMAL

            cv2.rectangle(out, (x1, y1), (x2, y2), color, S.bbox_thickness)

            reid_str  = f" R{self._reid_map[tid]}" if tid in self._reid_map else ""
            dwell_str = f" {self.loitering.get_dwell_time(tid):.0f}s"
            label = f"ID:{tid}{reid_str}{dwell_str}"
            cv2.putText(out, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Alert badges
            if tid in self._fallen:
                cv2.putText(out, "FALL!", (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif tid in self._loiterers:
                cv2.putText(out, "LOITERING", (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # Track trail
            for i in range(1, len(track.history)):
                cv2.line(out, track.history[i-1], track.history[i],
                         S.COLOR_TRACK, 1, cv2.LINE_AA)

        # HUD
        hud = [
            f"FPS: {fps:.1f}",
            f"Persons: {person_count}",
            f"Loitering: {len(self._loiterers)}",
            f"Falls: {len(self._fallen)}",
            f"Heatmap: {'ON' if self._show_heatmap else 'OFF'} [H]",
        ]
        for i, txt in enumerate(hud):
            cv2.putText(out, txt, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, S.COLOR_INFO, 2, cv2.LINE_AA)

        # Crowd warning banner
        if person_count >= S.crowd_density_threshold:
            cv2.putText(out, f"⚠ CROWD ALERT ({person_count} persons)",
                        (out.shape[1] // 2 - 160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        return out

    def _print_stats(self, tracks, fps, elapsed):
        print_section("Surveillance Stats")
        print_stat("FPS",            f"{fps:.1f}")
        print_stat("Active persons", len(tracks))
        print_stat("Loitering",      len(self._loiterers))
        print_stat("Falls detected", len(self._fallen))
        print_stat("Re-ID gallery",  self.gallery._next_id - 1)
        print_stat("Elapsed",        f"{elapsed:.0f}s")

    def _print_summary(self, frames, elapsed):
        print_section("Session Summary – Surveillance")
        print_stat("Total frames",      frames)
        print_stat("Loitering events",  len(self._loiterers))
        print_stat("Fall events",       len(self._fallen))
        print_stat("Re-ID identities",  self.gallery._next_id - 1)
        print_stat("Duration",          f"{elapsed:.1f}s")
