"""
Traffic Management System
─────────────────────────
Features:
  • Vehicle detection & multi-object tracking
  • Vehicle counting (per class, per direction)
  • Speed estimation
  • Lane violation detection
  • Vehicle Re-ID across cameras
"""

import cv2
import numpy as np
import time
from collections import defaultdict

from core.detector  import Detector
from core.tracker   import Tracker
from core.reid      import ReIDFeatureExtractor, ReIDGallery
from core.video_io  import VideoReader, VideoWriter, FPSCounter
from config.settings import Settings
from utils.display  import print_info, print_warning, print_alert, print_stat, print_section
from utils.logger   import setup_logger

logger = setup_logger("traffic")


class SpeedEstimator:
    """
    Estimates speed using frame-to-frame displacement and a pixels-per-meter
    calibration factor.
    Call calibrate() to set real-world scale; default is an approximation.
    """

    def __init__(self, fps: float, scale: float = 0.05):
        self.fps   = fps
        self.scale = scale           # metres per pixel (rough approximation)
        self._prev: dict[int, tuple] = {}   # track_id -> (cx, cy, timestamp)

    def update(self, track_id: int, center: tuple) -> float | None:
        now = time.time()
        if track_id in self._prev:
            px, py, pt = self._prev[track_id]
            dt = now - pt
            if dt > 0:
                dist_px = np.hypot(center[0] - px, center[1] - py)
                speed_ms  = dist_px * self.scale / dt
                speed_kmh = speed_ms * 3.6
                self._prev[track_id] = (*center, now)
                return round(speed_kmh, 1)
        self._prev[track_id] = (*center, now)
        return None

    def remove(self, track_id: int):
        self._prev.pop(track_id, None)


class LaneViolationDetector:
    """
    Detects lane violations using configurable line zones.
    A violation is triggered when a vehicle crosses a restricted lane boundary.
    """

    def __init__(self, frame_width: int, frame_height: int):
        # Default: divider line in the centre of the frame
        self.lane_lines = [
            {
                "id": "centre_divider",
                "pt1": (frame_width // 2, 0),
                "pt2": (frame_width // 2, frame_height),
                "direction": "left_to_right",  # which crossing is a violation
            }
        ]
        self._prev_sides: dict[int, str] = {}

    def check(self, track_id: int, center: tuple) -> bool:
        """Return True if this track has just crossed a violation line."""
        cx = center[0]
        side = "left" if cx < self.lane_lines[0]["pt1"][0] else "right"
        violation = False

        if track_id in self._prev_sides:
            prev = self._prev_sides[track_id]
            rule = self.lane_lines[0]["direction"]
            if rule == "left_to_right" and prev == "left" and side == "right":
                violation = True
            elif rule == "right_to_left" and prev == "right" and side == "left":
                violation = True

        self._prev_sides[track_id] = side
        return violation

    def draw(self, frame: np.ndarray):
        for line in self.lane_lines:
            cv2.line(frame, line["pt1"], line["pt2"], (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "LANE BOUNDARY", (line["pt1"][0] + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


class VehicleCounter:
    """Count vehicles crossing a horizontal counting line."""

    def __init__(self, frame_height: int, line_y_ratio: float = 0.55):
        self.line_y  = int(frame_height * line_y_ratio)
        self._counted: set  = set()
        self.counts: dict   = defaultdict(int)
        self.total: int     = 0

    def update(self, track_id: int, center: tuple, class_name: str,
               prev_center: tuple | None = None) -> bool:
        if track_id in self._counted or prev_center is None:
            return False
        py, cy = prev_center[1], center[1]
        # Crossed the line downward (top → bottom)
        if py < self.line_y <= cy:
            self._counted.add(track_id)
            self.counts[class_name] += 1
            self.total += 1
            return True
        return False

    def draw(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        cv2.line(frame, (0, self.line_y), (w, self.line_y), (255, 255, 0), 2)
        cv2.putText(frame, "COUNTING LINE", (10, self.line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


class TrafficSystem:
    """Main Traffic Management pipeline."""

    def __init__(self, source, save_output: bool = False,
                 show_display: bool = True, conf_threshold: float = 0.4):
        self.settings     = Settings()
        self.settings.confidence_threshold = conf_threshold
        self.show_display = show_display
        self.save_output  = save_output
        self.source       = source

        print_section("Initialising Traffic Management System")
        self.detector  = Detector(self.settings)
        self.tracker   = Tracker(self.settings)
        self.extractor = ReIDFeatureExtractor()
        self.gallery   = ReIDGallery(self.settings.reid_similarity_threshold)
        self.fps_ctr   = FPSCounter()

        # Per-track data
        self._speeds:     dict[int, float]  = {}
        self._violations: set               = set()
        self._reid_map:   dict[int, int]    = {}   # track_id -> reid_id
        self._prev_centers: dict[int, tuple] = {}

        print_info("Traffic system ready.")

    def run(self):
        reader = VideoReader(self.source, self.settings)
        writer = VideoWriter(self.source, "traffic", self.settings) if self.save_output else None

        speed_est  = SpeedEstimator(reader.fps, self.settings.speed_scale_factor)
        lane_det   = LaneViolationDetector(reader.width, reader.height)
        counter    = VehicleCounter(reader.height)

        frame_num  = 0
        start_time = time.time()

        print_info("Processing video… Press Q to stop.\n")

        try:
            while True:
                ret, frame = reader.read()
                if not ret:
                    break

                frame_num += 1
                fps = self.fps_ctr.tick()

                # ── Detection ────────────────────────────────────────────────
                detections = self.detector.detect_vehicles(frame)

                # ── Tracking ─────────────────────────────────────────────────
                tracks = self.tracker.update(detections, frame)

                for track in tracks:
                    tid = track.track_id

                    # Speed
                    speed = speed_est.update(tid, track.center)
                    if speed is not None:
                        self._speeds[tid] = speed

                    # Counting
                    prev = self._prev_centers.get(tid)
                    counted = counter.update(tid, track.center, track.class_name, prev)
                    if counted:
                        print_info(f"Vehicle counted | ID:{tid} | Class:{track.class_name} | Total:{counter.total}")

                    # Lane violation
                    if lane_det.check(tid, track.center):
                        self._violations.add(tid)
                        print_alert(f"LANE VIOLATION! Track ID: {tid}  Class: {track.class_name}")

                    # Re-ID (every 10 frames per track to save compute)
                    if frame_num % 10 == 0:
                        feat = self.extractor.extract(frame, track.bbox)
                        reid_id, is_new = self.gallery.get_or_register(feat)
                        self._reid_map[tid] = reid_id
                        if is_new:
                            logger.info(f"New Re-ID identity registered: {reid_id}")

                    self._prev_centers[tid] = track.center

                # ── Draw ─────────────────────────────────────────────────────
                annotated = self._draw(frame, tracks, counter, fps)
                lane_det.draw(annotated)
                counter.draw(annotated)

                if writer:
                    writer.write(annotated)

                if self.show_display:
                    cv2.imshow("Smart Vision – Traffic Mode", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        snap = f"output/snapshot_traffic_{frame_num}.jpg"
                        cv2.imwrite(snap, annotated)
                        print_info(f"Snapshot saved: {snap}")
                    elif key == ord("r"):
                        counter.total = 0
                        counter.counts.clear()
                        print_info("Counters reset.")

                # Console stats every 60 frames
                if frame_num % 60 == 0:
                    self._print_stats(counter, fps, time.time() - start_time)

        finally:
            reader.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self._print_summary(counter, frame_num, time.time() - start_time)

    def _draw(self, frame, tracks, counter, fps):
        out = frame.copy()
        S   = self.settings

        for track in tracks:
            tid  = track.track_id
            x1, y1, x2, y2 = track.bbox
            color = S.COLOR_ALERT if tid in self._violations else S.COLOR_NORMAL

            cv2.rectangle(out, (x1, y1), (x2, y2), color, S.bbox_thickness)

            # Label
            speed_str = f" {self._speeds[tid]:.0f}km/h" if tid in self._speeds else ""
            reid_str  = f" R{self._reid_map[tid]}" if tid in self._reid_map else ""
            label = f"ID:{tid} {track.class_name}{speed_str}{reid_str}"
            cv2.putText(out, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

            # Speed alert
            spd = self._speeds.get(tid, 0)
            if spd > self.settings.speed_limit_kmh:
                cv2.putText(out, "SPEEDING!", (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, S.COLOR_ALERT, 2)

            # Track trail
            for i in range(1, len(track.history)):
                cv2.line(out, track.history[i-1], track.history[i],
                         S.COLOR_TRACK, 1, cv2.LINE_AA)

        # HUD overlay
        hud_lines = [
            f"FPS: {fps:.1f}",
            f"Vehicles: {counter.total}",
            f"Violations: {len(self._violations)}",
            f"Speed limit: {self.settings.speed_limit_kmh} km/h",
        ]
        for i, txt in enumerate(hud_lines):
            cv2.putText(out, txt, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, S.COLOR_INFO, 2, cv2.LINE_AA)

        return out

    def _print_stats(self, counter, fps, elapsed):
        print_section("Traffic Stats")
        print_stat("FPS",            f"{fps:.1f}")
        print_stat("Vehicles counted", counter.total)
        print_stat("Violations",     len(self._violations))
        print_stat("Active tracks",  len(self._reid_map))
        print_stat("Elapsed",        f"{elapsed:.0f}s")

    def _print_summary(self, counter, frames, elapsed):
        print_section("Session Summary – Traffic")
        print_stat("Total frames processed", frames)
        print_stat("Total vehicles counted", counter.total)
        for cls, cnt in counter.counts.items():
            print_stat(f"  {cls}", cnt)
        print_stat("Lane violations",   len(self._violations))
        print_stat("Re-ID identities",  self.gallery._next_id - 1)
        print_stat("Duration",          f"{elapsed:.1f}s")
