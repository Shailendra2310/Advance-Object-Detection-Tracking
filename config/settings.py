"""
Configuration settings for Smart Vision System.
Modify these values to tune the system behaviour.
"""

import os
import torch


class Settings:
    # ── Model ──────────────────────────────────────────────
    model_size: str = "yolov8m"          # yolov8n / yolov8s / yolov8m / yolov8l / yolov8x
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.45
    input_size: int = 640                # YOLO input resolution

    # ── Tracker (DeepSORT) ─────────────────────────────────
    max_track_age: int = 30              # Frames before track is dropped
    min_hits: int = 3                    # Frames before track is confirmed
    iou_threshold_tracker: float = 0.3

    # ── Re-ID ──────────────────────────────────────────────
    reid_similarity_threshold: float = 0.55
    reid_feature_dim: int = 512

    # ── Traffic Mode ───────────────────────────────────────
    speed_estimation_fps: int = 30
    speed_scale_factor: float = 0.05    # pixels-per-frame to km/h conversion factor
    speed_limit_kmh: int = 60
    lane_violation_sensitivity: float = 0.7
    vehicle_classes: list = None        # Set in __init__

    # ── Surveillance Mode ──────────────────────────────────
    loitering_time_threshold: int = 60  # seconds before loitering alert
    fall_aspect_ratio_threshold: float = 0.6  # width/height ratio for fall
    crowd_density_threshold: int = 10   # persons in zone before crowd alert
    heatmap_alpha: float = 0.5          # heatmap overlay transparency

    # ── Display ────────────────────────────────────────────
    display_width: int = 1280
    display_height: int = 720
    show_fps: bool = True
    show_track_ids: bool = True
    show_confidence: bool = True
    bbox_thickness: int = 2

    # ── Output ─────────────────────────────────────────────
    output_dir: str = "output"
    recordings_dir: str = "output/recordings"
    logs_dir: str = "output/logs"
    video_codec: str = "mp4v"
    save_fps: int = 20

    # ── Device ─────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Colours (BGR) ──────────────────────────────────────
    COLOR_NORMAL    = (0, 255, 0)        # Green  – normal
    COLOR_WARNING   = (0, 165, 255)      # Orange – warning
    COLOR_ALERT     = (0, 0, 255)        # Red    – alert/violation
    COLOR_INFO      = (255, 255, 0)      # Cyan   – info text
    COLOR_TRACK     = (255, 128, 0)      # Blue   – track line

    def __init__(self):
        self.vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs("models/weights", exist_ok=True)
