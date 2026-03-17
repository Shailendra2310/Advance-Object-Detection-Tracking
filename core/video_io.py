"""
Video I/O – wraps OpenCV VideoCapture and VideoWriter.
Supports webcam, video files, and RTSP streams.
"""

import cv2
import os
import numpy as np
from datetime import datetime
from config.settings import Settings
from utils.logger import setup_logger

logger = setup_logger("video_io")


class VideoReader:
    """Read frames from webcam, file, or RTSP stream."""

    def __init__(self, source, settings: Settings = None):
        self.settings = settings or Settings()
        self.source   = source
        self.cap      = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Opened source: {source}  {self.width}x{self.height}  {self.fps:.1f}fps")

    def read(self) -> tuple[bool, np.ndarray | None]:
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.read()
        if not ret:
            raise StopIteration
        return frame


class VideoWriter:
    """Write annotated frames to an MP4 file."""

    def __init__(self, source_path, mode: str, settings: Settings = None):
        self.settings = settings or Settings()
        self.writer   = None
        self._init_writer(source_path, mode)

    def _init_writer(self, source, mode):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        src_name = "webcam" if isinstance(source, int) else os.path.splitext(os.path.basename(source))[0]
        filename = f"{mode}_{src_name}_{ts}.mp4"
        out_path = os.path.join(self.settings.recordings_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*self.settings.video_codec)
        self.writer = cv2.VideoWriter(
            out_path, fourcc, self.settings.save_fps,
            (self.settings.display_width, self.settings.display_height)
        )
        logger.info(f"VideoWriter initialised: {out_path}")

    def write(self, frame: np.ndarray):
        if self.writer:
            resized = cv2.resize(frame, (self.settings.display_width, self.settings.display_height))
            self.writer.write(resized)

    def release(self):
        if self.writer:
            self.writer.release()


class FPSCounter:
    """Rolling-average FPS counter."""

    def __init__(self, window: int = 30):
        self._times  = []
        self._window = window
        self._last   = None

    def tick(self) -> float:
        import time
        now = time.perf_counter()
        if self._last is not None:
            self._times.append(now - self._last)
            if len(self._times) > self._window:
                self._times.pop(0)
        self._last = now
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))
