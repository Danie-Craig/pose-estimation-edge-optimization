"""Data loading utilities for pose estimation inference."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class FrameData:
    """Container for a single frame of input data."""
    image: np.ndarray          # BGR image (H, W, 3)
    source_path: str           # Where it came from
    frame_id: int              # Sequential frame number
    original_size: tuple[int, int]  # (height, width) before any resizing


class ImageDirectoryLoader:
    """Load images from a directory."""

    def __init__(self, directory: str, extensions: tuple[str, ...] = (".jpg", ".png", ".jpeg")) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        self.image_paths = sorted(
            p for p in self.directory.iterdir()
            if p.suffix.lower() in extensions
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {directory} with extensions {extensions}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self) -> Iterator[FrameData]:
        for idx, path in enumerate(self.image_paths):
            image = cv2.imread(str(path))
            if image is None:
                print(f"Warning: Could not read {path}, skipping.")
                continue
            h, w = image.shape[:2]
            yield FrameData(
                image=image,
                source_path=str(path),
                frame_id=idx,
                original_size=(h, w),
            )


class VideoLoader:
    """Load frames from a video file."""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def __len__(self) -> int:
        return self.total_frames

    def __iter__(self) -> Iterator[FrameData]:
        cap = cv2.VideoCapture(self.video_path)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            yield FrameData(
                image=frame,
                source_path=self.video_path,
                frame_id=frame_id,
                original_size=(h, w),
            )
            frame_id += 1
        cap.release()


class WebcamLoader:
    """Load frames from a webcam (or virtual camera)."""

    def __init__(self, camera_id: int = 0, max_frames: int | None = None) -> None:
        self.camera_id = camera_id
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[FrameData]:
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        frame_id = 0
        while True:
            if self.max_frames and frame_id >= self.max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            yield FrameData(
                image=frame,
                source_path=f"camera:{self.camera_id}",
                frame_id=frame_id,
                original_size=(h, w),
            )
            frame_id += 1
        cap.release()


def get_loader(source: str, **kwargs) -> ImageDirectoryLoader | VideoLoader | WebcamLoader:
    """Factory function to get the right loader based on source type."""
    if os.path.isdir(source):
        return ImageDirectoryLoader(source, **kwargs)
    elif os.path.isfile(source):
        return VideoLoader(source, **kwargs)
    elif source.startswith("camera:"):
        cam_id = int(source.split(":")[1])
        return WebcamLoader(cam_id, **kwargs)
    else:
        raise ValueError(f"Unknown source type: {source}")
