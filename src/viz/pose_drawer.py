"""Visualization tools for pose estimation results."""

import cv2
import numpy as np
import yaml


# Color palette for different people (BGR format)
PERSON_COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 255),    # Purple
    (255, 128, 0),    # Light blue
    (0, 128, 255),    # Orange
    (128, 255, 0),    # Light green
]


class PoseDrawer:
    """Draw pose estimation results on images."""

    def __init__(self, config_path: str = "configs/model_config.yaml") -> None:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.keypoint_names: list[str] = config["keypoints"]["names"]
        self.skeleton: list[list[int]] = config["keypoints"]["skeleton"]
        self.confidence_threshold: float = config["inference"]["confidence_threshold"]

    def draw_frame(self, image: np.ndarray, frame_result,
                   show_bbox: bool = True,
                   show_keypoints: bool = True,
                   show_skeleton: bool = True,
                   show_confidence: bool = False,
                   show_info: bool = True) -> np.ndarray:
        """Draw all pose results on an image.

        Args:
            image: BGR image (will be copied, not modified in place).
            frame_result: FramePoseResult from PoseEstimator.
            show_bbox: Draw bounding boxes.
            show_keypoints: Draw keypoint circles.
            show_skeleton: Draw skeleton lines.
            show_confidence: Show confidence values next to keypoints.
            show_info: Show frame info (person count, FPS).

        Returns:
            Annotated image copy.
        """
        canvas = image.copy()

        for person in frame_result.persons:
            color = PERSON_COLORS[person.person_id % len(PERSON_COLORS)]

            if show_bbox:
                self._draw_bbox(canvas, person, color)

            if show_skeleton:
                self._draw_skeleton(canvas, person, color)

            if show_keypoints:
                self._draw_keypoints(canvas, person, color, show_confidence)

        if show_info:
            self._draw_info(canvas, frame_result)

        return canvas

    def _draw_bbox(self, canvas: np.ndarray, person, color: tuple) -> None:
        """Draw bounding box with person ID."""
        x1, y1, x2, y2 = person.bbox.astype(int)
        
        # Adaptive bbox thickness based on image size
        h = canvas.shape[0]
        bbox_thickness = max(2, int(h / 500))  # Thicker for larger images
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, bbox_thickness)

        label = f"P{person.person_id} ({person.bbox_score:.2f})"
        
        # Adaptive font size
        font_scale = max(0.4, min(1.2, h / 1000.0))
        thickness = max(1, int(font_scale * 2.5))
        
        (w, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(canvas, (x1, y1 - h_text - 12), (x1 + w + 8, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    def _draw_keypoints(self, canvas: np.ndarray, person, color: tuple,
                        show_confidence: bool) -> None:
        """Draw keypoint circles."""
        for idx, (kp, score) in enumerate(zip(person.keypoints, person.keypoint_scores)):
            if score < self.confidence_threshold:
                continue

            x, y = int(kp[0]), int(kp[1])
            radius = max(3, int(score * 6))  # Larger circle = higher confidence
            cv2.circle(canvas, (x, y), radius, color, -1)
            cv2.circle(canvas, (x, y), radius, (255, 255, 255), 1)

            if show_confidence:
                cv2.putText(canvas, f"{score:.1f}", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    def _draw_skeleton(self, canvas: np.ndarray, person, color: tuple) -> None:
        """Draw skeleton connections between keypoints."""
        # Adaptive line thickness based on image size
        h = canvas.shape[0]
        base_thickness_multiplier = max(2, int(h / 600))
        
        for start_idx, end_idx in self.skeleton:
            start_score = person.keypoint_scores[start_idx]
            end_score = person.keypoint_scores[end_idx]

            if start_score < self.confidence_threshold or end_score < self.confidence_threshold:
                continue

            start_pt = tuple(person.keypoints[start_idx].astype(int))
            end_pt = tuple(person.keypoints[end_idx].astype(int))

            # Line thickness based on confidence AND image size
            avg_conf = (start_score + end_score) / 2
            thickness = max(1, int(avg_conf * base_thickness_multiplier))

            cv2.line(canvas, start_pt, end_pt, color, thickness)

    def _draw_info(self, canvas: np.ndarray, frame_result) -> None:
        """Draw frame information overlay."""
        fps = 1000.0 / frame_result.inference_time_ms if frame_result.inference_time_ms > 0 else 0
        info_lines = [
            f"Frame: {frame_result.frame_id}",
            f"Persons: {frame_result.num_persons}",
            f"Inference: {frame_result.inference_time_ms:.1f}ms ({fps:.1f} FPS)",
        ]

        # Adaptive font size based on image height
        h = canvas.shape[0]
        font_scale = max(0.5, min(1.5, h / 1000.0))  # Scale with image size
        thickness = max(1, int(font_scale * 2))
        outline_thickness = max(2, int(font_scale * 3))

        y_offset = int(30 * font_scale)
        line_spacing = int(35 * font_scale)
        
        for line in info_lines:
            cv2.putText(canvas, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline_thickness)
            cv2.putText(canvas, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            y_offset += line_spacing


class VideoWriter:
    """Write annotated frames to an output video."""

    def __init__(self, output_path: str, fps: float = 30.0,
                 frame_size: tuple[int, int] = (1280, 720)) -> None:
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)
        self.frame_count += 1

    def release(self) -> None:
        self.writer.release()
        print(f"Video saved: {self.output_path} ({self.frame_count} frames)")
