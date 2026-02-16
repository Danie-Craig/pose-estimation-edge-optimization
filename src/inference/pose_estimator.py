"""Core pose estimation module using MMPose."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from mmpose.apis import MMPoseInferencer


@dataclass
class PoseResult:
    """Structured output for a single detected person's pose."""
    person_id: int
    keypoints: np.ndarray          # Shape: (17, 2) – x, y coordinates
    keypoint_scores: np.ndarray    # Shape: (17,) – confidence per keypoint
    bbox: np.ndarray               # Shape: (4,) – x1, y1, x2, y2
    bbox_score: float              # Detection confidence
    mean_confidence: float         # Average keypoint confidence


@dataclass
class FramePoseResult:
    """All pose results for a single frame."""
    frame_id: int
    persons: list[PoseResult]
    num_persons: int
    inference_time_ms: float = 0.0


class PoseEstimator:
    """Wrapper around MMPose for multi-person pose estimation."""

    def __init__(self, config_path: str = "configs/model_config.yaml",
                 model_variant: str = "lightweight") -> None:
        """Initialize pose estimator.

        Args:
            config_path: Path to model configuration YAML.
            model_variant: Which model to use ('lightweight' or 'accurate').
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        model_config = self.config["models"][model_variant]
        inference_config = self.config["inference"]

        print(f"Loading model: {model_config['name']}")
        print(f"Description: {model_config['description']}")
        print(f"Device: {inference_config['device']}")

        # Initialize MMPose inferencer
        self.inferencer = MMPoseInferencer(
            pose2d=model_config["config"],
            device=inference_config["device"],
        )

        self.confidence_threshold = inference_config["confidence_threshold"]
        self.bbox_threshold = inference_config["bbox_threshold"]
        self.model_name = model_config["name"]

        print(f"Model loaded successfully.")

    def predict(self, image: np.ndarray, frame_id: int = 0) -> FramePoseResult:
        """Run pose estimation on a single image.

        Args:
            image: BGR image as numpy array (H, W, 3).
            frame_id: Frame identifier for tracking.

        Returns:
            FramePoseResult containing all detected persons and their poses.
        """
        import time

        start = time.perf_counter()

        # Run inference
        result_generator = self.inferencer(
            image,
            show=False,
            return_vis=False,
        )
        results = list(result_generator)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Parse results into structured format
        persons = []
        if results and "predictions" in results[0]:
            predictions = results[0]["predictions"][0]  # First (only) frame

            for i, pred in enumerate(predictions):
                keypoints = np.array(pred["keypoints"])         # (17, 2)
                scores = np.array(pred["keypoint_scores"])      # (17,)
                bbox = np.array(pred["bbox"][0]) if "bbox" in pred else np.zeros(4)
                bbox_score = pred.get("bbox_score", 0.0)

                # Filter by bbox confidence
                if bbox_score < self.bbox_threshold:
                    continue

                mean_conf = float(np.mean(scores))

                persons.append(PoseResult(
                    person_id=i,
                    keypoints=keypoints,
                    keypoint_scores=scores,
                    bbox=bbox,
                    bbox_score=float(bbox_score),
                    mean_confidence=mean_conf,
                ))

        return FramePoseResult(
            frame_id=frame_id,
            persons=persons,
            num_persons=len(persons),
            inference_time_ms=elapsed_ms,
        )

    def predict_batch(self, images: list[np.ndarray],
                      start_frame_id: int = 0) -> list[FramePoseResult]:
        """Run pose estimation on multiple images.

        Args:
            images: List of BGR images.
            start_frame_id: Frame ID for the first image.

        Returns:
            List of FramePoseResult, one per image.
        """
        results = []
        for i, image in enumerate(images):
            result = self.predict(image, frame_id=start_frame_id + i)
            results.append(result)
        return results
