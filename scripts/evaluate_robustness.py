"""Evaluate pose estimation robustness under challenging conditions.

'Evaluate perception robustness under motion, occlusion, and lighting variability'
"""

import argparse
import json
import os

import cv2
import numpy as np

from src.data.augmentations import AUGMENTATIONS
from src.data.loader import get_loader
from src.inference.pose_estimator import PoseEstimator
from src.viz.pose_drawer import PoseDrawer


def evaluate_condition(estimator: PoseEstimator, drawer: PoseDrawer,
                       loader, augmentation_fn, condition_name: str,
                       max_frames: int, output_dir: str) -> dict:
    """Evaluate model performance under a specific condition."""
    latencies = []
    person_counts = []
    mean_confidences = []

    vis_dir = os.path.join(output_dir, "vis", condition_name)
    os.makedirs(vis_dir, exist_ok=True)

    for i, frame_data in enumerate(loader):
        if i >= max_frames:
            break

        # Apply augmentation
        augmented = augmentation_fn(frame_data.image)

        # Run inference
        result = estimator.predict(augmented, frame_id=i)

        latencies.append(result.inference_time_ms)
        person_counts.append(result.num_persons)
        for person in result.persons:
            mean_confidences.append(person.mean_confidence)

        # Save first 3 visualizations
        if i < 3:
            vis = drawer.draw_frame(augmented, result)
            vis_path = os.path.join(vis_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(vis_path, vis)

    return {
        "condition": condition_name,
        "frames_processed": len(latencies),
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0,
        "avg_fps": float(1000.0 / np.mean(latencies)) if latencies else 0,
        "avg_persons_detected": float(np.mean(person_counts)) if person_counts else 0,
        "avg_keypoint_confidence": float(np.mean(mean_confidences)) if mean_confidences else 0,
        "detection_rate": float(sum(1 for c in person_counts if c > 0) / len(person_counts)) if person_counts else 0,
    }


def main(args: argparse.Namespace) -> None:
    # Load model
    estimator = PoseEstimator(
        config_path=args.config,
        model_variant=args.model,
    )
    drawer = PoseDrawer(config_path=args.config)

    # Load data
    loader = get_loader(args.source)
    print(f"Source: {args.source}")
    print(f"Testing {args.max_frames} frames per condition\n")

    # Test conditions
    conditions_to_test = [
        "clean",
        "motion_blur_light",
        "motion_blur_heavy",
        "low_light_moderate",
        "low_light_severe",
        "overexposure",
        "noise_moderate",
        "noise_heavy",
        "occlusion",
    ]

    results = []
    os.makedirs(args.output_dir, exist_ok=True)

    for condition in conditions_to_test:
        print(f"Testing condition: {condition}")
        aug_fn = AUGMENTATIONS[condition]
        
        result = evaluate_condition(
            estimator, drawer, loader, aug_fn, condition,
            args.max_frames, args.output_dir
        )
        results.append(result)
        
        print(f"  Avg FPS: {result['avg_fps']:.2f}")
        print(f"  Avg confidence: {result['avg_keypoint_confidence']:.3f}")
        print(f"  Detection rate: {result['detection_rate']:.1%}\n")

    # Save results
    results_path = os.path.join(args.output_dir, "robustness_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n=== Robustness Comparison ===")
    print(f"{'Condition':<25} {'FPS':<8} {'Confidence':<12} {'Detection Rate':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['condition']:<25} {r['avg_fps']:<8.2f} {r['avg_keypoint_confidence']:<12.3f} {r['detection_rate']:<15.1%}")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate robustness")
    parser.add_argument("--source", type=str, required=True,
                        help="Image directory or video file")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--model", type=str, default="lightweight",
                        choices=["lightweight", "accurate"])
    parser.add_argument("--max-frames", type=int, default=30,
                        help="Frames to test per condition")
    parser.add_argument("--output-dir", type=str, default="results/robustness")
    args = parser.parse_args()
    main(args)
