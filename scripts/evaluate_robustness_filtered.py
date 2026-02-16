"""Evaluate robustness on filtered person-only images."""

import argparse
import json
import os

import cv2
import numpy as np

from src.data.augmentations import AUGMENTATIONS
from src.inference.pose_estimator import PoseEstimator
from src.viz.pose_drawer import PoseDrawer


def load_person_images(image_dir: str, list_file: str, max_images: int):
    """Load images from filtered list."""
    with open(list_file) as f:
        filenames = [line.strip() for line in f]
    
    filenames = filenames[:max_images]
    
    for filename in filenames:
        path = os.path.join(image_dir, filename)
        image = cv2.imread(path)
        if image is not None:
            yield image, filename


def evaluate_condition(estimator: PoseEstimator, drawer: PoseDrawer,
                       image_dir: str, list_file: str, augmentation_fn,
                       condition_name: str, max_images: int, output_dir: str) -> dict:
    """Evaluate model performance under a specific condition."""
    latencies = []
    person_counts = []
    mean_confidences = []
    detection_count = 0

    vis_dir = os.path.join(output_dir, "vis", condition_name)
    os.makedirs(vis_dir, exist_ok=True)

    for i, (image, filename) in enumerate(load_person_images(image_dir, list_file, max_images)):
        # Apply augmentation
        augmented = augmentation_fn(image)

        # Run inference
        result = estimator.predict(augmented, frame_id=i)

        latencies.append(result.inference_time_ms)
        person_counts.append(result.num_persons)
        
        if result.num_persons > 0:
            detection_count += 1
            for person in result.persons:
                mean_confidences.append(person.mean_confidence)

        # Save first 5 visualizations
        if i < 5:
            vis = drawer.draw_frame(augmented, result)
            vis_path = os.path.join(vis_dir, f"{i:04d}_{filename}")
            cv2.imwrite(vis_path, vis)
        
        # Progress
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{max_images} images...")

    total_images = len(latencies)
    
    return {
        "condition": condition_name,
        "images_processed": total_images,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0,
        "avg_fps": float(1000.0 / np.mean(latencies)) if latencies else 0,
        "avg_persons_detected": float(np.mean(person_counts)) if person_counts else 0,
        "avg_keypoint_confidence": float(np.mean(mean_confidences)) if mean_confidences else 0,
        "detection_rate": float(detection_count / total_images) if total_images > 0 else 0,
        "images_with_detections": detection_count,
    }


def main(args: argparse.Namespace) -> None:
    # Load model
    estimator = PoseEstimator(
        config_path=args.config,
        model_variant=args.model,
    )
    drawer = PoseDrawer(config_path=args.config)

    print(f"Image directory: {args.image_dir}")
    print(f"Person images list: {args.list_file}")
    print(f"Testing up to {args.max_images} person images per condition\n")

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
            estimator, drawer, args.image_dir, args.list_file,
            aug_fn, condition, args.max_images, args.output_dir
        )
        results.append(result)
        
        print(f"  Avg FPS: {result['avg_fps']:.2f}")
        print(f"  Avg confidence: {result['avg_keypoint_confidence']:.3f}")
        print(f"  Detection rate: {result['detection_rate']:.1%}")
        print(f"  Images with detections: {result['images_with_detections']}/{result['images_processed']}\n")

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
    parser = argparse.ArgumentParser(description="Evaluate robustness on person images")
    parser.add_argument("--image-dir", type=str, default="data/coco/val2017",
                        help="Directory containing COCO images")
    parser.add_argument("--list-file", type=str, default="person_images_list.txt",
                        help="File listing person images")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--model", type=str, default="lightweight",
                        choices=["lightweight", "accurate"])
    parser.add_argument("--max-images", type=int, default=2693,
                        help="Max images to test per condition")
    parser.add_argument("--output-dir", type=str, default="results/robustness_coco")
    args = parser.parse_args()
    main(args)
