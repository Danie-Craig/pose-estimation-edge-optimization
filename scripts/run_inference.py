"""Run pose estimation on images or videos and save results."""

import argparse
import json
import os
import time

import cv2
import numpy as np

from src.data.loader import get_loader
from src.inference.pose_estimator import PoseEstimator
from src.viz.pose_drawer import PoseDrawer, VideoWriter


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
    
    # Check if source is a video
    is_video = args.source.endswith(('.mp4', '.avi', '.mov'))
    
    if is_video:
        print(f"Processing video: {loader.total_frames} frames at {loader.fps:.1f} FPS")
    else:
        print(f"Processing {len(loader)} images")

    # Results storage
    all_results = []
    latencies = []

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup video writer if processing video
    video_writer = None
    if is_video and args.save_video:
        output_video_path = os.path.join(args.output_dir, "output_video.mp4")
        video_writer = VideoWriter(
            output_video_path,
            fps=loader.fps,
            frame_size=(loader.width, loader.height)
        )

    for i, frame_data in enumerate(loader):
        if args.max_frames and i >= args.max_frames:
            break

        # Run pose estimation
        result = estimator.predict(frame_data.image, frame_id=frame_data.frame_id)
        latencies.append(result.inference_time_ms)

        # Store result summary
        all_results.append({
            "frame_id": result.frame_id,
            "source": frame_data.source_path,
            "num_persons": result.num_persons,
            "inference_time_ms": result.inference_time_ms,
            "persons": [
                {
                    "person_id": p.person_id,
                    "mean_confidence": p.mean_confidence,
                    "bbox_score": p.bbox_score,
                }
                for p in result.persons
            ],
        })

        # Create visualization
        vis = drawer.draw_frame(frame_data.image, result)
        
        # Save to video or as individual images
        if video_writer:
            video_writer.write(vis)
        elif i < args.save_vis_count or not is_video:
            os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
            vis_path = os.path.join(args.output_dir, "visualizations", f"frame_{i:04d}.jpg")
            cv2.imwrite(vis_path, vis)

        # Progress
        if (i + 1) % 10 == 0:
            avg_lat = sum(latencies[-10:]) / len(latencies[-10:])
            print(f"  Processed {i+1} frames | Avg latency (last 10): {avg_lat:.1f}ms")

    if video_writer:
        video_writer.release()

    # Summary statistics
    latencies_arr = np.array(latencies)
    summary = {
        "model": args.model,
        "total_frames": len(latencies),
        "avg_latency_ms": float(np.mean(latencies_arr)),
        "median_latency_ms": float(np.median(latencies_arr)),
        "min_latency_ms": float(np.min(latencies_arr)),
        "max_latency_ms": float(np.max(latencies_arr)),
        "avg_fps": float(1000.0 / np.mean(latencies_arr)),
        "avg_persons_per_frame": float(
            np.mean([r["num_persons"] for r in all_results])
        ),
    }

    print(f"\n=== Results Summary ({args.model}) ===")
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "frames": all_results}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose estimation inference")
    parser.add_argument("--source", type=str, required=True,
                        help="Image directory or video file")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--model", type=str, default="lightweight",
                        choices=["lightweight", "accurate"])
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process (None = all)")
    parser.add_argument("--save-vis-count", type=int, default=50,
                        help="How many frames to save as images")
    parser.add_argument("--save-video", action="store_true",
                        help="Save output as video (for video input)")
    parser.add_argument("--output-dir", type=str, default="results/inference")
    args = parser.parse_args()
    main(args)
