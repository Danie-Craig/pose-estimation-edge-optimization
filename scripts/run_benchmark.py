"""Run comprehensive benchmarks comparing PyTorch vs ONNX Runtime."""

import argparse
import json
import os

import cv2
import numpy as np
import torch

from src.data.loader import get_loader
from src.inference.pose_estimator import PoseEstimator
from src.optimization.benchmark import Benchmarker
from src.optimization.onnx_inference import ONNXPoseEstimator


def main(args: argparse.Namespace) -> None:
    """Run benchmarks."""
    print("=== Pose Estimation Performance Benchmark ===\n")
    
    # Load test image
    print(f"Loading test image from: {args.test_images}")
    loader = get_loader(args.test_images)
    test_image = next(iter(loader)).image
    print(f"Test image size: {test_image.shape}\n")
    
    # Initialize benchmarker
    benchmarker = Benchmarker(
        warmup_runs=args.warmup,
        num_runs=args.runs,
    )
    
    results = []
    
    # Benchmark 1: PyTorch GPU (Baseline)
    print("="*70)
    print("BENCHMARK 1: PyTorch on GPU (Baseline)")
    print("="*70)
    estimator_gpu = PoseEstimator(
        config_path=args.config,
        model_variant=args.model,
    )
    result_gpu = benchmarker.benchmark(
        inference_fn=lambda img: estimator_gpu.predict(img),
        test_image=test_image,
        name=f"{args.model}_pytorch_gpu",
        device="cuda",
    )
    results.append(result_gpu)
    
    # Benchmark 2: ONNX Runtime GPU (if model exists)
    onnx_path = f"models/onnx/pose_{args.model}.onnx"
    if os.path.exists(onnx_path):
        print("\n" + "="*70)
        print("BENCHMARK 2: ONNX Runtime on GPU")
        print("="*70)
        onnx_gpu = ONNXPoseEstimator(
            onnx_path=onnx_path,
            input_size=estimator_gpu.config["models"][args.model]["input_size"],
            providers=['CUDAExecutionProvider']
        )
        result_onnx_gpu = benchmarker.benchmark(
            inference_fn=lambda img: onnx_gpu.predict(img),
            test_image=test_image,
            name=f"{args.model}_onnx_gpu",
            device="cuda",
        )
        results.append(result_onnx_gpu)
    else:
        print(f"\n⚠ ONNX model not found at: {onnx_path}")
        print("  Run: python src/optimization/export_onnx.py first")
    
    # Benchmark 3: ONNX Runtime CPU
    if os.path.exists(onnx_path):
        print("\n" + "="*70)
        print("BENCHMARK 3: ONNX Runtime on CPU")
        print("="*70)
        onnx_cpu = ONNXPoseEstimator(
            onnx_path=onnx_path,
            input_size=estimator_gpu.config["models"][args.model]["input_size"],
            providers=['CPUExecutionProvider']
        )
        result_onnx_cpu = benchmarker.benchmark(
            inference_fn=lambda img: onnx_cpu.predict(img),
            test_image=test_image,
            name=f"{args.model}_onnx_cpu",
            device="cpu",
        )
        results.append(result_onnx_cpu)
    
    # Summary table
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY")
    print("="*90)
    print(f"{'Configuration':<25} {'Avg (ms)':<12} {'FPS':<10} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Speedup':<10}")
    print("-"*90)
    
    baseline_fps = results[0].fps
    for r in results:
        speedup = r.fps / baseline_fps if baseline_fps > 0 else 0
        print(f"{r.name:<25} {r.avg_latency_ms:<12.2f} {r.fps:<10.2f} {r.p95_latency_ms:<12.2f} {r.p99_latency_ms:<12.2f} {speedup:<10.2f}x")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "benchmark_results.json")
    
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "test_image_shape": test_image.shape,
            "warmup_runs": args.warmup,
            "measurement_runs": args.runs,
            "results": [r.to_dict() for r in results],
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Create markdown table
    md_path = os.path.join(args.output_dir, "benchmark_table.md")
    with open(md_path, "w") as f:
        f.write("# Pose Estimation Performance Benchmark\n\n")
        f.write(f"**Model**: {args.model}\n\n")
        f.write(f"**Test Setup**: {args.runs} runs after {args.warmup} warmup runs\n\n")
        f.write("## Results\n\n")
        f.write("| Configuration | Avg Latency (ms) | FPS | P95 (ms) | P99 (ms) | Speedup |\n")
        f.write("|---------------|------------------|-----|----------|----------|----------|\n")
        for r in results:
            speedup = r.fps / baseline_fps if baseline_fps > 0 else 0
            f.write(f"| {r.name} | {r.avg_latency_ms:.2f} | {r.fps:.2f} | {r.p95_latency_ms:.2f} | {r.p99_latency_ms:.2f} | {speedup:.2f}x |\n")
        
        f.write("\n## Analysis\n\n")
        f.write(f"- **Baseline (PyTorch GPU)**: {results[0].fps:.2f} FPS\n")
        if len(results) > 1:
            speedup = results[1].fps / baseline_fps
            f.write(f"- **ONNX GPU**: {results[1].fps:.2f} FPS ({speedup:.2f}x speedup)\n")
        if len(results) > 2:
            f.write(f"- **ONNX CPU**: {results[2].fps:.2f} FPS (for edge deployment)\n")
    
    print(f"Markdown table saved to: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose estimation benchmarks")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--model", type=str, default="lightweight",
                        choices=["lightweight", "accurate"])
    parser.add_argument("--test-images", type=str, default="data/coco/val2017")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results/benchmark")
    args = parser.parse_args()
    main(args)
