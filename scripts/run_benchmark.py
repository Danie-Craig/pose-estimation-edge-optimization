"""Run comprehensive benchmarks comparing PyTorch vs ONNX Runtime vs TensorRT.

Phase 5B/5C — 8-benchmark matrix:
  1. PyTorch Full GPU      (detect + pose, GPU)        ← baseline
  2. PyTorch Full CPU      (detect + pose, CPU)
  3. ONNX Pose GPU         (pose-only component, GPU)
  4. ONNX Pose CPU         (pose-only component, CPU)
  5. ONNX Full GPU         (detect + pose, GPU)
  6. ONNX Full CPU         (detect + pose, CPU)
  7. TensorRT Full FP16    (detect + pose, GPU, FP16)  ← Phase 5C
  8. TensorRT Pose FP16    (pose-only component, FP16) ← Phase 5C
"""

import argparse
import json
import os

import numpy as np

from src.data.loader import get_loader
from src.inference.pose_estimator import PoseEstimator
from src.optimization.benchmark import Benchmarker
from src.optimization.onnx_inference import ONNXPoseEstimator

try:
    from src.optimization.onnx_full_pipeline import ONNXFullPipelineEstimator
    FULL_PIPELINE_AVAILABLE = True
except ImportError:
    FULL_PIPELINE_AVAILABLE = False

try:
    from src.optimization.tensorrt_inference import (
        TRTPoseEstimator, TRTFullPipelineEstimator
    )
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


def main(args: argparse.Namespace) -> None:
    """Run all 8 benchmarks."""
    print("=" * 50)
    print("   Pose Estimation — Benchmark")
    print("=" * 50)

    # ── Load test image ──────────────────────────────────────────────────────
    print(f"\nLoading test image from: {args.test_images}")
    loader = get_loader(args.test_images)
    test_image = next(iter(loader)).image
    print(f"Test image size: {test_image.shape}")

    benchmarker = Benchmarker(warmup_runs=args.warmup, num_runs=args.runs)
    results = []

    # ── BENCHMARK 1: PyTorch Full GPU (Baseline) ─────────────────────────────
    print("\n" + "=" * 70)
    print("BENCHMARK 1: PyTorch Full Pipeline — GPU (Baseline)")
    print("=" * 70)
    est_gpu = PoseEstimator(
        config_path=args.config,
        model_variant=args.model,
        device="cuda",
    )
    results.append(benchmarker.benchmark(
        inference_fn=lambda img: est_gpu.predict(img),
        test_image=test_image,
        name=f"{args.model}_pytorch_full_gpu",
        device="cuda",
    ))

    # ── BENCHMARK 2: PyTorch Full CPU ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BENCHMARK 2: PyTorch Full Pipeline — CPU")
    print("=" * 70)
    est_cpu = PoseEstimator(
        config_path=args.config,
        model_variant=args.model,
        device="cpu",
    )
    results.append(benchmarker.benchmark(
        inference_fn=lambda img: est_cpu.predict(img),
        test_image=test_image,
        name=f"{args.model}_pytorch_full_cpu",
        device="cpu",
    ))

    # ── BENCHMARK 3 & 4: ONNX Pose-Only (GPU + CPU) ──────────────────────────
    pose_onnx_path = f"models/onnx/pose_{args.model}_simplified.onnx"
    if not os.path.exists(pose_onnx_path):
        pose_onnx_path = f"models/onnx/pose_{args.model}.onnx"

    input_size = est_gpu.config["models"][args.model]["input_size"]

    if os.path.exists(pose_onnx_path):
        print("\n" + "=" * 70)
        print("BENCHMARK 3: ONNX Pose-Only — GPU (component)")
        print("=" * 70)
        onnx_pose_gpu = ONNXPoseEstimator(
            onnx_path=pose_onnx_path,
            input_size=input_size,
            providers=["CUDAExecutionProvider"],
        )
        results.append(benchmarker.benchmark(
            inference_fn=lambda img: onnx_pose_gpu.predict(img),
            test_image=test_image,
            name=f"{args.model}_onnx_pose_gpu",
            device="cuda",
        ))

        print("\n" + "=" * 70)
        print("BENCHMARK 4: ONNX Pose-Only — CPU (component)")
        print("=" * 70)
        onnx_pose_cpu = ONNXPoseEstimator(
            onnx_path=pose_onnx_path,
            input_size=input_size,
            providers=["CPUExecutionProvider"],
        )
        results.append(benchmarker.benchmark(
            inference_fn=lambda img: onnx_pose_cpu.predict(img),
            test_image=test_image,
            name=f"{args.model}_onnx_pose_cpu",
            device="cpu",
        ))
    else:
        print(f"\nPose ONNX not found at: {pose_onnx_path}")
        print("   Run: python src/optimization/export_onnx.py first")

    # ── BENCHMARK 5 & 6: ONNX Full Pipeline (GPU + CPU) ──────────────────────
    detector_onnx_path = f"models/onnx/detector_{args.model}_simplified.onnx"
    if not os.path.exists(detector_onnx_path):
        detector_onnx_path = f"models/onnx/detector_{args.model}.onnx"

    if FULL_PIPELINE_AVAILABLE and os.path.exists(detector_onnx_path):
        print("\n" + "=" * 70)
        print("BENCHMARK 5: ONNX Full Pipeline (detect + pose) — GPU")
        print("=" * 70)
        onnx_full_gpu = ONNXFullPipelineEstimator(
            detector_onnx_path=detector_onnx_path,
            pose_onnx_path=pose_onnx_path,
            input_size=input_size,
            providers=["CUDAExecutionProvider"],
        )
        results.append(benchmarker.benchmark(
            inference_fn=lambda img: onnx_full_gpu.predict(img),
            test_image=test_image,
            name=f"{args.model}_onnx_full_gpu",
            device="cuda",
        ))

        print("\n" + "=" * 70)
        print("BENCHMARK 6: ONNX Full Pipeline (detect + pose) — CPU")
        print("=" * 70)
        onnx_full_cpu = ONNXFullPipelineEstimator(
            detector_onnx_path=detector_onnx_path,
            pose_onnx_path=pose_onnx_path,
            input_size=input_size,
            providers=["CPUExecutionProvider"],
        )
        results.append(benchmarker.benchmark(
            inference_fn=lambda img: onnx_full_cpu.predict(img),
            test_image=test_image,
            name=f"{args.model}_onnx_full_cpu",
            device="cpu",
        ))
    else:
        print("\nONNX Full Pipeline benchmarks SKIPPED")
        if not FULL_PIPELINE_AVAILABLE:
            print("   Reason: src/optimization/onnx_full_pipeline.py not yet created")
        else:
            print(f"   Reason: {detector_onnx_path} not found")

    # ── BENCHMARK 7 & 8: TensorRT FP16 ───────────────────────────────────────
    if TRT_AVAILABLE and os.path.exists(detector_onnx_path) and os.path.exists(pose_onnx_path):
        print("\n" + "=" * 70)
        print("BENCHMARK 7: TensorRT Full Pipeline FP16 (detect + pose) — GPU")
        print("First run compiles TRT engines (3–5 min). Cached after that.")
        print("=" * 70)
        trt_full = TRTFullPipelineEstimator(
            detector_onnx_path=detector_onnx_path,
            pose_onnx_path=pose_onnx_path,
            input_size=input_size,
            use_fp16=True,
        )
        results.append(benchmarker.benchmark(
            inference_fn=lambda img: trt_full.predict(img),
            test_image=test_image,
            name=f"{args.model}_trt_full_fp16",
            device="cuda",
        ))

        print("\n" + "=" * 70)
        print("BENCHMARK 8: TensorRT Pose-Only FP16 — GPU (component)")
        print("=" * 70)
        trt_pose = TRTPoseEstimator(
            onnx_path=pose_onnx_path,
            input_size=input_size,
            use_fp16=True,
        )
        results.append(benchmarker.benchmark(
            inference_fn=lambda img: trt_pose.predict(img),
            test_image=test_image,
            name=f"{args.model}_trt_pose_fp16",
            device="cuda",
        ))
    else:
        print("\n TensorRT benchmarks SKIPPED")
        if not TRT_AVAILABLE:
            print("  Reason: src/optimization/tensorrt_inference.py not found")
        else:
            print("  Reason: TensorRT library not available")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("BENCHMARK SUMMARY")
    print("=" * 95)
    print(f"{'Configuration':<35} {'Avg (ms)':<12} {'FPS':<10} "
          f"{'P95 (ms)':<12} {'P99 (ms)':<12} {'Speedup'}")
    print("-" * 95)

    baseline_fps = results[0].fps
    for r in results:
        speedup = r.fps / baseline_fps if baseline_fps > 0 else 0
        print(f"{r.name:<35} {r.avg_latency_ms:<12.2f} {r.fps:<10.2f} "
              f"{r.p95_latency_ms:<12.2f} {r.p99_latency_ms:<12.2f} {speedup:.2f}x")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "test_image_shape": list(test_image.shape),
            "warmup_runs": args.warmup,
            "measurement_runs": args.runs,
            "results": [r.to_dict() for r in results],
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ── Save Markdown table ───────────────────────────────────────────────────
    PIPELINE_LABELS = {
        f"{args.model}_pytorch_full_gpu": ("detect+pose", "GPU (PyTorch)"),
        f"{args.model}_pytorch_full_cpu": ("detect+pose", "CPU (PyTorch)"),
        f"{args.model}_onnx_pose_gpu":    ("pose only",   "GPU (ONNX CUDA)"),
        f"{args.model}_onnx_pose_cpu":    ("pose only",   "CPU (ONNX)"),
        f"{args.model}_onnx_full_gpu":    ("detect+pose", "GPU (ONNX CUDA)"),
        f"{args.model}_onnx_full_cpu":    ("detect+pose", "CPU (ONNX)"),
        f"{args.model}_trt_full_fp16":    ("detect+pose", "GPU (TensorRT FP16)"),
        f"{args.model}_trt_pose_fp16":    ("pose only",   "GPU (TensorRT FP16)"),
    }

    md_path = os.path.join(args.output_dir, "benchmark_table.md")
    with open(md_path, "w") as f:
        f.write("# Pose Estimation Performance Benchmark\n\n")
        f.write(f"**Model**: {args.model}  \n")
        f.write(f"**Hardware**: NVIDIA Tesla T4 (16GB), CUDA 12.2  \n")
        f.write(f"**Runs**: {args.runs} measurement + {args.warmup} warmup\n\n")
        f.write("| Configuration | Pipeline | Runtime/Device | "
                "Avg Latency (ms) | FPS | P95 (ms) | P99 (ms) | Speedup |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            speedup = r.fps / baseline_fps if baseline_fps > 0 else 0
            pipeline, rt_device = PIPELINE_LABELS.get(r.name, ("N/A", "N/A"))
            f.write(f"| {r.name} | {pipeline} | {rt_device} | "
                    f"{r.avg_latency_ms:.2f} | {r.fps:.2f} | "
                    f"{r.p95_latency_ms:.2f} | {r.p99_latency_ms:.2f} | "
                    f"{speedup:.2f}x |\n")

    print(f"Markdown table saved to: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose estimation benchmarks")
    parser.add_argument("--config",      type=str, default="configs/model_config.yaml")
    parser.add_argument("--model",       type=str, default="lightweight",
                        choices=["lightweight", "accurate"])
    parser.add_argument("--test-images", type=str, default="data/coco/val2017")
    parser.add_argument("--warmup",      type=int, default=10)
    parser.add_argument("--runs",        type=int, default=100)
    parser.add_argument("--output-dir",  type=str, default="results/benchmark")
    args = parser.parse_args()
    main(args)
