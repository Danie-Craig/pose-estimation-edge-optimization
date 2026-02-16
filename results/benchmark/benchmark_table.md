# Pose Estimation Performance Benchmark

**Model**: lightweight

**Test Setup**: 200 runs after 10 warmup runs

## Results

| Configuration | Avg Latency (ms) | FPS | P95 (ms) | P99 (ms) | Speedup |
|---------------|------------------|-----|----------|----------|----------|
| lightweight_pytorch_gpu | 45.42 | 22.02 | 48.10 | 51.67 | 1.00x |
| lightweight_onnx_gpu | 4.38 | 228.52 | 4.68 | 5.37 | 10.38x |
| lightweight_onnx_cpu | 35.63 | 28.07 | 38.18 | 43.05 | 1.27x |

## Analysis

- **Baseline (PyTorch GPU)**: 22.02 FPS
- **ONNX GPU**: 228.52 FPS (10.38x speedup)
- **ONNX CPU**: 28.07 FPS (for edge deployment)
