# Pose Estimation Performance Benchmark

**Model**: lightweight  
**Hardware**: NVIDIA Tesla T4 (16GB), CUDA 12.2  
**Runs**: 200 measurement + 10 warmup

| Configuration | Pipeline | Runtime/Device | Avg Latency (ms) | FPS | P95 (ms) | P99 (ms) | Speedup |
|---|---|---|---|---|---|---|---|
| lightweight_pytorch_full_gpu | detect+pose | GPU (PyTorch) | 46.37 | 21.57 | 47.95 | 49.74 | 1.00x |
| lightweight_pytorch_full_cpu | detect+pose | CPU (PyTorch) | 875.46 | 1.14 | 906.04 | 944.22 | 0.05x |
| lightweight_onnx_pose_gpu | pose only | GPU (ONNX CUDA) | 4.48 | 223.29 | 5.15 | 5.99 | 10.35x |
| lightweight_onnx_pose_cpu | pose only | CPU (ONNX) | 55.65 | 17.97 | 57.83 | 58.99 | 0.83x |
| lightweight_onnx_full_gpu | detect+pose | GPU (ONNX CUDA) | 41.35 | 24.18 | 42.25 | 42.94 | 1.12x |
| lightweight_onnx_full_cpu | detect+pose | CPU (ONNX) | 648.07 | 1.54 | 669.58 | 702.84 | 0.07x |
| lightweight_trt_full_fp16 | detect+pose | GPU (TensorRT FP16) | 18.43 | 54.24 | 19.73 | 23.03 | 2.52x |
| lightweight_trt_pose_fp16 | pose only | GPU (TensorRT FP16) | 2.08 | 480.64 | 2.17 | 2.26 | 22.29x |
