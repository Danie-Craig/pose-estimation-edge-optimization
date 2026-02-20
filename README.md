# Real-Time Multi-Person Pose Estimation with Edge Optimization

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)
![MMPose](https://img.shields.io/badge/MMPose-1.3.2-green)
![COCO](https://img.shields.io/badge/Dataset-COCO%202017-red)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-CUDA%2012.2%20%7C%20TensorRT-brightgreen)

A production-ready multi-person pose estimation pipeline using **RTMPose-m**, 
optimized for edge deployment with **ONNX Runtime** and **TensorRT FP16**, 
achieving a **2.52× speedup** over baseline (21.57 FPS → 54.24 FPS).

## Demo

### Pose Estimation on Real-World Scenes

<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/44652a43-b30d-4a9a-8e12-6cdc93174957" width="100%" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/af042332-0bbf-4ae6-ba60-a11a8a9de2cc" width="100%" controls></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/2e94ffad-993e-4dff-9dd6-3b6c6f4a5dd5" width="100%" controls></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/56c1ba81-1d59-41c0-ac30-d5ad06b13854" width="100%" controls></video>
    </td>
  </tr>
</table>

> All videos run the full RTMPose-m pipeline (person detection + pose estimation).
> Keypoints show 17-point COCO skeleton format with per-person confidence scores.

## Performance Benchmark

> **Hardware**: NVIDIA Tesla T4 (16GB) · **Runtime**: CUDA 12.2 · **Runs**: 200 measurement + 10 warmup

| Configuration | Pipeline | Runtime / Device | Avg Latency (ms) | FPS | P95 (ms) | P99 (ms) | Speedup |
|---|---|---|---|---|---|---|---|
| PyTorch Full GPU | detect+pose | GPU (PyTorch) | 46.37 | 21.57 | 47.95 | 49.74 | 1.00× |
| PyTorch Full CPU | detect+pose | CPU (PyTorch) | 875.46 | 1.14 | 906.04 | 944.22 | 0.05× |
| ONNX Full GPU | detect+pose | GPU (ONNX CUDA) | 41.35 | 24.18 | 42.25 | 42.94 | 1.12× |
| ONNX Full CPU | detect+pose | CPU (ONNX) | 648.07 | 1.54 | 669.58 | 702.84 | 0.07× |
| ONNX Pose GPU | pose only | GPU (ONNX CUDA) | 4.48 | 223.29 | 5.15 | 5.99 | 10.35× |
| ONNX Pose CPU | pose only | CPU (ONNX) | 55.65 | 17.97 | 57.83 | 58.99 | 0.83× |
| **TensorRT Full FP16** | **detect+pose** | **GPU (TensorRT FP16)** | **18.43** | **54.24** | **19.73** | **23.03** | **2.52×** |
| TensorRT Pose FP16 | pose only | GPU (TensorRT FP16) | 2.08 | 480.64 | 2.17 | 2.26 | 22.29× |

**Key finding**: TensorRT FP16 is the only configuration that crosses the 
real-time threshold on the full detect+pose pipeline — achieving **54.24 FPS 
(2.52× over PyTorch baseline)** by activating the T4's dedicated Tensor Cores 
for FP16 matrix multiplication.

## Robustness Evaluation

Evaluated on **2,693 person images** from COCO val2017 across 9 conditions
to simulate real-world deployment challenges.

> **Note**: Robustness evaluation was run offline for analysis purposes.
> GPU inference performance is reported in the Performance Benchmark section above.

| Condition | FPS (GPU) | Avg Confidence | Detection Rate | Images Detected |
|---|---|---|---|---|
| Clean (baseline) | 14.19 | 0.563 | **95.2%** | 2563 / 2693 |
| Low Light Moderate | 12.20 | 0.556 | **93.6%** | 2520 / 2693 |
| Overexposure | 11.53 | 0.548 | **90.6%** | 2441 / 2693 |
| Occlusion | 11.27 | 0.529 | **90.3%** | 2432 / 2693 |
| Low Light Severe | 11.93 | 0.529 | **89.5%** | 2411 / 2693 |
| Noise Moderate | 11.74 | 0.508 | **84.2%** | 2268 / 2693 |
| Motion Blur Light | 14.35 | 0.499 | **80.2%** | 2159 / 2693 |
| Noise Heavy | 11.99 | 0.420 | **66.7%** | 1795 / 2693 |
| Motion Blur Heavy | 14.56 | 0.389 | **67.0%** | 1804 / 2693 |

**Key finding**: The model maintains **>90% detection rate** under lighting 
variations and occlusion. Motion blur and heavy noise are the primary failure 
modes, consistent with published robustness benchmarks (COCO-C).

### Visualizations

1 clean baseline and 8 degradation conditions shown on the same scene:

<table>
  <tr>
    <th align="center">Clean</th>
    <th align="center">Motion Blur Light</th>
    <th align="center">Motion Blur Heavy</th>
  </tr>
  <tr>
    <td><img src="results/robustness_coco/vis/clean/0003_000000000885.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/motion_blur_light/0003_000000000885.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/motion_blur_heavy/0003_000000000885.jpg" width="100%"/></td>
  </tr>
  <tr>
    <th align="center">Low Light Moderate</th>
    <th align="center">Low Light Severe</th>
    <th align="center">Overexposure</th>
  </tr>
  <tr>
    <td><img src="results/robustness_coco/vis/low_light_moderate/0003_000000000885.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/low_light_severe/0003_000000000885.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/overexposure/0003_000000000885.jpg" width="100%"/></td>
  </tr>
  <tr>
    <th align="center">Noise Moderate</th>
    <th align="center">Noise Heavy</th>
    <th align="center">Occlusion</th>
  </tr>
  <tr>
    <td><img src="results/robustness_coco/vis/noise_moderate/0003_000000000885.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/noise_heavy/0003_000000000885.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/occlusion/0003_000000000885.jpg" width="100%"/></td>
  </tr>
</table>

<details>
<summary>View some more visualizations (Press here)</summary>

<table>
  <tr>
    <th align="center">Condition</th>
    <th align="center">Image 1</th>
    <th align="center">Image 2</th>
    <th align="center">Image 3</th>
    <th align="center">Image 4</th>
  </tr>
  <tr>
    <td align="center"><b>Clean</b></td>
    <td><img src="results/robustness_coco/vis/clean/0000_000000000139.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/clean/0001_000000000785.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/clean/0002_000000000872.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/clean/0004_000000001000.jpg" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>Motion Blur Light</b></td>
    <td><img src="results/robustness_coco/vis/motion_blur_light/0000_000000000139.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/motion_blur_light/0001_000000000785.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/motion_blur_light/0002_000000000872.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/motion_blur_light/0004_000000001000.jpg" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>Motion Blur Heavy</b></td>
    <td><img src="results/robustness_coco/vis/motion_blur_heavy/0000_000000000139.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/motion_blur_heavy/0001_000000000785.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/motion_blur_heavy/0002_000000000872.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/motion_blur_heavy/0004_000000001000.jpg" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>Low Light Moderate</b></td>
    <td><img src="results/robustness_coco/vis/low_light_moderate/0000_000000000139.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/low_light_moderate/0001_000000000785.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/low_light_moderate/0002_000000000872.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/low_light_moderate/0004_000000001000.jpg" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>Low Light Severe</b></td>
    <td><img src="results/robustness_coco/vis/low_light_severe/0000_000000000139.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/low_light_severe/0001_000000000785.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/low_light_severe/0002_000000000872.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/low_light_severe/0004_000000001000.jpg" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>Overexposure</b></td>
    <td><img src="results/robustness_coco/vis/overexposure/0000_000000000139.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/overexposure/0001_000000000785.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/overexposure/0002_000000000872.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/overexposure/0004_000000001000.jpg" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>Noise Moderate</b></td>
    <td><img src="results/robustness_coco/vis/noise_moderate/0000_000000000139.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/noise_moderate/0001_000000000785.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/noise_moderate/0002_000000000872.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/noise_moderate/0004_000000001000.jpg" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>Noise Heavy</b></td>
    <td><img src="results/robustness_coco/vis/noise_heavy/0000_000000000139.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/noise_heavy/0001_000000000785.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/noise_heavy/0002_000000000872.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/noise_heavy/0004_000000001000.jpg" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>Occlusion</b></td>
    <td><img src="results/robustness_coco/vis/occlusion/0000_000000000139.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/occlusion/0001_000000000785.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/occlusion/0002_000000000872.jpg" width="100%"/></td>
    <td><img src="results/robustness_coco/vis/occlusion/0004_000000001000.jpg" width="100%"/></td>
  </tr>
</table>

</details>