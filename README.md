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

## Architecture

The pipeline consists of two sequential deep learning models running in series:

```
Input Image → [RTMDet-m] → Person Bounding Boxes → [RTMPose-m] → 17 Keypoints per Person
```

### Stage 1: Person Detection (RTMDet-m)
- **Task**: Locate all people in the frame as bounding boxes
- **Architecture**: Single-stage object detector with CSPNeXt backbone
- **Output**: Bounding boxes with confidence scores per detected person
- **Checkpoint**: `rtmdet_m_8xb32-100e_coco-obj365-person`

### Stage 2: Pose Estimation (RTMPose-m)
- **Task**: Predict 17 keypoint locations per detected person
- **Architecture**: SimCC-based pose estimator with CSPNeXt backbone
- **Keypoints**: COCO 17-point skeleton (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- **Input size**: 192×256 pixels (per cropped person)
- **Checkpoint**: `rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192`

### Framework
- **MMPose 1.3.2** — model loading, inference pipeline, visualization
- **MMDet 3.2.0** — person detector
- **ONNX Runtime 1.23.2** — optimized GPU/CPU inference
- **TensorRT** (via ONNX Runtime TensorrtExecutionProvider) — FP16 acceleration

## Robustness Evaluation

Evaluated on **2,693 person images** from COCO val2017 across 9 conditions
to simulate real-world deployment challenges.

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

**Key finding**: The model maintains **~90% detection rate** under all lighting 
and occlusion conditions. Motion blur and heavy noise are the primary failure 
modes, dropping to **~67% detection rate** under severe degradation.

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

## Failure Analysis

Two conditions cause significant performance degradation — **heavy motion blur** (67.0% 
detection rate) and **heavy noise** (66.7%) — both dropping ~28 percentage points below 
the clean baseline (95.2%).

### Primary Failure Mode 1: Motion Blur

| Metric | Clean | Motion Blur Light | Motion Blur Heavy |
|---|---|---|---|
| Detection Rate | 95.2% | 80.2% | **67.0%** |
| Avg Confidence | 0.563 | 0.499 | **0.389** |
| Images Detected | 2563 / 2693 | 2159 / 2693 | **1804 / 2693** |

**Why it fails**: RTMDet-m was trained on static COCO images without motion blur 
augmentation. Heavy blur destroys the edge features the detector relies on for 
person localization. When the detector misses a person entirely, RTMPose never 
runs — there is no bounding box to crop and feed into the pose stage.

### Primary Failure Mode 2: Heavy Noise

| Metric | Clean | Noise Moderate | Noise Heavy |
|---|---|---|---|
| Detection Rate | 95.2% | 84.2% | **66.7%** |
| Avg Confidence | 0.563 | 0.508 | **0.420** |
| Images Detected | 2563 / 2693 | 2268 / 2693 | **1795 / 2693** |

**Why it fails**: Gaussian noise corrupts pixel-level features uniformly across the 
image. The detector's confidence scores drop below the 0.5 person detection threshold, 
causing valid persons to be filtered out entirely. Unlike motion blur (which degrades 
directional edges), heavy noise corrupts all spatial frequencies simultaneously — 
making it equally damaging to all feature scales.

### Root Cause: Detector Bottleneck

Both failure modes share a common root cause — the **RTMDet-m detector is the 
bottleneck**, not RTMPose. RTMDet-m uses a **CSPNeXt backbone**, whose early 
convolutional layers extract edge and texture features to locate people. Motion 
blur degrades directional edges; heavy noise corrupts all spatial frequencies 
simultaneously — both destroy the low-level features CSPNeXt depends on, causing 
the detector to miss persons entirely before pose estimation can run.

### Potential Mitigations

- **Higher camera shutter speed** — reduces motion blur at the sensor level before inference
- **Blur-aware frame filtering** — detect and skip heavily degraded frames rather than producing unreliable outputs
- **Augmentation-based fine-tuning** — retrain RTMDet-m with motion blur and noise augmentation in the training pipeline
- **Temporal smoothing** — use keypoint tracking (e.g., Kalman filter) to interpolate through missed detections across video frames

## Installation & Setup

### Prerequisites

- Python 3.10
- CUDA 12.2 compatible GPU (tested on NVIDIA Tesla T4)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Danie-Craig/pose-estimation-edge-optimization.git
cd pose-estimation-edge-optimization
```

### 2. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux
```

### 3. Install Dependencies
```bash
pip install --upgrade pip "setuptools<70"
pip install -r requirements.txt
pip install -e .
```
> **Note**: The `-e .` installs the `src/` package in editable mode so all
> modules are importable without path configuration.

### 4. Install MMPose Stack
The OpenMMLab packages require mim for dependency resolution:
```bash
pip install openmim
mim install mmengine
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.2"
```

### 5. Export ONNX Models
ONNX model files are excluded from the repository (too large for Git).
Re-export them after cloning:
```bash
# Export RTMPose to ONNX
python src/optimization/export_onnx.py --model lightweight

# Export RTMDet detector to ONNX
python src/optimization/export_detector_onnx.py --model lightweight
```
Exported models will be saved to `models/onnx/`.

> **Note**: If ONNX export succeeds, `models/onnx/pose_lightweight_simplified.onnx` 
> will be created. If only a `_info.json` file appears, the export fell back to 
> PyTorch inference mode — the benchmark scripts still work via PyTorch.

### 6. TensorRT
TensorRT engines are generated automatically on first run when
TensorrtExecutionProvider is available. No manual export needed —
the engine cache will be saved to `models/trt_cache/`.

> **Note**: COCO dataset, model checkpoints (`.pth`), ONNX models (`.onnx`),
> and TensorRT engines (`.engine`) are excluded from the repository via
> `.gitignore`. TensorRT kernel profile metadata (`.profile`) is committed
> to `models/trt_cache/`. Model weights download automatically from MMPose
> on first inference.

## Usage

### Verify Setup
Confirm the environment is correctly installed before running anything:
```bash
python scripts/verify_setup.py
```
Expected output: `All checks passed. You're ready to start.`

### Run Inference
Run pose estimation on images or video:
```bash
# Image directory
python scripts/run_inference.py --source data/ --output-dir results/inference

# Video file
python scripts/run_inference.py --source data/test_video1.mp4 --save-video --output-dir results/inference
```

Key arguments:
- `--source`          — Directory of `.jpg` images, or `.mp4` video file (required)
- `--model`           — `lightweight` (default) or `accurate`
- `--output-dir`      — Where to save results (default: `results/inference`)
- `--max-frames`      — Limit number of frames processed (default: all)
- `--save-video`      — Save full annotated video as `.mp4` (video input only)
- `--save-vis-count`  — Number of frames to save as individual `.jpg` images when processing video (default: `50`)

**Output:**
- `results/inference/visualizations/` — annotated `.jpg` frames (all frames for image input; first `--save-vis-count` frames for video input)
- `results/inference/output_video.mp4` — full annotated video (only when `--source` is a video **and** `--save-video` is passed)
- `results/inference/results.json` — latency and detection summary (always saved)

### Reproduce Benchmark
Reproduce the 8-configuration performance benchmark:
```bash
python scripts/run_benchmark.py --model lightweight --runs 200
```

Key arguments:
- `--runs` — Number of measurement iterations (default: `100`)
- `--warmup` — Warmup iterations before measurement (default: `10`)
- `--test-images` — Image directory for benchmark input (default: `data/coco/val2017`)
- `--output-dir` — Where to save results (default: `results/benchmark`)

Output: `results/benchmark/benchmark_results.json` and `results/benchmark/benchmark_table.md`.

> **Note**: TensorRT benchmarks (rows 7–8) require ONNX models from Step 5 and a GPU with TensorRT support. They will be skipped with a clear message if unavailable.

### Reproduce Robustness Evaluation
Reproduce the 9-condition robustness evaluation on COCO val2017:
```bash
python scripts/evaluate_robustness_filtered.py --max-images 2693 --output-dir results/robustness_coco
```

Key arguments:
- `--max-images` — Number of person images to evaluate per condition (default: `2693`)
- `--output-dir` — Where to save results and visualizations (default: `results/robustness_coco`)

Output: Per-condition JSON results in `results/robustness_coco/robustness_results.json` and visualization images in `results/robustness_coco/vis/`.

> **Note**: Requires COCO val2017 images at `data/coco/val2017/` and `person_images_list.txt` which is a 
> pre-filtered list of the 2,693 val2017 images containing people, generated from  
> `data/coco/person_keypoints_val2017.json` (excluded from the repository).

## Repository Structure

```
pose-estimation-edge-optimization/
├── configs/
│   └── model_config.yaml                     # RTMPose-m / RTMDet-m config, thresholds, skeleton
├── data/
│   ├── coco/
│   │   ├── val2017/                          # 5,000 COCO val images — not included in repo
│   │   └── person_keypoints_val2017.json     # COCO keypoint annotations — not included in repo
│   ├── test_image0.jpg                       # High-quality test images
│   ├── test_image1.jpg
│   ├── test_video1.mp4                       # Demo videos
│   ├── test_video2.mp4
│   ├── test_video3.mp4
│   └── test_video4.mp4
├── models/
│   ├── onnx/
│   │   └── detector_lightweight_simplified_meta.json   # Export metadata; .onnx files not in repo
│   └── trt_cache/
│       ├── detector_lightweight_simplified/
│       │   └── *.profile                               # TensorRT kernel profile (committed)
│       └── pose_lightweight_simplified/
│           └── *.profile                               # TensorRT kernel profile (committed)
├── results/
│   ├── benchmark/
│   │   ├── benchmark_results.json            # 8-configuration benchmark data
│   │   └── benchmark_table.md                # Formatted benchmark table
│   ├── robustness_coco/                      # 9-condition evaluation on 2,693 COCO person images
│   │   ├── vis/                              # 5 sample visualizations per condition (45 total)
│   │   └── robustness_results.json
│   ├── robustness_images/                    # Early 2-image robustness test
│   │   ├── vis/
│   │   └── robustness_results.json
│   ├── test_images/                          # Inference on custom test images
│   │   ├── visualizations/
│   │   └── results.json
│   ├── test_video1/                          # Inference on demo videos
│   │   ├── output_video.mp4
│   │   └── results.json
│   ├── test_video2/
│   ├── test_video3/
│   └── test_video4/
├── scripts/
│   ├── run_inference.py                      # Run pose estimation on images or video
│   ├── run_benchmark.py                      # 8-configuration performance benchmark
│   ├── evaluate_robustness_filtered.py       # 9-condition robustness on COCO person images
│   ├── filter_person_images.py               # Filter COCO images containing people
│   ├── verify_setup.py                       # Environment verification
│   ├── evaluate_robustness.py                # Early 2-image robustness script
│   ├── quick_test.py                         # Rapid single-image test
│   └── test_data_loader.py                   # Data loader verification
├── src/
│   ├── data/
│   │   ├── loader.py                         # Image, video, and webcam data loaders
│   │   └── augmentations.py                  # 9 robustness augmentations
│   ├── inference/
│   │   └── pose_estimator.py                 # MMPose wrapper with structured outputs
│   ├── optimization/
│   │   ├── benchmark.py                      # Benchmarking framework with percentile metrics
│   │   ├── export_onnx.py                    # RTMPose → ONNX export
│   │   ├── export_detector_onnx.py           # RTMDet → ONNX export
│   │   ├── onnx_inference.py                 # ONNX Runtime pose-only inference
│   │   ├── onnx_full_pipeline.py             # ONNX full pipeline (detect + pose)
│   │   └── tensorrt_inference.py             # TensorRT FP16 inference
│   └── viz/
│       └── pose_drawer.py                    # Skeleton visualization, bounding boxes, video writer
├── .gitignore
├── person_images_list.txt                    # Pre-filtered list of 2,693 COCO person images
├── requirements.txt
├── setup.py
└── README.md
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- **[MMPose](https://github.com/open-mmlab/mmpose)** — pose estimation framework and RTMPose-m model
- **[MMDetection](https://github.com/open-mmlab/mmdetection)** — RTMDet-m person detector
- **[RTMPose](https://arxiv.org/abs/2303.07399)** — Jiang et al., 2023 — the underlying pose estimation architecture
- **[COCO Dataset](https://cocodataset.org/)** — Lin et al., 2014 — evaluation dataset (val2017, 2,693 person images)
- **[ONNX Runtime](https://onnxruntime.ai/)** — cross-platform inference acceleration
- **[TensorRT](https://developer.nvidia.com/tensorrt)** — NVIDIA FP16 inference optimization
