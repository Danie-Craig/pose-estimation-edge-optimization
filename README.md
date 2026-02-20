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

https://github.com/user-attachments/assets/44652a43-b30d-4a9a-8e12-6cdc93174957

https://github.com/user-attachments/assets/af042332-0bbf-4ae6-ba60-a11a8a9de2cc

https://github.com/user-attachments/assets/2e94ffad-993e-4dff-9dd6-3b6c6f4a5dd5

https://github.com/user-attachments/assets/56c1ba81-1d59-41c0-ac30-d5ad06b13854

> All videos run the full RTMPose-m pipeline (person detection + pose estimation) 
> on COCO val2017 images. Keypoints show 17-point COCO skeleton format.
