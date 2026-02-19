"""TensorRT-accelerated inference via ONNX Runtime TensorrtExecutionProvider."""

import os
import cv2
import numpy as np
import onnxruntime as ort


def build_trt_session(onnx_path: str, use_fp16: bool = True,
                      cache_dir: str = "models/trt_cache") -> ort.InferenceSession:
    """
    Build an ONNX Runtime session backed by TensorRT.

    First call: TensorRT compiles an optimised engine (~1-3 min).
    Subsequent calls: loads cached engine instantly.
    """
    os.makedirs(cache_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    engine_cache = os.path.join(cache_dir, model_name)

    trt_options = {
        # FP16 — uses T4 Tensor Cores (8x throughput vs FP32)
        "trt_fp16_enable": use_fp16,
        # Cache compiled engine so we only wait once
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": engine_cache,
        # Workspace memory (2 GB — T4 has 16 GB)
        "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
    }

    providers = [
        ("TensorrtExecutionProvider", trt_options),
        ("CUDAExecutionProvider", {}),   # fallback for unsupported ops
    ]

    print(f"Building TensorRT session for: {onnx_path}")
    print(f"  FP16: {use_fp16}  |  Cache: {engine_cache}")
    print("  (First run compiles engine — takes 1-3 min, cached after that)")

    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"  Active provider: {session.get_providers()[0]}")
    return session


class TRTPoseEstimator:
    """Pose-only estimator using TensorRT (component benchmark)."""

    def __init__(self, onnx_path: str, input_size: tuple,
                 use_fp16: bool = True):
        self.session = build_trt_session(onnx_path, use_fp16=use_fp16)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size   # (width, height)

    def predict(self, image: np.ndarray) -> np.ndarray:
        w, h = self.input_size
        resized    = cv2.resize(image, (w, h))
        rgb        = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        tensor     = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        return self.session.run(None, {self.input_name: tensor})[0]


class TRTFullPipelineEstimator:
    """Full pipeline (detect + pose) using TensorRT for both models."""

    DET_INPUT_SIZE = (640, 640)
    DET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    DET_STD  = np.array([57.375,  57.12,  58.395],  dtype=np.float32)
    DET_STRIDES = [8, 16, 32]

    def __init__(self, detector_onnx_path: str, pose_onnx_path: str,
                 input_size: tuple, use_fp16: bool = True,
                 score_threshold: float = 0.3, nms_threshold: float = 0.65):

        self.det_session  = build_trt_session(detector_onnx_path, use_fp16)
        self.pose_session = build_trt_session(pose_onnx_path,     use_fp16)
        self.det_input_name  = self.det_session.get_inputs()[0].name
        self.pose_input_name = self.pose_session.get_inputs()[0].name
        self.pose_input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold   = nms_threshold

    def _preprocess_detector(self, image):
        orig_h, orig_w = image.shape[:2]
        scale = min(640 / orig_h, 640 / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        img = cv2.resize(image, (new_w, new_h))
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = img
        img_f = (padded.astype(np.float32) - self.DET_MEAN) / self.DET_STD
        return img_f.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32), \
               scale, new_h, new_w

    def _decode_and_nms(self, det_outputs, scale, ph, pw, oh, ow):
        all_boxes, all_scores = [], []
        for cls_s, box_p, stride in zip(
                det_outputs[:3], det_outputs[3:], self.DET_STRIDES):
            H, W = cls_s.shape[2], cls_s.shape[3]
            cls_f = cls_s[0, 0].ravel()
            box_f = box_p[0].reshape(4, -1).T
            scores = 1.0 / (1.0 + np.exp(-cls_f.astype(np.float64)))
            mask = scores > self.score_threshold
            if not mask.any():
                continue
            yi, xi = np.unravel_index(np.where(mask)[0], (H, W))
            cx = (xi + 0.5) * stride
            cy = (yi + 0.5) * stride
            bf = box_f[mask]
            boxes = np.stack([cx-bf[:,0], cy-bf[:,1],
                              cx+bf[:,2], cy+bf[:,3]], axis=1)
            all_boxes.append(boxes)
            all_scores.append(scores[mask].astype(np.float32))
        if not all_boxes:
            return np.zeros((0, 4), dtype=np.float32)
        all_boxes  = np.concatenate(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_boxes[:, 0::2] = all_boxes[:, 0::2].clip(0, pw)
        all_boxes[:, 1::2] = all_boxes[:, 1::2].clip(0, ph)
        idx = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(),
                               self.score_threshold, self.nms_threshold)
        if len(idx) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        kept = all_boxes[np.array(idx).flatten()] / scale
        kept[:, 0::2] = kept[:, 0::2].clip(0, ow)
        kept[:, 1::2] = kept[:, 1::2].clip(0, oh)
        return kept.astype(np.float32)

    def _preprocess_pose(self, image, bbox):
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        w, h = self.pose_input_size
        rgb = cv2.cvtColor(cv2.resize(crop, (w, h)), cv2.COLOR_BGR2RGB)
        return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

    def predict(self, image: np.ndarray) -> dict:
        orig_h, orig_w = image.shape[:2]
        det_in, scale, ph, pw = self._preprocess_detector(image)
        det_out = self.det_session.run(None, {self.det_input_name: det_in})
        bboxes  = self._decode_and_nms(det_out, scale, ph, pw, orig_h, orig_w)
        kps = []
        for bbox in bboxes:
            pose_in = self._preprocess_pose(image, bbox)
            if pose_in is not None:
                kps.append(self.pose_session.run(
                    None, {self.pose_input_name: pose_in})[0])
        return {"num_persons": len(bboxes), "keypoints": kps}
