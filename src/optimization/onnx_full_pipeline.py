"""ONNX Full Pipeline: RTMDet (person detection) + RTMPose (pose estimation)."""

import cv2
import numpy as np
import onnxruntime as ort


class ONNXFullPipelineEstimator:
    """
    Full inference pipeline using ONNX Runtime for both stages.

    Stage 1 — RTMDet: detects people (ONNX, pre-NMS outputs)
    Stage 2 — RTMPose: estimates pose for each person (ONNX)
    NMS is applied in Python (fast, does not significantly affect timing).
    """

    # RTMDet preprocessing constants (BGR, no color conversion)
    DET_INPUT_SIZE = (640, 640)
    DET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    DET_STD  = np.array([57.375,  57.12,  58.395],  dtype=np.float32)
    DET_STRIDES = [8, 16, 32]

    def __init__(
        self,
        detector_onnx_path: str,
        pose_onnx_path: str,
        input_size: tuple,
        providers: list = None,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.65,
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        print(f"Loading detector ONNX: {detector_onnx_path}")
        self.det_session = ort.InferenceSession(
            detector_onnx_path, providers=providers
        )
        self.det_input_name = self.det_session.get_inputs()[0].name
        print(f"  Using: {self.det_session.get_providers()[0]}")

        print(f"Loading pose ONNX: {pose_onnx_path}")
        self.pose_session = ort.InferenceSession(
            pose_onnx_path, providers=providers
        )
        self.pose_input_name = self.pose_session.get_inputs()[0].name
        print(f"  Using: {self.pose_session.get_providers()[0]}")

        self.pose_input_size = input_size   # (width, height)
        self.score_threshold = score_threshold
        self.nms_threshold   = nms_threshold

    # ── Detector preprocessing ────────────────────────────────────────────────

    def _preprocess_for_detector(self, image: np.ndarray):
        """Letterbox-resize to 640×640 and normalize for RTMDet."""
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.DET_INPUT_SIZE

        scale = min(target_h / orig_h, target_w / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        img = cv2.resize(image, (new_w, new_h))

        # Pad with 114 (standard YOLO-family grey padding)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = img

        # Normalize: subtract mean, divide std (stays BGR — RTMDet expects BGR)
        img_f = (padded.astype(np.float32) - self.DET_MEAN) / self.DET_STD

        # HWC → NCHW
        img_f = img_f.transpose(2, 0, 1)[np.newaxis, ...]
        return img_f.astype(np.float32), scale, new_h, new_w

    # ── Detector output decoding ──────────────────────────────────────────────

    def _decode_rtmdet_outputs(
        self,
        det_outputs: list,
        scale: float,
        padded_h: int,
        padded_w: int,
        orig_h: int,
        orig_w: int,
    ) -> np.ndarray:
        """
        Decode RTMDet pre-NMS ONNX outputs to person bboxes.

        det_outputs: 6 arrays  [cls_0, cls_1, cls_2, box_0, box_1, box_2]
          cls_i : (1, 1, H_i, W_i)  raw logits  (person-only model → 1 class)
          box_i : (1, 4, H_i, W_i)  distances in pixels [dl, dt, dr, db]
        """
        cls_scores = det_outputs[:3]
        bbox_preds = det_outputs[3:]

        all_bboxes = []
        all_scores = []

        for cls_score, bbox_pred, stride in zip(
            cls_scores, bbox_preds, self.DET_STRIDES
        ):
            H, W = cls_score.shape[2], cls_score.shape[3]

            # Flatten: (1,1,H,W) → (H*W,)  and  (1,4,H,W) → (H*W,4)
            cls_flat = cls_score[0, 0].ravel()
            box_flat = bbox_pred[0].reshape(4, -1).T  # (H*W, 4)

            # Sigmoid
            scores = 1.0 / (1.0 + np.exp(-cls_flat.astype(np.float64)))

            # Early filter
            mask = scores > self.score_threshold
            if not mask.any():
                continue

            scores   = scores[mask].astype(np.float32)
            box_flat = box_flat[mask]

            # Anchor centers (pixel coords in the 640×640 padded space)
            y_idx, x_idx = np.unravel_index(np.where(mask)[0], (H, W))
            cx = (x_idx + 0.5) * stride
            cy = (y_idx + 0.5) * stride

            # RTMDet dist2bbox: bbox_pred values are already in pixel units
            # (stride was multiplied inside forward_single during training)
            x1 = cx - box_flat[:, 0]
            y1 = cy - box_flat[:, 1]
            x2 = cx + box_flat[:, 2]
            y2 = cy + box_flat[:, 3]

            bboxes = np.stack([x1, y1, x2, y2], axis=1)
            all_bboxes.append(bboxes)
            all_scores.append(scores)

        if not all_bboxes:
            return np.zeros((0, 4), dtype=np.float32)

        all_bboxes = np.concatenate(all_bboxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)

        # Clip to padded image bounds
        all_bboxes[:, 0::2] = all_bboxes[:, 0::2].clip(0, padded_w)
        all_bboxes[:, 1::2] = all_bboxes[:, 1::2].clip(0, padded_h)

        # NMS (Python — fast, not the ONNX bottleneck)
        indices = cv2.dnn.NMSBoxes(
            all_bboxes.tolist(),
            all_scores.tolist(),
            self.score_threshold,
            self.nms_threshold,
        )
        if len(indices) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        kept = all_bboxes[np.array(indices).flatten()]

        # Rescale from padded 640×640 back to original image coords
        kept /= scale
        kept[:, 0::2] = kept[:, 0::2].clip(0, orig_w)
        kept[:, 1::2] = kept[:, 1::2].clip(0, orig_h)
        return kept.astype(np.float32)

    # ── Pose preprocessing ────────────────────────────────────────────────────

    def _preprocess_for_pose(
        self, image: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray:
        """Crop person bbox and prepare for RTMPose."""
        x1, y1, x2, y2 = bbox.astype(int)
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(image.shape[1], x2);  y2 = min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        w, h = self.pose_input_size
        resized    = cv2.resize(crop, (w, h))
        rgb        = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return normalized.transpose(2, 0, 1)[np.newaxis, ...]   # NCHW

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> dict:
        """Detect people → estimate poses.  Returns dict for timing purposes."""
        orig_h, orig_w = image.shape[:2]

        # Stage 1 — Detection
        det_input, scale, padded_h, padded_w = self._preprocess_for_detector(image)
        det_outputs = self.det_session.run(
            None, {self.det_input_name: det_input}
        )

        # Decode + NMS
        bboxes = self._decode_rtmdet_outputs(
            det_outputs, scale, padded_h, padded_w, orig_h, orig_w
        )

        # Stage 2 — Pose (one run per detected person)
        all_keypoints = []
        for bbox in bboxes:
            pose_input = self._preprocess_for_pose(image, bbox)
            if pose_input is None:
                continue
            kp = self.pose_session.run(
                None, {self.pose_input_name: pose_input}
            )
            all_keypoints.append(kp[0])

        return {"num_persons": len(bboxes), "keypoints": all_keypoints}
