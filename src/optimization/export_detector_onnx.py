"""Export RTMDet person detector to ONNX format."""

import argparse
import json
import os

import torch
import torch.nn as nn
import yaml

try:
    import onnx
    from onnxsim import simplify as onnx_simplify
    SIMPLIFY_AVAILABLE = True
except ImportError:
    SIMPLIFY_AVAILABLE = False


class RTMDetWrapper(nn.Module):
    """Wrap RTMDet backbone+neck+head for ONNX export (pre-NMS outputs)."""

    def __init__(self, det_model):
        super().__init__()
        self.backbone = det_model.backbone
        self.neck = det_model.neck
        self.bbox_head = det_model.bbox_head

    def forward(self, x):
        """
        Args:
            x: (1, 3, 640, 640) — normalized input tensor
        Returns:
            6 tensors: cls_score_0/1/2, bbox_pred_0/1/2
        """
        feats = self.backbone(x)
        feats = self.neck(feats)
        cls_scores, bbox_preds = self.bbox_head(feats)[:2]
        return (*cls_scores, *bbox_preds)


def export_detector(config_path: str, model_variant: str,
                    output_dir: str, simplify: bool = True) -> str:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("Loading MMPose inferencer to access detector model...")
    from mmpose.apis import MMPoseInferencer
    inferencer = MMPoseInferencer(
        pose2d=config["models"][model_variant]["config"],
        device="cuda:0",
    )

    # Access the RTMDet model buried inside MMPoseInferencer
    det_model = inferencer.inferencer.detector.model
    det_model.eval()

    wrapper = RTMDetWrapper(det_model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 640, 640).cuda()
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"detector_{model_variant}.onnx")

    print(f"Exporting to: {onnx_path}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["image"],
            output_names=[
                "cls_score_0", "cls_score_1", "cls_score_2",
                "bbox_pred_0", "bbox_pred_1", "bbox_pred_2",
            ],
            dynamic_axes={"image": {0: "batch_size"}},
        )

    print("ONNX export successful!")

    import onnx as onnx_lib
    model_onnx = onnx_lib.load(onnx_path)
    onnx_lib.checker.check_model(model_onnx)
    print("ONNX verification passed!")

    if simplify and SIMPLIFY_AVAILABLE:
        print("Simplifying ONNX model...")
        simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
        model_simplified, check = onnx_simplify(model_onnx)
        if check:
            onnx_lib.save(model_simplified, simplified_path)
            print(f"Simplified model saved to: {simplified_path}")
            onnx_path = simplified_path
        else:
            print("Simplification failed, using original.")

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Detector model size: {size_mb:.2f} MB")

    # Save metadata needed by the full pipeline decoder
    meta = {
        "input_size": [640, 640],
        "strides": [8, 16, 32],
        "num_classes": 1,
        "mean": [103.53, 116.28, 123.675],
        "std": [57.375, 57.12, 58.395],
    }
    meta_path = onnx_path.replace(".onnx", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")
    return onnx_path


def main(args):
    onnx_path = export_detector(
        config_path=args.config,
        model_variant=args.model,
        output_dir=args.output_dir,
        simplify=args.simplify,
    )
    print(f"\nDetector exported to: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RTMDet detector to ONNX")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--model", type=str, default="lightweight")
    parser.add_argument("--output-dir", type=str, default="models/onnx")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false")
    args = parser.parse_args()
    main(args)
