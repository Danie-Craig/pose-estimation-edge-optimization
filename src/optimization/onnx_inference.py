"""ONNX Runtime inference for optimized edge deployment."""

import numpy as np
import onnxruntime as ort
import cv2


class ONNXPoseEstimator:
    """ONNX Runtime inference wrapper for pose estimation."""
    
    def __init__(
        self,
        onnx_path: str,
        input_size: tuple[int, int],
        providers: list[str] = None,
    ) -> None:
        """Initialize ONNX inference session.
        
        Args:
            onnx_path: Path to ONNX model file.
            input_size: Model input size as (width, height).
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        """
        if providers is None:
            # Try GPU first, fall back to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        print(f"Initializing ONNX Runtime session...")
        print(f"  Model: {onnx_path}")
        print(f"  Providers: {providers}")
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get actual provider used
        actual_provider = self.session.get_providers()[0]
        print(f"  Using: {actual_provider}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.
        
        Args:
            image: BGR image (H, W, 3).
            
        Returns:
            Preprocessed tensor (1, 3, H, W).
        """
        # Resize
        resized = cv2.resize(image, self.input_size)
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W)
        transposed = normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a single image.
        
        Args:
            image: BGR image (H, W, 3).
            
        Returns:
            Model output (raw heatmaps or keypoints).
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        return outputs[0]
