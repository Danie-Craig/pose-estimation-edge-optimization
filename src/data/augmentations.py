"""Image augmentations to simulate challenging real-world conditions.

These simulate the conditions: motion blur, occlusion, and lighting variability.
"""

import cv2
import numpy as np


def apply_motion_blur(image: np.ndarray, kernel_size: int = 15,
                      angle: float = 0.0) -> np.ndarray:
    """Simulate motion blur from camera or subject movement.

    Args:
        image: Input BGR image.
        kernel_size: Blur kernel size (larger = more blur).
        angle: Direction of motion in degrees.

    Returns:
        Motion-blurred image.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2

    # Create motion blur kernel along the specified angle
    cos_val = np.cos(np.radians(angle))
    sin_val = np.sin(np.radians(angle))

    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * cos_val)
        y = int(center + offset * sin_val)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def apply_low_light(image: np.ndarray, gamma: float = 2.5) -> np.ndarray:
    """Simulate low-light conditions by darkening the image.

    Args:
        image: Input BGR image.
        gamma: Gamma value (>1 = darker, <1 = brighter).

    Returns:
        Darkened image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_overexposure(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Simulate overexposure / bright lighting conditions.

    Args:
        image: Input BGR image.
        factor: Brightness multiplier (>1 = brighter).

    Returns:
        Overexposed image.
    """
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_gaussian_noise(image: np.ndarray, std: float = 25.0) -> np.ndarray:
    """Add Gaussian noise to simulate sensor noise in low light.

    Args:
        image: Input BGR image.
        std: Standard deviation of noise.

    Returns:
        Noisy image.
    """
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def apply_partial_occlusion(image: np.ndarray, num_patches: int = 3,
                            patch_fraction: float = 0.1) -> np.ndarray:
    """Simulate partial occlusion with random black patches.

    Args:
        image: Input BGR image.
        num_patches: Number of occlusion patches.
        patch_fraction: Size of each patch relative to image size.

    Returns:
        Occluded image.
    """
    result = image.copy()
    h, w = image.shape[:2]
    patch_h = int(h * patch_fraction)
    patch_w = int(w * patch_fraction)

    for _ in range(num_patches):
        y = np.random.randint(0, h - patch_h)
        x = np.random.randint(0, w - patch_w)
        result[y:y+patch_h, x:x+patch_w] = 0

    return result


# Registry for easy access from config files
AUGMENTATIONS = {
    "motion_blur_light": lambda img: apply_motion_blur(img, kernel_size=15),
    "motion_blur_heavy": lambda img: apply_motion_blur(img, kernel_size=30),
    "low_light_moderate": lambda img: apply_low_light(img, gamma=2.0),
    "low_light_severe": lambda img: apply_low_light(img, gamma=4.0),
    "overexposure": lambda img: apply_overexposure(img, factor=2.5),
    "noise_moderate": lambda img: apply_gaussian_noise(img, std=50),
    "noise_heavy": lambda img: apply_gaussian_noise(img, std=100),
    "occlusion": lambda img: apply_partial_occlusion(img, num_patches=30),
    "clean": lambda img: img,  # No augmentation (baseline)
}
