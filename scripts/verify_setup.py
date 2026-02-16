"""Verify that all dependencies are installed correctly."""

import sys


def check_import(module_name: str, display_name: str | None = None) -> bool:
    """Try importing a module and print status."""
    name = display_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  [OK] {name} ({version})")
        return True
    except ImportError as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def main() -> None:
    print("=== Checking Python ===")
    print(f"  Python {sys.version}")

    print("\n=== Checking GPU ===")
    import torch
    if torch.cuda.is_available():
        print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  [OK] CUDA version: {torch.version.cuda}")
    else:
        print("  [WARN] CUDA not available — will run on CPU (slow)")

    print("\n=== Checking Core Libraries ===")
    all_ok = True
    for mod in ["torch", "torchvision", "cv2", "numpy", "matplotlib", "yaml", "tqdm"]:
        all_ok &= check_import(mod)

    print("\n=== Checking MMPose Stack ===")
    for mod, name in [("mmengine", "mmengine"), ("mmcv", "mmcv"), ("mmdet", "mmdet"), ("mmpose", "mmpose")]:
        all_ok &= check_import(mod, name)

    print("\n=== Checking ONNX ===")
    for mod in ["onnx", "onnxruntime"]:
        all_ok &= check_import(mod)

    print()
    if all_ok:
        print("✅ All checks passed. You're ready to start.")
    else:
        print("❌ Some checks failed. Fix the issues above before proceeding.")


if __name__ == "__main__":
    main()
