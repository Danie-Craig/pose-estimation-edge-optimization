"""Verify the data loader works with test images and videos."""

from src.data.loader import get_loader


def main() -> None:
    print("=== Testing Image Loader ===")
    image_loader = get_loader("data")
    print(f"Found {len(image_loader)} images")

    # Load first 3 images as a test
    for i, frame in enumerate(image_loader):
        print(f"  Frame {frame.frame_id}: {frame.original_size}, source={frame.source_path}")
        if i >= 2:
            break

    print("\n=== Testing Video Loader ===")
    video_loader = get_loader("data/test_video1.mp4")
    print(f"Video: {video_loader.total_frames} frames, {video_loader.fps:.1f} FPS, {video_loader.width}x{video_loader.height}")

    # Load first 5 frames
    for i, frame in enumerate(video_loader):
        print(f"  Frame {frame.frame_id}: {frame.original_size}")
        if i >= 4:
            break

    print("\nData loader is working correctly!")


if __name__ == "__main__":
    main()
