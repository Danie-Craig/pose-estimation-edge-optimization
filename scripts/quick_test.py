"""Quick sanity check - run MMPose on a single image."""

from mmpose.apis import MMPoseInferencer


def main() -> None:
    # Use the high-level inferencer API for a quick test
    # This auto-downloads a pre-trained model
    inferencer = MMPoseInferencer("human")

    # Run on a sample image
    result_generator = inferencer(
        "data/test_image.jpg",
        show=False,
        out_dir="results/quick_test/",
    )

    # Consume the generator
    results = [r for r in result_generator]
    print(f"Detected {len(results)} frame(s)")

    # Inspect the result structure
    result = results[0]
    predictions = result["predictions"][0]  # First frame
    print(f"Found {len(predictions)} person(s)")

    for i, pred in enumerate(predictions):
        keypoints = pred["keypoints"]
        scores = pred["keypoint_scores"]
        print(f"  Person {i}: {len(keypoints)} keypoints, "
              f"mean confidence: {sum(scores)/len(scores):.3f}")


if __name__ == "__main__":
    main()
