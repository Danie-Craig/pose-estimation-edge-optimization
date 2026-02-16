"""Filter COCO images that contain people from annotation file."""

import json
import argparse
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    # Load COCO annotations
    print(f"Loading annotations from {args.annotations}")
    with open(args.annotations) as f:
        coco = json.load(f)
    
    # Get all image IDs that have person annotations
    person_image_ids = set()
    
    for ann in coco['annotations']:
        # COCO category_id 1 = person
        if ann['category_id'] == 1:
            person_image_ids.add(ann['image_id'])
    
    # Get image filenames
    person_images = []
    for img in coco['images']:
        if img['id'] in person_image_ids:
            person_images.append(img['file_name'])
    
    print(f"\nFound {len(person_images)} images with people (out of {len(coco['images'])} total)")
    
    # Save to file
    output_file = args.output
    with open(output_file, 'w') as f:
        for filename in sorted(person_images):
            f.write(filename + '\n')
    
    print(f"Saved person image list to {output_file}")
    print(f"\nTo copy these images, use the list file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter COCO images with people")
    parser.add_argument("--annotations", type=str, required=True,
                        help="Path to person_keypoints_val2017.json")
    parser.add_argument("--output", type=str, default="person_images.txt",
                        help="Output file listing person images")
    args = parser.parse_args()
    main(args)
