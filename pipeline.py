# pipeline.py
import argparse
from ingestor import discover_images
from gemini import GeminiError
from detector import detect_tvs
from processor import save_crops, find_screen_quad, replace_screen

REPLACEMENT_PATH = "photos/replacement/autohdr.beach.jpeg"

def main():
    parser = argparse.ArgumentParser(description="AutoHDR TV screen replacement pipeline")
    parser.add_argument("--input_dir", required=True, help="Directory of source images")
    parser.add_argument("--output_dir", required=True, help="Directory for output images")
    args = parser.parse_args()

    image_paths = discover_images(args.input_dir)
    print(f"Found {len(image_paths)} images")

    for image_path in image_paths:
        print(f"\nProcessing {image_path}")
        try:
            detections = detect_tvs(image_path)
        except GeminiError as e:
            print(f"  Gemini error: {e}")
            continue

        if not detections:
            print("  No TVs detected")
            continue

        crop_paths = save_crops(image_path, detections)
        for path in crop_paths:
            quad = find_screen_quad(path)
            if quad is not None:
                replace_screen(path, quad, REPLACEMENT_PATH, args.output_dir)

if __name__ == "__main__":
    main()
