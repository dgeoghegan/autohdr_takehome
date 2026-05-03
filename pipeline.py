# pipeline.py
import argparse
from ingestor import discover_images
from gemini import GeminiError
from detector import detect_tvs
from processor import replace_screen

REPLACEMENT_PATH = "photos/replacement/autohdr.beach.jpeg"

def main():
    parser = argparse.ArgumentParser(description="AutoHDR TV screen replacement pipeline")
    parser.add_argument("--input_dir", required=True, help="Directory of source images")
    parser.add_argument("--output_dir", required=True, help="Directory for output images")
    parser.add_argument("--mock", action="store_true", help="Use mock Gemini instead of real API")
    parser.add_argument("--tv_noconfirm", action="store_true", help="Only with --mock. Force Gemini TV confirmatin to return no")
    args = parser.parse_args()

    image_paths = discover_images(args.input_dir)
    print(f"Found {len(image_paths)} images")

    for image_path in image_paths:
        print(f"\nProcessing {image_path}")
        try:
            detections = detect_tvs(image_path, mock=args.mock, tv_noconfirm=args.tv_noconfirm)
        except GeminiError as e:
            print(f"  Gemini error: {e}")
            continue

        if not detections:
            print("  No confirmed TVs detected")
            continue

        for det in detections:
            replace_screen(image_path, det["quad"], REPLACEMENT_PATH, args.output_dir)

if __name__ == "__main__":
    main()
