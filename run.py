# pipeline.py
import argparse
import json
from ingestor import discover_images
from gemini import GeminiError
from detector import detect_tvs
from processor import replace_screen, save_result, quad_to_bbox
from evaluator import evaluate_result_from_image, iou
from logger import RunStats, log_image_result
from datetime import datetime, UTC
from pathlib import Path

REPLACEMENT_PATH = "photos/replacement/autohdr.beach.jpeg"
GROUND_TRUTH_PATH = "photos/ground_truth.json"

def main():
    parser = argparse.ArgumentParser(description="AutoHDR TV screen replacement pipeline")
    parser.add_argument("--input_dir", required=True, help="Directory of source images")
    parser.add_argument("--output_dir", required=True, help="Directory for output images")
    parser.add_argument("--mock", action="store_true", help="Use mock Gemini instead of real API")
    parser.add_argument("--tv_noconfirm", action="store_true", help="Only with --mock. Force Gemini TV confirmatin to return no")
    parser.add_argument("--compare", action="store_true", help="Compare results against ground truth bbox using IoU")
    args = parser.parse_args()

    image_paths = discover_images(args.input_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    stats = RunStats()
    stats.total_images = len(image_paths)
    stats.run_id = run_id
    print(f"Found {len(image_paths)} images")

    ground_truth = {}
    if args.compare:
        gt_path = Path(GROUND_TRUTH_PATH)
        if gt_path.exists():
            ground_truth = json.loads(gt_path.read_text())
        else:
            print("Warning: --compare set but ground_truth.json not found")    

    for image_path in image_paths:
        print(f"\nProcessing {image_path}")
        try:
            detections = detect_tvs(image_path, run_id, mock=args.mock, tv_noconfirm=args.tv_noconfirm)
        except GeminiError as e:
            print(f"  Gemini error: {e}")
            stats.gemini_error += 1
            log_image_result(image_path, "gemini_error", run_id, str(e))
            continue

        if not detections:
            print("  No confirmed TVs detected")
            stats.no_tv_detected += 1
            log_image_result(image_path, "no_tv_detected", run_id)
            continue

        for det in detections:
            if not det.get("quad"):
                stats.cv2_no_quad += 1
                log_image_result(image_path, "cv2_no_quad", run_id)
                continue

            replaced_img, out_path = replace_screen(image_path, det["quad"], REPLACEMENT_PATH, args.output_dir)
            evaluation = evaluate_result_from_image(replaced_img, image_path, run_id, mock=args.mock)
            if evaluation["success"]:
                save_result(replaced_img, out_path)
                stats.successes += 1
    
                quad_bbox = quad_to_bbox(det["quad"])
    
                if args.compare:
                    gt = ground_truth.get(Path(image_path).name)
                    if gt:
                        score = iou(quad_bbox, gt)
                        print(f"  IoU: {score:.2f}") 
                        log_image_result(image_path, "success", run_id, f"iou={score:.2f}")
                    else:
                        log_image_result(image_path, "success", run_id)
                else:
                    log_image_result(image_path, "success", run_id)
            else:
                print("   Evaluation failed, discarding output")
                stats.evaluation_failed += 1
                log_image_result(image_path, "evaluation_failed", run_id, evaluation["reasoning"])

    token_log_path = Path("logs/token_usage.jsonl")
    if token_log_path.exists():
        with open(token_log_path) as f:
            for line in f:
                record = json.loads(line)
                if record.get("run_id") == run_id:
                    stats.total_tokens += record.get("total_tokens", 0)
    
    stats.log_summary()

if __name__ == "__main__":
    main()
