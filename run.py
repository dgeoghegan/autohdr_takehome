# run.py
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, UTC
from pathlib import Path

from ingestor import discover_images
from gemini import GeminiError
from detector import detect_tvs
from processor import replace_screen, save_result, quad_to_bbox
from evaluator import evaluate_result_from_image, iou
from logger import RunStats, log_image_result

REPLACEMENT_PATH = "photos/replacement/autohdr.beach.jpeg"
GROUND_TRUTH_PATH = "photos/ground_truth.json"
MAX_EVAL_RETRIES = 3


def process_image(image_path, run_id, args, ground_truth, stats, stats_lock):
    evaluation = {}
    saved = False

    for eval_attempt in range(MAX_EVAL_RETRIES):
        try:
            detections = detect_tvs(image_path, run_id, mock=args.mock, tv_noconfirm=args.tv_noconfirm)
        except GeminiError as e:
            print(f"  [{Path(image_path).name}] Gemini error: {e}")
            with stats_lock:
                stats.gemini_error += 1
            log_image_result(image_path, "gemini_error", run_id, str(e))
            break

        if not detections:
            gt = ground_truth.get(Path(image_path).name)
            if gt and gt.get("no_tv"):
                log_image_result(image_path, "success", run_id, "correctly_no_tv")
                with stats_lock:
                    stats.successes += 1
            else:
                print(f"  [{Path(image_path).name}] No confirmed TVs detected")
                with stats_lock:
                    stats.no_tv_detected += 1
                log_image_result(image_path, "no_tv_detected", run_id)
            break

        for det in detections:
            if not det.get("quad"):
                with stats_lock:
                    stats.cv2_no_quad += 1
                log_image_result(image_path, "cv2_no_quad", run_id)
                continue

            replaced_img, out_path = replace_screen(image_path, det["quad"], REPLACEMENT_PATH, args.output_dir)

            if args.evaluate:
                evaluation = evaluate_result_from_image(replaced_img, image_path, run_id, mock=args.mock)
                if not evaluation["success"]:
                    print(f"  [{Path(image_path).name}] Eval attempt {eval_attempt+1} failed, retrying...")
                    continue
            
            save_result(replaced_img, out_path)
            with stats_lock:
                stats.successes += 1

            quad_bbox = quad_to_bbox(det["quad"])
            gt = ground_truth.get(Path(image_path).name)
            if gt and gt.get("no_tv"):
                log_image_result(image_path, "false_positive", run_id)
            elif gt and "bbox" in gt:
                gt_bbox = gt.get("bbox")
                score = iou(quad_bbox, gt_bbox)
                print(f"  [{Path(image_path).name}] IoU: {score:.2f}")
                log_image_result(image_path, "success", run_id, f"iou={score:.2f}")
            else:
                log_image_result(image_path, "success", run_id)
            saved = True
            break

        if saved:
            break

    if not saved and args.evaluate:
        with stats_lock:
            stats.evaluation_failed += 1
        log_image_result(image_path, "evaluation_failed", run_id, evaluation.get("reasoning", ""))


def main():
    parser = argparse.ArgumentParser(description="AutoHDR TV screen replacement pipeline")
    parser.add_argument("--input_dir", required=True, help="Directory of source images")
    parser.add_argument("--output_dir", required=True, help="Directory for output images")
    parser.add_argument("--mock", action="store_true", help="Use mock Gemini instead of real API")
    parser.add_argument("--tv_noconfirm", action="store_true", help="Only with --mock. Force Gemini TV confirmation to return no")
    parser.add_argument("--compare", action="store_true", help="Compare results against ground truth bbox using IoU")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--evaluate", action="store_true", help="Run Gemini evaluator on replacements (default: off)")
    args = parser.parse_args()

    image_paths = discover_images(args.input_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    stats = RunStats()
    stats.total_images = len(image_paths)
    stats.run_id = run_id
    stats_lock = threading.Lock()
    print(f"Found {len(image_paths)} images, workers={args.workers}")

    ground_truth = {}
    if args.compare:
        gt_path = Path(GROUND_TRUTH_PATH)
        if gt_path.exists():
            ground_truth = json.loads(gt_path.read_text())
        else:
            print("Warning: --compare set but ground_truth.json not found")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_image, p, run_id, args, ground_truth, stats, stats_lock): p
            for p in image_paths
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                image_path = futures[future]
                print(f"  [{Path(image_path).name}] Unhandled error: {e}")

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
