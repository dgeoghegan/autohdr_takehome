#!/usr/bin/env python3
# test_run.py
import argparse
import subprocess
import json
import shutil
from datetime import datetime, UTC
from pathlib import Path

INPUT_DIR = "./input_images"
OUTPUT_DIR = "./output_dir"
TEST_LOG = "logs/test_runs.jsonl"
IOU_THRESHOLD = 0.5

def main():
    parser = argparse.ArgumentParser(description="Test runner for run.py")
    parser.add_argument("-m", "--message", default="", help="Note about what you're testing")
    parser.add_argument("-s", "--save", action="store_true", help="Save output dir to photos/<timestamp>")
    parser.add_argument("-n", "--no_log", action="store_true", help="Skip writing to test log")
    args = parser.parse_args()

    run_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")

    # if --save, move existing output to photos/ before clearing
    if args.save:
        output_path = Path(OUTPUT_DIR)
        if output_path.exists() and any(output_path.iterdir()):
            save_path = Path("photos") / run_timestamp
            shutil.move(str(output_path), str(save_path))
            print(f"Previous output saved to {save_path}")
    else:
        for dir_path in [Path(OUTPUT_DIR), Path("photos/crops")]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True)

    # clear crops
    crops_path = Path("photos/crops")
    if crops_path.exists():
        shutil.rmtree(crops_path)
    crops_path.mkdir(parents=True)

    # capture git diff
    diff_result = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True)
    diff_dir = Path("logs/diffs")
    diff_dir.mkdir(parents=True, exist_ok=True)
    diff_path = diff_dir / f"{run_timestamp}.diff"
    diff_path.write_text(diff_result.stdout)

    # run pipeline
    result = subprocess.run(["python", "run.py", "--input_dir", INPUT_DIR, "--output_dir", OUTPUT_DIR, "--compare", "--workers", "6"])

    # pull run summary from last line
    run_summary = {}
    summary_path = Path("logs/run_summary.jsonl")
    if summary_path.exists():
        last_line = summary_path.read_text().strip().splitlines()[-1]
        run_summary = json.loads(last_line)
    total = run_summary.get("total_images", 0)
    successes = run_summary.get("successes", 0)
    success_pct = round(successes / total * 100, 1) if total else 0
    run_id = run_summary.get("run_id", "")
    
    ground_truth_successes = 0
    iou_scores = []
    
    result_log_path = Path("logs/image_results.jsonl")
    if result_log_path.exists():
        with open(result_log_path) as f:
            for line in f:
                record = json.loads(line)
                if record.get("run_id") != run_id:
                    continue
                if record.get("status") != "success":
                    continue
                reason = record.get("reason", "")
                if reason.startswith("iou="):
                    try:
                        score = float(reason.split("=")[1])
                        iou_scores.append(score)
                        if score >= IOU_THRESHOLD:
                            ground_truth_successes += 1
                    except ValueError:
                        pass
    
    true_success_rate = round(ground_truth_successes / total * 100, 1) if total else 0
    avg_iou = round(sum(iou_scores) / len(iou_scores), 2) if iou_scores else 0
    if not args.no_log:
        # write test log
        Path(TEST_LOG).parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": run_timestamp,
            "message": args.message,
            "diff_file": str(diff_path),
            "exit_code": result.returncode,
            "total_images": total,
            "successes": successes,
            "success_pct": success_pct,
            "true_success_rate": true_success_rate,
            "avg_iou": avg_iou,
            "iou_scores": iou_scores,
            "total_tokens": run_summary.get("total_tokens", 0),
            "runtime_seconds": run_summary.get("runtime_seconds", 0),
            "no_tv_detected": run_summary.get("no_tv_detected", 0),
            "confirmation_failed": run_summary.get("confirmation_failed", 0),
            "evaluation_failed": run_summary.get("evaluation_failed", 0),
            "gemini_error": run_summary.get("gemini_error", 0),
            "cv2_no_quad": run_summary.get("cv2_no_quad", 0),
            "ground_truth_successes": ground_truth_successes,
            "iou_threshold": IOU_THRESHOLD,
        }
        with open(TEST_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    true_success_rate = round(ground_truth_successes / total * 100, 1) if total else 0
    print(f"\nResults: {successes}/{total} ({success_pct}%) true={true_success_rate}% avg_iou={avg_iou} gt_successes={ground_truth_successes}/{total} — ...")

if __name__ == "__main__":
    main()
