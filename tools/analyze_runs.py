#!/usr/bin/env python3
# analyze_runs.py
import json
from pathlib import Path
from collections import defaultdict

gt = json.loads(Path("ground_truth.json").read_text())

summaries = {}
summary_path = Path("logs/run_summary.jsonl")
if summary_path.exists():
    with open(summary_path) as f:
        for line in f:
            r = json.loads(line)
            summaries[r["run_id"]] = r

runs = defaultdict(lambda: {"no_tv_in_image": [], "yolo_missed": [], "iou_scores": [], "zero_iou": []})

with open("logs/image_results.jsonl") as f:
    for line in f:
        r = json.loads(line)
        run_id = r.get("run_id")
        if not run_id:
            continue
        name = r["image"]
        status = r["status"]
        reason = r.get("reason", "")

        if gt.get(name, {}).get("no_tv"):
            if status in ("success", "correctly_no_tv"):
                runs[run_id]["no_tv_in_image"].append(name)
        elif status == "no_tv_detected":
            runs[run_id]["yolo_missed"].append(name)
        elif status == "success" and reason.startswith("iou="):
            score = float(reason.split("=")[1])
            if score > 0:
                runs[run_id]["iou_scores"].append(score)
            else:
                runs[run_id]["zero_iou"].append(name)

print(f"{'run_id':<20} {'total':>5} {'no_tv':>6} {'miss':>5} {'placed':>7} {'avg_iou':>8} {'zero':>5} {'tok/img':>8} {'tok/win':>8}")
print("-" * 95)

for run_id, buckets in sorted(runs.items()):
    s = summaries.get(run_id, {})
    total = s.get("total_images", 0)
    tokens = s.get("total_tokens", 0)
    iou = buckets["iou_scores"]
    placed = len(iou)
    avg = f"{sum(iou)/len(iou):.2f}" if iou else "—"
    tok_per_image = str(round(tokens / total)) if total else "—"
    tok_per_success = str(round(tokens / placed)) if placed else "—"
    print(f"{run_id:<20} {total:>5} {len(buckets['no_tv_in_image']):>6} {len(buckets['yolo_missed']):>5} {placed:>7} {avg:>8} {len(buckets['zero_iou']):>5} {tok_per_image:>8} {tok_per_success:>8}")
