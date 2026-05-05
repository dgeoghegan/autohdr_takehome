AutoHDR TV Screen Replacement Project

This tool is a set of Python scripts that detects TV and monitor screens in photographs and replaces them with a provided image.

## How it works

Each image in the input_images directory is processed in the following steps:

1. **Detection.** YOLO scans the full image for TVs (class 62). If it finds one above the confidence threshold, the bounding box is padded and passed forward. If YOLO finds nothing, Gemini is called with the full image to attempt bbox detection as a fallback.

2. **Classification.** The detected region is cropped and sent to Gemini with a classification prompt that returns two things: whether a TV is actually present in the crop, and which lighting preset best describes the scene (standard, bright reflection, sharp angle, dim lighting, partial occlusion). Images where Gemini says no TV is present are skipped.

3. **Quad detection.** The crop is processed with OpenCV edge detection using Canny parameters tuned for the classified preset. Contour candidates are filtered by area and approximated to quadrilaterals. Each candidate is unprojected back to full-image pixel coordinates.

4. **Confirmation.** A pink highlight is drawn over the candidate quad and sent to Gemini, which returns whether it looks like a valid TV screen boundary. Candidates with diagonal edges, furniture anchoring, or incomplete coverage are rejected. If all cv2 candidates fail, Gemini is asked directly for quad coordinates as a fallback.

5. **Replacement.** The confirmed quad is used to compute a perspective transform. The beach image is warped to fit the quad and composited into the original image before being copied to output_dir.

The whole loop retries up to 5 times per image before giving up.

## How to run it

### Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
```

Place your replacement image in `./assets` as `replacement.jpg` before running. `replacement-sample.jpg` is provided as an example.

### Run
```bash
python run.py --input_dir ./input_images --output_dir ./output_images [--workers <N>]
```

### Flags

| Flag | Description |
|------|-------------|
| `--input_dir` | Required. Directory of source images. |
| `--output_dir` | Required. Directory for output images. |
| `--workers` | Number of parallel workers (default: 1). Added mid-development when sequential runs were too slow to iterate on. |
| `--evaluate` | Enable Gemini-based post-placement quality evaluation (default: off — unreliable in practice). |
| `--mock` | Use mock Gemini responses. Runs the full pipeline without API calls. (not supported) |
| `--tv_noconfirm` | With --mock only. Forces confirmation to return no, for testing rejection logic. (not supported) |
| `--compare` | Score results against ground truth using IoU. (not supported) |

## Approach and Decisions

My general approach was to build logging, mocking, error handling, and scoring before trying to improve accuracy. Although my metrics for success evolved as I learned more about the domain, I felt I needed a way to measure the effects of tooling and prompt modifications.

Key decisions:

- **YOLO over pure Gemini for detection**, with Gemini as fallback when YOLO finds nothing
- **OpenCV edge detection for perspective-correct quad coordinates**, rather than relying on Gemini's bbox which tends to return a simple rectangle. Getting actual screen corners is what makes the perspective warp look correct rather than just scaling a bbox. This is also why lighting presets matter — Canny parameters that work on a well-lit screen fall apart on a dark or reflective one.
- **Lighting classification step** to select tuned OpenCV presets per scene rather than one set of parameters, with Gemini deciding which preset to apply
- **Confirmation loop** — sending a highlighted quad back to Gemini before committing to a placement
- **Parallel workers** added mid-development to make iteration speed practical
- **Abandoning certain loops and features** like a Gemini-based post-placement quality evaluation that seemed useful but instead increased time and token use without improving accuracy

## Sample Output

| Before | After |
|--------|-------|
| [1001_src.jpg](assets/1001_src.jpg) | [1001_src_replaced.jpg](assets/1001_src_replaced.jpg) |
| [1009_src.jpg](assets/1009_src.jpg) | [1009_src_replaced.jpg](assets/1009_src_replaced.jpg) |
| [102_src.jpg](assets/102_src.jpg) | [102_src_replaced.jpg](assets/102_src_replaced.jpg) |
| [1022_src.jpg](assets/1022_src.jpg) | [1022_src_replaced.jpg](assets/1022_src_replaced.jpg) |
| [1025_src.jpg](assets/1025_src.jpg) | [1025_src_replaced.jpg](assets/1025_src_replaced.jpg) |
| [1250_src.jpg](assets/1250_src.jpg) | [1250_src_replaced.jpg](assets/1250_src_replaced.jpg) |
| [1360_src.jpg](assets/1360_src.jpg) | [1360_src_replaced.jpg](assets/1360_src_replaced.jpg) |
| [1361_src.jpg](assets/1361_src.jpg) | [1361_src_replaced.jpg](assets/1361_src_replaced.jpg) |

## Metrics

The pipeline reports two success metrics. `success_pct` counts any image where a quad was confirmed and a replacement was saved, including cases where the placement is visually wrong. `true_success_rate` counts only images where the saved placement scored IoU ≥ 0.5 against ground truth bboxes, or where a no-TV image was correctly skipped.

"Ground truth" was generated by pixel-diffing the provided src/tar image pairs. The changed region in the tar image marks where the replacement was placed, giving the correct bounding box to score against.

One problem is that ground truth does not generate reliably for all image pairs, so `true_success_rate` undercounts real successes: Images where the placement is correct but no ground truth exists just don't score. The 85% figure cited here is based on visual inspection of outputs, not a clean metric. Complete ground truth coverage across all images would make `true_success_rate` authoritative.

Despite the limitations of these metrics, they did provide a benchmark against which to measure my progress.

## Logging

- `token_usage.jsonl` — one record per Gemini API call with token counts and run_id for aggregation.
- `image_results.jsonl` — one record per image per run with status and failure reason. Successful runs include the IoU score. No-TV images that are correctly skipped are logged as `correctly_no_tv` and counted as successes.
- `run_summary.jsonl` — aggregate counts, total tokens, and runtime per run.

Sample log output is included in `assets/logs/`.

## Module structure

- `run.py` — entry point, ThreadPoolExecutor, per-image orchestration, IoU scoring, stats logging
- `detector.py` — YOLO detection, Gemini bbox fallback, crop classification, cv2 quad detection, Gemini quad fallback, confirmation loop
- `processor.py` — cv2 operations: edge detection presets, contour finding, perspective transform, image compositing
- `evaluator.py` — Gemini-based post-placement quality check (unreliable, off by default)
- `ingestor.py` — image discovery
- `gemini.py` — Gemini API wrapper
- `logger.py` — JSONL logging for token usage, image results, run summaries
- `mock_gemini.py` — fixture responses for all Gemini stages, used with `--mock`

**Development/analysis tools (unsupported):**
- `test_run.py` — runs the pipeline and captures metrics, git diffs, and console logs per run
- `analyze_runs.py` — cross-run metrics analysis
- `analyze_image_costs.py` — links token usage to image results by run_id
- `extract_ground_truth.py` — diffs src/tar image pairs to extract ground truth bboxes

Dependency direction is one-way: run → detector/processor/evaluator → gemini/ingestor → nothing.

Prompts are stored as flat text files in `prompts/` so they can be edited without touching code. Every Gemini call has a corresponding mock fixture so the pipeline can be tested end-to-end without network access.

## Known limitations

**Non-convex placements.** Occasionally the pipeline produces a skewed or triangular composite rather than a clean rectangular fill. This happens when cv2 finds a non-convex contour — typically caused by a window reflection creating a dominant diagonal edge across the TV screen — and the confirmation step passes it anyway. The fix is a `cv2.isContourConvex()` check before the confirm call, which I identified but didn't implement to avoid a late-stage refactor.

**Aspect ratio mismatch.** The replacement image is a fixed asset. When the detected screen has a significantly different aspect ratio the warp stretches or compresses the replacement noticeably. The fix would be cropping the replacement to match the target aspect ratio before warping.

**Debug artifacts.** Crop images, edge maps, and highlight overlays are written to `/tmp/autohdr_crops/` on every run. In production these should be gated behind a `--debug` flag, but I opted not to refactor this late in development.
