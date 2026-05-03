# AutoHDR TV Screen Replacement Pipeline

A Python pipeline that detects TV screens in real estate photos and replaces them with a provided image. Built as a take-home project for AutoHDR.

---

## What It Does

Given a folder of source images, the pipeline:

1. Detects TV screen candidates using Gemini Vision
2. Crops the detected region and uses OpenCV to find the screen quadrilateral
3. Draws a pink highlight around the candidate quad and sends the full image back to Gemini to confirm it's actually a TV
4. If confirmed, warps the replacement image to fit the detected screen geometry using a perspective transform
5. Evaluates the final output with a third Gemini pass before saving
6. Logs token usage, per-image results, and run summaries throughout

```bash
python run.py --input_dir ./input_images --output_dir ./output_images
```

---

## How to Build and Run

### Requirements

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Required. Gemini API key. |

Set it before running:

```bash
export GEMINI_API_KEY=your_key_here
```

### CLI Flags

| Flag | Description |
|---|---|
| `--input_dir` | Required. Directory of source images. |
| `--output_dir` | Required. Directory for output images. |
| `--mock` | Use mock Gemini responses instead of real API calls. Useful for testing pipeline logic without burning tokens. |
| `--tv_noconfirm` | Only meaningful with `--mock`. Forces the TV confirmation step to return no, for testing rejection logic. |

---

## Architecture

### Module Structure

```
run.py      # Thin orchestrator. Argparse, loop, nothing else.
detector.py      # All Gemini interactions for detection and confirmation.
processor.py     # All OpenCV work: cropping, quad detection, compositing.
evaluator.py     # Post-replacement quality check via Gemini.
ingestor.py      # Image discovery and path validation.
logger.py        # Structured logging: tokens, per-image results, run summaries.
gemini.py        # Low-level Gemini API client and exception hierarchy.
mock_gemini.py   # Drop-in mock with fixture payloads for each pipeline stage.
```

Dependency direction is one-way: `pipeline` → `detector/processor/evaluator` → `gemini/ingestor` → nothing. No module imports from a layer above it.

### Detection Strategy

Gemini Vision is the primary detection mechanism. It receives the full source image and returns bounding boxes (normalized 0-1000) for TV candidates, along with confidence scores and reasoning.

### The Confirmation Loop

After Gemini returns a bounding box, OpenCV crops the region, detects quadrilateral contours, and finds screen candidates. For each candidate, the quad is drawn in pink on the full source image and sent back to Gemini with the question: "is this highlighted region a complete television screen?"

This second pass catches false positives (fireplaces, reflections, artwork) and partial detections. Only confirmed quads proceed to replacement.

### Perspective Transform

The replacement image is warped to fit the detected quadrilateral using `cv2.getPerspectiveTransform`. Quad points are sorted into consistent top-left, top-right, bottom-right, bottom-left order before the transform to prevent rotation artifacts. The result is composited back onto the original image at full resolution.

### Evaluation Before Save

The pipeline evaluates the composited image with a third Gemini call before writing it to disk. If Gemini determines the replacement was not correctly applied, the output is discarded rather than saved. This keeps the output directory clean and provides a signal for the accuracy metric.

### Mocking

Every external dependency has a mock. `mock_gemini.py` contains fixture payloads for each stage: initial detection, TV confirmation (yes/no/uncertain), and evaluation (success/failure). The `--mock` flag swaps real API calls for fixtures end-to-end, allowing the full pipeline to run without network access or token spend.

---

## Logging

Three log files are written to `logs/`:

**`token_usage.jsonl`** — one record per Gemini API call, with token counts and a `run_id` for aggregation.

**`image_results.jsonl`** — one record per image per run, with status (`success`, `no_tv_detected`, `evaluation_failed`, `gemini_error`, `cv2_no_quad`) and a failure reason where applicable.

**`run_summary.jsonl`** — one record per run, with aggregate counts, total tokens, and runtime.

Token counts are aggregated by `run_id` at the end of each run rather than being passed up through the call stack. This is a pragmatic tradeoff — threading counts through `detect_tvs` → `confirm_tv` → `evaluate_result` would be cleaner but was too invasive given time constraints. A comment in the code marks where this should be refactored.

---

## Failure Modes

**Gemini's bounding boxes are inconsistent run-to-run on the same image. The confirmation loop catches most bad boxes: a poor localization produces a quad Gemini won't confirm. Failed images are logged and skipped rather than retried.

**OpenCV quad sensitivity.** cv2 contour detection is sensitive to lighting, contrast, and competing edges in the crop. High-noise crops (angled ceilings, complex furniture) can produce zero quad candidates or pick the wrong contour. The confirmation step catches most of these, but some images will produce no confirmed quad.

**Partial TV crops.** If Gemini's bounding box clips the TV, OpenCV works with an incomplete outline. Bbox padding (configurable via `BBOX_PADDING_PIX`) reduces this, but large localization errors can't be recovered without a retry.

**Occlusion.** TVs partially obscured by plants, furniture, or people will confuse both Gemini localization and cv2 contour detection. This is expected to be a small percentage of the dataset and is not currently handled.

**Confounders.** Fireplaces, windows, mirrors, and appliance displays can be mistaken for TVs. The detection prompt explicitly names these, and the confirmation prompt requires a complete TV screen. The combination handles most cases.

---

## Production Considerations

These are design decisions for a production deployment. None of this is built.

**Queuing.** At scale, put an SQS queue in front of an ECS service and autoscale on queue depth. The pipeline is stateless by design (no local writes during execution except explicit output artifacts), which makes this straightforward. The rough architecture would be: S3 → SQS → ECS → S3/RDS.

**Scaling surfaces.** Ingestion, Gemini inference, and output writing scale independently. Gemini calls are the bottleneck because they're sequential per image right now. Concurrent processing with bounded parallelism (asyncio or a thread pool) would be the first optimization.

**Kubernetes.** Not the right call yet. ECS or Cloud Run is appropriate at this stage. Kubernetes becomes worth the operational overhead when you have multiple services with different scaling profiles that need independent deployment. A modular monolith that's proven its seams is the right precondition for that conversation.

**Model evolution.** Gemini 2.5 Flash is the current model. If accuracy plateaus before 95%, the next step is fine-tuning a domain-specific model on AutoHDR's labeled dataset rather than continuing to prompt-engineer a general model. The pluggable detection strategy interface is designed to make this swap clean.

**Retry loop.** Detection and confirmation should retry up to N times per image, with a fresh Gemini call each attempt. Gemini's bounding box variability means a second attempt often produces a better result. This is the highest-priority unbuilt feature for accuracy improvement.

---

## What Was Not Built and Why

**Retry loop.** The most impactful missing feature. Architecture supports it — `detect_tvs` would loop up to a configurable max, returning on first confirmed detection. Not built due to time, not due to complexity.

**Containerization.** `requirements.txt` is the source of truth for dependencies. A Dockerfile is straightforward given the stateless design. Not built yet.

**Ground truth accuracy measurement.** The S3 bucket contains before/after pairs (`_src` and `_tar`). A ground truth JSON mapping filenames to expected outcomes would allow automated accuracy scoring against those pairs. The evaluation step produces a per-image signal; the aggregation layer is not built.

**Unit tests.** The mock infrastructure is in place. pytest tests for `_descale_bbox`, `sort_quad_points`, `discover_images`, and the exception hierarchy are the obvious starting points. Not written yet.

**Bezel-aware placement.** The replacement image currently fills the full detected quad including bezel. Nudging quad points slightly inward toward the centroid before compositing would preserve the bezel. One-line change, not yet done.

**Lighting normalization.** The replacement image is pasted without color grading to match room lighting. A post-processing step to match luminance could improve realism. Not in scope for this project.
