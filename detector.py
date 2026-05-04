# detector.py
import json
from pathlib import Path
from gemini import ask_gemini_vision
from logger import log_token_comment
import cv2
from processor import find_screen_quad, draw_quad_highlight, CV2_PRESETS
from mock_gemini import mock_gemini_vision, mock_confirm_tv, FIXTURE_TV_SINGLE, FIXTURE_TV_QUAD, FIXTURE_CONFIRM_TV_YES, FIXTURE_CONFIRM_TV_NO, FIXTURE_CLASSIFY_STANDARD, MockGenerateContentResponse
from ultralytics import YOLO


DETECT_PROMPT_PATH = "prompts/id_tv_1.txt"
REFINE_PROMPT_PATH = "prompts/id_tv_2.txt"
CONFIRM_PROMPT_PATH = "prompts/confirm_tv.txt"
CLASSIFY_PROMPT_PATH = "prompts/classify_crop.txt"
CONFIRM_THRESHOLD = 0.7
BBOX_PADDING_PIX = 0
MAX_RETRIES = 5
SAVE_CROPS = True
YOLO_MODEL_PATH = "yolov8l.pt"
YOLO_TV_CLASS = 62
YOLO_CONFIDENCE = 0.3
_yolo_model = None

def _descale_bbox(box_2d: list, width: int, height: int) -> dict:
    """Convert Gemini's [ymin, xmin, ymax, xmax] normalized 0-1000 to pixel coords."""
    ymin, xmin, ymax, xmax = box_2d
    return {
        "x1": int(xmin / 1000 * width),
        "y1": int(ymin / 1000 * height),
        "x2": int(xmax / 1000 * width),
        "y2": int(ymax / 1000 * height)
    }


def _pad_bbox(bbox: dict, padding: int, img_w: int, img_h: int) -> dict:
    return {
        "x1": max(0, bbox["x1"] - padding),
        "y1": max(0, bbox["y1"] - padding),
        "x2": min(img_w, bbox["x2"] + padding),
        "y2": min(img_h, bbox["y2"] + padding),
    }


def _crop_bytes(img, bbox: dict) -> tuple:
    """Crop img to bbox, return (crop_img, jpeg_bytes)."""
    crop = img[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
    _, buf = cv2.imencode(".jpg", crop)
    return crop, buf.tobytes()


def _unproject_quad(quad_norm: list, bbox: dict) -> list:
    """Convert crop-space 0-1000 normalized quad to full image pixel coords."""
    crop_w = bbox["x2"] - bbox["x1"]
    crop_h = bbox["y2"] - bbox["y1"]
    return [
        [
            bbox["x1"] + int(pt[1] / 1000 * crop_w),
            bbox["y1"] + int(pt[0] / 1000 * crop_h)
        ]
        for pt in quad_norm
    ]

def _unproject_crop_quad(crop_quad: list, bbox: dict) -> list:
    """Convert cv2 crop-space pixel coords to full image pixel coords."""
    return [
        [bbox["x1"] + pt[0], bbox["y1"] + pt[1]]
        for pt in crop_quad
    ]

def confirm_tv(highlighted_bytes: bytes, reasoning: str, image_path: str, run_id: str = "", mock: bool = False, tv_noconfirm: bool = False) -> dict:
    template = Path(CONFIRM_PROMPT_PATH).read_text()
    prompt = template.replace("{REASONING}", reasoning)

    if mock:
        fixture = FIXTURE_CONFIRM_TV_NO if tv_noconfirm else FIXTURE_CONFIRM_TV_YES
        raw, usage = mock_confirm_tv(prompt, highlighted_bytes, fixture)
    else:
        raw, usage = ask_gemini_vision(prompt, highlighted_bytes)
        log_token_comment(f"confirm_tv:{Path(image_path).name}", usage, run_id)

    return json.loads(raw)

def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model

def detect_tvs(image_path: str, run_id: str = "", mock: bool = False, fixture: MockGenerateContentResponse = FIXTURE_TV_SINGLE, tv_noconfirm: bool = False) -> list[dict]:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    _, buf = cv2.imencode(".jpg", img)
    image_bytes = buf.tobytes()

    refine_prompt = Path(REFINE_PROMPT_PATH).read_text()

    confirmed = []

    # Pass 1: get bboxes
    if mock:
        prompt = Path(DETECT_PROMPT_PATH).read_text()
        raw, usage = mock_gemini_vision(prompt, image_bytes, fixture)
        result = json.loads(raw)
        detections = []
        for det in result.get("detections", []):
            if det.get("tv_confidence", 0) < 0.7:
                continue
            try:
                bbox = _descale_bbox(det["box_2d"], w, h)
                bbox = _pad_bbox(bbox, BBOX_PADDING_PIX, w, h)
                detections.append({"bbox": bbox, "reasoning": det.get("reasoning", ""), "tv_confidence": det.get("tv_confidence", 0.9)})
            except (ValueError, KeyError) as e:
                print(f"  Skipping mock detection: {e}")
    else:
        yolo = _get_yolo()
        yolo_results = yolo(image_path, verbose=False)
        detections = []
        for box in yolo_results[0].boxes:
            if int(box.cls) != YOLO_TV_CLASS:
                continue
            conf = float(box.conf)
            if conf < YOLO_CONFIDENCE:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            print(f"  YOLO TV: conf={conf:.2f} bbox={bbox}")
            padding = max(50, int(max(x2 - x1, y2 - y1) * 0.15))
            bbox = _pad_bbox(bbox, padding, w, h)
            detections.append({"bbox": bbox, "reasoning": f"YOLO conf={conf:.2f}", "tv_confidence": conf})

    # save YOLO debug image regardless of detections
    if SAVE_CROPS:
        debug_img = img.copy()
        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            color = (0, 255, 0) if cls == YOLO_TV_CLASS else (128, 128, 128)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_img, f"cls={cls} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        yolo_debug_dir = Path("photos/crops") / Path(image_path).stem
        yolo_debug_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(yolo_debug_dir / f"{Path(image_path).stem}_yolo_debug.jpg")
        cv2.imwrite(out_path, debug_img)
        print(f"  YOLO debug saved: {out_path}")

    if not detections:
        print(f"  No TV detected in {image_path}")
        return confirmed

    # Pass 2: refine each bbox to a quad, with retries
    for attempt in range(MAX_RETRIES):
        for det in detections:
            bbox = det["bbox"]

            crop_img, crop_bytes = _crop_bytes(img, bbox)

            # save YOLO debug image regardless of detections
            if SAVE_CROPS:
                debug_img = img.copy()
                for box in yolo_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cls = int(box.cls)
                    color = (0, 255, 0) if cls == YOLO_TV_CLASS else (128, 128, 128)
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(debug_img, f"cls={cls} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                yolo_debug_dir = Path("photos/crops") / Path(image_path).stem
                yolo_debug_dir.mkdir(parents=True, exist_ok=True)
                out_path = str(yolo_debug_dir / f"{Path(image_path).stem}_yolo_debug.jpg")
                cv2.imwrite(out_path, debug_img)
                print(f"  YOLO debug saved: {out_path}")

            if not detections:
                print(f"  No TV detected in {image_path}")
                return confirmed

            for attempt in range(MAX_RETRIES):
                for det in detections:
                    bbox = det["bbox"]

                    crop_img, crop_bytes = _crop_bytes(img, bbox)

                    # always save crop — needed for cv2
                    crop_debug_dir = Path("photos/crops") / Path(image_path).stem
                    crop_debug_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = str(crop_debug_dir / f"{Path(image_path).stem}_p1_attempt{attempt}.jpg")
                    success = cv2.imwrite(crop_path, crop_img)
                    print(f"  Crop saved: {success} -> {crop_path}")

                    # Classify crop
                    classify_prompt = Path(CLASSIFY_PROMPT_PATH).read_text()
                    if mock:
                        raw_classify, _ = mock_gemini_vision(classify_prompt, crop_bytes, FIXTURE_CLASSIFY_STANDARD)
                    else:
                        raw_classify, usage_classify = ask_gemini_vision(classify_prompt, crop_bytes)
                        log_token_comment(f"classify_tv:{Path(image_path).name}", usage_classify, run_id)

                    try:
                        classify_result = json.loads(raw_classify)
                    except (ValueError, TypeError) as e:
                        print(f"  Classify parse error: {e}")
                        continue

                    if not classify_result.get("has_tv"):
                        print(f"  Classifier says no TV in crop: {classify_result.get('reasoning')}")
                        continue

                    preset_name = classify_result.get("preset", "standard")
                    if preset_name not in CV2_PRESETS:
                        print(f"  Unknown preset '{preset_name}', falling back to standard")
                        preset_name = "standard"
                    print(f"  Preset: {preset_name} — {classify_result.get('reasoning')}")

                    # cv2 quad detection with preset
                    params = CV2_PRESETS[preset_name]
                    quads = find_screen_quad(crop_path, params)

                    if quads:
                        quad_confirmed = False
                        for candidate_quad in quads:
                            area = cv2.contourArea(candidate_quad)
                            min_area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"]) * 0.1
                            if area < min_area:
                                print(f"  cv2 candidate too small (area={area:.0f}), skipping")
                                continue
                    
                            crop_quad = candidate_quad.reshape(4, 2).tolist()
                            print(f"  crop_quad raw: {crop_quad}")
                            pixel_quad = _unproject_crop_quad(crop_quad, bbox)
                            print(f"  cv2 quad unprojected: {pixel_quad}")
                    
                            highlight_path, highlighted_bytes = draw_quad_highlight(image_path, pixel_quad, attempt)
                            print(f"  Highlight saved: {highlight_path}")
                            confirmation = confirm_tv(highlighted_bytes, classify_result.get("reasoning", ""), image_path, run_id, mock=mock, tv_noconfirm=tv_noconfirm)
                            print(f"  Attempt {attempt+1} confirm: is_tv={confirmation['is_tv']} confidence={confirmation['tv_confidence']}")
                    
                            if confirmation["is_tv"] and confirmation["tv_confidence"] >= CONFIRM_THRESHOLD:
                                det["quad"] = pixel_quad
                                print(f"  Quad: {det['quad']}")
                                det["confirm_confidence"] = confirmation["tv_confidence"]
                                confirmed.append(det)
                                return confirmed
                    
                            print(f"  cv2 candidate rejected by confirm, trying next")
                            quad_confirmed = False

                    else:
                        print("  cv2 found no quad, falling back to Gemini")
                        if mock:
                            raw2, usage2 = mock_gemini_vision(refine_prompt, crop_bytes, FIXTURE_TV_QUAD)
                        else:
                            raw2, usage2 = ask_gemini_vision(refine_prompt, crop_bytes)
                            log_token_comment(f"refine_tv:{Path(image_path).name}", usage2, run_id)

                        try:
                            refine_result = json.loads(raw2)
                            quad_norm = refine_result.get("quad_points")
                            if not quad_norm or len(quad_norm) != 4:
                                print(f"  Bad quad_points from Gemini fallback: {quad_norm}")
                                continue
                            pixel_quad = _unproject_quad(quad_norm, bbox)
                            print(f"  Gemini fallback quad: {pixel_quad}")
                        except (ValueError, KeyError, TypeError) as e:
                            print(f"  Gemini fallback parse error: {e}")
                            continue

                    highlight_path, highlighted_bytes = draw_quad_highlight(image_path, pixel_quad, attempt)
                    print(f"  Highlight saved: {highlight_path}")
                    confirmation = confirm_tv(highlighted_bytes, classify_result.get("reasoning", ""), image_path, run_id, mock=mock, tv_noconfirm=tv_noconfirm)
                    print(f"  Attempt {attempt+1} confirm: is_tv={confirmation['is_tv']} confidence={confirmation['tv_confidence']}")

                    if confirmation["is_tv"] and confirmation["tv_confidence"] >= CONFIRM_THRESHOLD:
                        det["quad"] = pixel_quad
                        print(f"  Quad: {det['quad']}")
                        det["confirm_confidence"] = confirmation["tv_confidence"]
                        confirmed.append(det)
                        return confirmed

                print(f"  Attempt {attempt+1} failed, retrying...")

    if not confirmed:
        print(f"  No confirmed quad after {MAX_RETRIES} attempts for {image_path}")

    return confirmed
