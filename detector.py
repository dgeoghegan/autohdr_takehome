# detector.py
import json
from pathlib import Path
from gemini import ask_gemini_vision
from logger import log_token_comment
import cv2
from processor import draw_quad_highlight
from mock_gemini import mock_gemini_vision, mock_confirm_tv, FIXTURE_TV_SINGLE, FIXTURE_TV_QUAD, FIXTURE_CONFIRM_TV_YES, FIXTURE_CONFIRM_TV_NO, MockGenerateContentResponse

DETECT_PROMPT_PATH = "prompts/id_tv_1.txt"
REFINE_PROMPT_PATH = "prompts/id_tv_2.txt"
CONFIRM_PROMPT_PATH = "prompts/confirm_tv.txt"
CONFIRM_THRESHOLD = 0.7
BBOX_PADDING_PIX = 30
MAX_RETRIES = 5
SAVE_CROPS = True

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


def detect_tvs(image_path: str, run_id: str = "", mock: bool = False, fixture: MockGenerateContentResponse = FIXTURE_TV_SINGLE, tv_noconfirm: bool = False) -> list[dict]:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    _, buf = cv2.imencode(".jpg", img)
    image_bytes = buf.tobytes()

    prompt = Path(DETECT_PROMPT_PATH).read_text()
    refine_prompt = Path(REFINE_PROMPT_PATH).read_text()

    confirmed = []

    for attempt in range(MAX_RETRIES):
        # Pass 1: full image, get bbox
        if mock:
            raw, usage = mock_gemini_vision(prompt, image_bytes, fixture)
        else:
            raw, usage = ask_gemini_vision(prompt, image_bytes)
            log_token_comment(f"detect_tv:{Path(image_path).name}", usage, run_id)

        result = json.loads(raw)
        print(f"  Pass 1 raw: {result}")

        for det in result.get("detections", []):
            if det.get("tv_confidence", 0) < 0.7:
                continue

            try:
                bbox = _descale_bbox(det["box_2d"], w, h)
                bbox = _pad_bbox(bbox, BBOX_PADDING_PIX, w, h)
            except (ValueError, KeyError) as e:
                print(f"  Skipping detection, bad box_2d: {det.get('box_2d')} — {e}")
                continue

            # Pass 2: zoomed crop, get quad
            crop_img, crop_bytes = _crop_bytes(img, bbox)
            if SAVE_CROPS:
                crop_debug_dir = Path("photos/crops") / Path(image_path).stem
                crop_debug_dir.mkdir(parents=True, exist_ok=True)
                out_path = str(crop_debug_dir / f"{Path(image_path).stem}_p1_attempt{attempt}.jpg")
                success = cv2.imwrite(out_path, crop_img)
                print(f"  Crop saved: {success} -> {out_path}")

            if mock:
                raw2, usage2 = mock_gemini_vision(refine_prompt, crop_bytes, FIXTURE_TV_QUAD)
            else:
                raw2, usage2 = ask_gemini_vision(refine_prompt, crop_bytes)
                log_token_comment(f"refine_tv:{Path(image_path).name}", usage2, run_id)

            try:
                refine_result = json.loads(raw2)
                quad_norm = refine_result.get("quad_points")
                if not quad_norm or len(quad_norm) != 4:
                    print(f"  Bad quad_points from refine pass: {quad_norm}")
                    continue
                pixel_quad = _unproject_quad(quad_norm, bbox)
            except (ValueError, KeyError, TypeError) as e:
                print(f"  Refine parse error: {e}")
                continue

            highlight_path, highlighted_bytes = draw_quad_highlight(image_path, pixel_quad, attempt)
            print(f"  Highlight saved: {highlight_path}")
            confirmation = confirm_tv(highlighted_bytes, det.get("reasoning", ""), image_path, run_id, mock=mock, tv_noconfirm=tv_noconfirm)
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
