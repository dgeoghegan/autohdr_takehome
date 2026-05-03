# detector.py
import json
from pathlib import Path
from gemini import ask_gemini_vision
from logger import log_token_usage
import cv2
from processor import save_crop, find_screen_quad, draw_quad_highlight
from mock_gemini import mock_gemini_vision, mock_confirm_tv, FIXTURE_TV_SINGLE, FIXTURE_CONFIRM_TV_YES, FIXTURE_CONFIRM_TV_NO, MockGenerateContentResponse
DETECT_PROMPT_PATH = "prompts/id_tv_1.txt"
BBOX_PADDING_PIX = 50
CONFIRM_PROMPT_PATH = "prompts/confirm_tv.txt"
CONFIRM_THRESHOLD = 0.7

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


def confirm_tv(highlighted_bytes: bytes, reasoning: str, mock: bool = False, tv_noconfirm: bool = False) -> dict:
    template = Path(CONFIRM_PROMPT_PATH).read_text()
    prompt = template.replace("{REASONING}", reasoning)

    if mock:
        fixture = FIXTURE_CONFIRM_TV_NO if tv_noconfirm else FIXTURE_CONFIRM_TV_YES
        raw, usage = mock_confirm_tv(prompt, highlighted_bytes, fixture)
    else:
        raw, usage = ask_gemini_vision(prompt, highlighted_bytes)
    
    # note: confirm calls don't log tokens yet — add later
    return json.loads(raw)


def detect_tvs(image_path: str, mock: bool = False, fixture: MockGenerateContentResponse = FIXTURE_TV_SINGLE, tv_noconfirm: bool = False) -> list[dict]:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    _, buf = cv2.imencode(".jpg", img)
    image_bytes = buf.tobytes()

    prompt = Path(DETECT_PROMPT_PATH).read_text()

    if mock:
        raw, usage = mock_gemini_vision(prompt, image_bytes, fixture)
    else:
        raw, usage = ask_gemini_vision(prompt, image_bytes)

    if not mock:
        log_token_usage(image_path, usage)

    result = json.loads(raw)

    confirmed = []
    for i, det in enumerate(result.get("detections", [])):
        if det.get("tv_confidence", 0) < 0.7:
            continue
        try:
            det["bbox"] = _descale_bbox(det["box_2d"], w, h)
            det["bbox"] = _pad_bbox(det["bbox"], BBOX_PADDING_PIX, w, h)
        except (ValueError, KeyError) as e:
            print(f"  Skipping detection, bad box_2d: {det.get('box_2d')} — {e}")
            continue

        crop_path = save_crop(image_path, det, i)

        quads = find_screen_quad(crop_path)
        for quad in quads:
            _, highlighted_bytes = draw_quad_highlight(crop_path, quad)
            confirmation = confirm_tv(highlighted_bytes, det.get("reasoning", ""), mock=mock, tv_noconfirm=tv_noconfirm)
            print(f"  Confirm: is_tv={confirmation['is_tv']} confidence={confirmation['tv_confidence']}")
            if confirmation["is_tv"] and confirmation["tv_confidence"] >= CONFIRM_THRESHOLD:
                det["quad"] = quad.reshape(4, 2).tolist()
                det["confirm_confidence"] = confirmation["tv_confidence"]
                confirmed.append(det)
                break  # stop at first confirmed quad

        if not confirmed or confirmed[-1].get("bbox") != det.get("bbox"):
            print(f"  No confirmed quad for detection in {image_path}")

    return confirmed
