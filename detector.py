# detector.py
import json
from pathlib import Path
from gemini import ask_gemini_vision
from logger import log_token_usage
from mock_gemini import mock_gemini_vision, FIXTURE_TV_SINGLE, MockGenerateContentResponse
import cv2

PROMPT_PATH = "prompts/id_tv_1.txt"
BBOX_PADDING_PIX = 50

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

from mock_gemini import mock_gemini_vision, FIXTURE_TV_SINGLE, MockGenerateContentResponse

def detect_tvs(image_path: str, mock: bool = False, fixture: MockGenerateContentResponse = FIXTURE_TV_SINGLE) -> list[dict]:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    _, buf = cv2.imencode(".jpg", img)
    image_bytes = buf.tobytes()

    prompt = Path(PROMPT_PATH).read_text()

    if mock:
        raw, usage = mock_gemini_vision(prompt, image_bytes, fixture)
    else:
        raw, usage = ask_gemini_vision(prompt, image_bytes)

    if not mock:
        log_token_usage(image_path, usage)

    result = json.loads(raw)

    detections = []
    for det in result.get("detections", []):
        try:
            det["bbox"] = _descale_bbox(det["box_2d"], w, h)
            det["bbox"] = _pad_bbox(det["bbox"], BBOX_PADDING_PIX, w, h)
            detections.append(det)
        except (ValueError, KeyError) as e:
            print(f"  Skipping detection, bad box_2d: {det.get('box_2d')} — {e}")

    return detections
