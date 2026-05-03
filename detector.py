# detector.py
import json
from pathlib import Path
from gemini import ask_gemini_vision
from logger import log_token_comment
import cv2
import numpy as np
from processor import save_crop, find_screen_quad, draw_quad_highlight
from mock_gemini import mock_gemini_vision, mock_confirm_tv, FIXTURE_TV_SINGLE, FIXTURE_CONFIRM_TV_YES, FIXTURE_CONFIRM_TV_NO, MockGenerateContentResponse

DETECT_PROMPT_PATH = "prompts/id_tv_1.txt"
CONFIRM_PROMPT_PATH = "prompts/confirm_tv.txt"
CONFIRM_THRESHOLD = 0.7
MAX_RETRIES = 5


def _descale_point(y_norm, x_norm, width, height):
    """Convert Gemini's normalized 0-1000 point to pixel coords."""
    return [int(x_norm / 1000 * width), int(y_norm / 1000 * height)]


def _descale_quad(quad_points: list, width: int, height: int) -> list:
    """Convert list of [y,x] normalized 0-1000 to [[px,py], ...] pixel coords."""
    return [_descale_point(pt[0], pt[1], width, height) for pt in quad_points]


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

    confirmed = []

    for attempt in range(MAX_RETRIES):
        if mock:
            raw, usage = mock_gemini_vision(prompt, image_bytes, fixture)
        else:
            raw, usage = ask_gemini_vision(prompt, image_bytes)
            log_token_comment(f"detect_tv:{Path(image_path).name}", usage, run_id)

        result = json.loads(raw)

        for det in result.get("detections", []):
            if det.get("tv_confidence", 0) < 0.7:
                continue

            quad_norm = det.get("quad_points")
            if not quad_norm or len(quad_norm) != 4:
                print(f"  Skipping detection, bad quad_points: {quad_norm}")
                continue

            try:
                pixel_quad = _descale_quad(quad_norm, w, h)
            except (ValueError, KeyError, TypeError) as e:
                print(f"  Skipping detection, could not descale quad: {quad_norm} — {e}")
                continue

            _, highlighted_bytes = draw_quad_highlight(image_path, pixel_quad, attempt)
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
