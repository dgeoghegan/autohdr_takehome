# evaluator.py
import json
from pathlib import Path
from gemini import ask_gemini_vision
from mock_gemini import mock_confirm_tv, FIXTURE_EVALUATE_SUCCESS, FIXTURE_EVALUATE_FAILURE
from logger import log_token_comment
import cv2

EVALUATE_PROMPT_PATH = "prompts/evaluate_result.txt"

def evaluate_result(image_path: str, mock: bool = False, force_fail: bool = False) -> dict:
    img = cv2.imread(image_path)
    _, buf = cv2.imencode(".jpg", img)
    image_bytes = buf.tobytes()

    prompt = Path(EVALUATE_PROMPT_PATH).read_text()

    if mock:
        fixture = FIXTURE_EVALUATE_FAILURE if force_fail else FIXTURE_EVALUATE_SUCCESS
        raw, usage = mock_confirm_tv(prompt, image_bytes, fixture)
    else:
        raw, usage = ask_gemini_vision(prompt, image_bytes)
        log_token_comment(f"evaluate_result:{Path(image_path).name}", usage)

    result = json.loads(raw)
    print(f"  Evaluate: success={result['success']} tv_confidence={result['tv_confidence']}")
    print(f"  Reasoning: {result['reasoning']}")
    return result

def evaluate_result_from_image(img, image_path: str, run_id: str = "",mock: bool = False, force_fail: bool = False) -> dict:
    _, buf = cv2.imencode(".jpg", img)
    image_bytes = buf.tobytes()

    prompt = Path(EVALUATE_PROMPT_PATH).read_text()

    if mock:
        fixture = FIXTURE_EVALUATE_FAILURE if force_fail else FIXTURE_EVALUATE_SUCCESS
        raw, usage = mock_confirm_tv(prompt, image_bytes, fixture)
    else:
        raw, usage = ask_gemini_vision(prompt, image_bytes)
        log_token_comment(f"evaluate_result:{Path(image_path).name}", usage, run_id)

    result = json.loads(raw)
    print(f"  Evaluate: success={result['success']} tv_confidence={result['tv_confidence']}")
    print(f"  Reasoning: {result['reasoning']}")
    return result

def iou(box_a: dict, box_b: dict) -> float:
    x1 = max(box_a["x1"], box_b["x1"])
    y1 = max(box_a["y1"], box_b["y1"])
    x2 = min(box_a["x2"], box_b["x2"])
    y2 = min(box_a["y2"], box_b["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0
