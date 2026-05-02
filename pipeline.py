# pipeline.py
import json
import sys
from pathlib import Path
from gemini import ask_gemini_vision, GeminiError
import cv2

PROMPT_PATH = "prompts/id_tv_1.txt"
IMAGE_PATH  = "photos/autohdr-orig/1250_src.jpg"
CROP_DIR    = "photos/crops"

def load_prompt(width: int, height: int) -> str:
    template = Path(PROMPT_PATH).read_text()
    return template.replace("{WIDTH}", str(width)).replace("{HEIGHT}", str(height))

def save_crops(image_path: str, detections: list):
    base = Path(image_path).stem
    out_dir = Path(CROP_DIR) / base
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img = cv2.imread(image_path)

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        crop = img[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
        out_path = f"{out_dir}/crop_{i}_{det['identified_as']}_{det['tv_confidence']}.jpg"
        cv2.imwrite(str(out_path), crop)
        print(f"    Saved {out_path}")

def descale_bbox(box_2d: list, width: int, height: int) -> dict:
    """Convert Gemini's [ymin, xmin, ymax, xmax] normalized 0-1000 to pixel coords."""
    ymin, xmin, ymax, xmax = box_2d
    return {
        "x1": int(xmin / 1000 * width),
        "y1": int(ymin / 1000 * height),
        "x2": int(xmax / 1000 * width),
        "y2": int(ymax / 1000 * height)
    }

def main():
    img = cv2.imread(IMAGE_PATH)
    h, w = img.shape[:2]
    _, buf = cv2.imencode(".jpg", img)
    image_bytes = buf.tobytes()

    prompt = load_prompt(w, h)
    print("prompt:")
    print(prompt)
    
    try:
        raw = ask_gemini_vision(prompt, image_bytes)
    except GeminiError as e:
        print(f"Gemini error: {e}")
        sys.exit(1)

    result = json.loads(raw)
    
    for det in result["detections"]:
        det["bbox"] = descale_bbox(det["box_2d"], w, h)

    print(json.dumps(result, indent=2))
    save_crops(IMAGE_PATH, result["detections"])

if __name__ == "__main__":
    main()
