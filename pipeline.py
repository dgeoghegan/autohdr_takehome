# pipeline.py
import json
import sys
from pathlib import Path
from gemini import ask_gemini_vision, GeminiError
import cv2

PROMPT_PATH = "prompts/id_tv_1.txt"
IMAGE_PATH  = "photos/autohdr-orig/1250_src.jpg"

def load_prompt(image_path: str) -> str:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    template = Path(PROMPT_PATH).read_text()
    return template.replace("{WIDTH}", str(w)).replace("{HEIGHT}", str(h))

def main():
    prompt = load_prompt(IMAGE_PATH)
    
    try:
        raw = ask_gemini_vision(prompt, IMAGE_PATH)
    except GeminiError as e:
        print(f"Gemini error: {e}")
        sys.exit(1)

    result = json.loads(raw)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
