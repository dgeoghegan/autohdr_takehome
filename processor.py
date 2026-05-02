# processor.py
from pathlib import Path
import cv2
import numpy as np

CROP_DIR    = "photos/crops"
OUTPUT_DIR = "photos/output"

def save_crops(image_path: str, detections: list) -> list[str]:
    base = Path(image_path).stem
    out_dir = Path(CROP_DIR) / base
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img = cv2.imread(image_path)
    crop_paths = []

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        crop = img[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
        out_path = f"{out_dir}/{base}_crop_{i}_{det['identified_as']}_{det['tv_confidence']}.jpg"
        cv2.imwrite(str(out_path), crop)
        crop_paths.append(out_path)
        print(f"    Saved {out_path}")
    return crop_paths

def find_screen_quad(crop_path: str):
    img = cv2.imread(crop_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    stem = Path(crop_path).stem
    parent = Path(crop_path).parent
    cv2.imwrite(str(parent / f"{stem}_gray.jpg"), gray)
    cv2.imwrite(str(parent / f"{stem}_edges.jpg"), edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    print(f"  Found {len(contours)} contours")

    quad = None
    for contour in contours[:10]: # log top 10
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        print(f"    contour area: {cv2.contourArea(contour):.0f}, vertices: {len(approx)}")

        if quad is None and len(approx) == 4:
            quad = approx

    out = img.copy()
    if quad is not None:
        print(f"    Quad found: {quad.reshape(4,2).tolist()}")
        cv2.drawContours(out, [quad], -1, (180, 105, 255), 3)  # pink in BGR
    else:
        print(f"  No quadrilateral found in {crop_path}")

    stem = Path(crop_path).stem
    suffix = Path(crop_path).suffix
    out_path = str(Path(crop_path).parent / f"{stem}_highlighted{suffix}")
    cv2.imwrite(out_path, out)
    print(f"  Saved {out_path}")
    return quad

def replace_screen(image_path: str, quad: list, replacement_path: str, out_dir: str) -> str:
    img = cv2.imread(image_path)
    replacement = cv2.imread(replacement_path)

    dst_pts = np.array(quad, dtype=np.float32)

    x, y, w, h = cv2.boundingRect(dst_pts.astype(np.int32))
    src_pts = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    resized = cv2.resize(replacement, (w, h))
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(resized, M, (img.shape[1], img.shape[0]))

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
    mask_3ch = cv2.merge([mask, mask, mask])

    img = np.where(mask_3ch == 255, warped, img)

    base = Path(image_path).stem
    out_path = Path(out_dir) / f"{base}_replaced.jpg"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"  Saved {out_path}")
    return str(out_path)
