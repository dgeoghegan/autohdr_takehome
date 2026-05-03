# processor.py
from pathlib import Path
import cv2
import numpy as np

CROP_DIR    = "photos/crops"
OUTPUT_DIR = "photos/output"

def save_crop(image_path: str, det: dict, index: int) -> str:
    base = Path(image_path).stem
    out_dir = Path(CROP_DIR) / base
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(image_path)

    bbox = det["bbox"]
    crop = img[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
    label = det["identified_as"].replace("/", "_").replace(" ", "_")
    out_path = str(out_dir / f"{base}_crop_{index}_{label}_{det['tv_confidence']}.jpg")
    cv2.imwrite(out_path, crop)
    print(f"    Saved {out_path}")
    return out_path

# keeping legacy save_crops until sure I don't need
def save_crops(image_path: str, detections: list) -> list[str]:
    base = Path(image_path).stem
    out_dir = Path(CROP_DIR) / base
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img = cv2.imread(image_path)
    crop_paths = []

    for i, det in enumerate(detections):
        bbox = det["bbox"]
        crop = img[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
        label = det['identified_as'].replace('/', '_').replace(' ', '_')
        out_path = f"{out_dir}/{base}_crop_{i}_{label}_{det['tv_confidence']}.jpg"
        cv2.imwrite(str(out_path), crop)
        crop_paths.append(out_path)
        print(f"    Saved {out_path}")
    return crop_paths

def find_screen_quad(crop_path: str):
    img = cv2.imread(crop_path)
    if img is None:
        print(f"    Could not read {crop_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    stem = Path(crop_path).stem
    parent = Path(crop_path).parent
    cv2.imwrite(str(parent / f"{stem}_edges.jpg"), edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    print(f"  Found {len(contours)} contours")

    quads = []
    for contour in contours[:10]: # log top 10
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        print(f"    contour area: {cv2.contourArea(contour):.0f}, vertices: {len(approx)}")
        if len(approx) == 4:
            quads.append(approx)

    print(f"   Found {len(quads)} quad candidates")
    return quads

def replace_screen(image_path: str, quad: list, replacement_path: str, out_dir: str) -> str:
    img = cv2.imread(image_path)
    replacement = cv2.imread(replacement_path)

    quad = sort_quad_points(quad)
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

    return img, out_path

def draw_quad_highlight(image_path: str, quad, attempt: int = 0) -> tuple[str, bytes]:

    img = cv2.imread(image_path)
    out = img.copy()
    cv2.drawContours(out, [np.array(quad)], -1, (180, 105, 255), 3)

    stem = Path(image_path).stem
    out_dir = Path(CROP_DIR) / stem
    out_dir.mkdir(parents = True, exist_ok = True)
    out_path = str(out_dir / f"{stem}_highlighted_{attempt}.jpg")
    cv2.imwrite(out_path, out)

    _, buf = cv2.imencode(".jpg", out)
    image_bytes = buf.tobytes()

    return out_path, image_bytes

def save_result(img, out_path: str) -> str:
    cv2.imwrite(out_path, img)
    print(f"  Saved {out_path}")
    return out_path

def sort_quad_points(quad: list) -> list:
    """Sort quad points into [top-left, top-right, bottom-right, bottom-left] order."""
    pts = sorted(quad, key=lambda p: p[1])  # sort by y
    top = sorted(pts[:2], key=lambda p: p[0])   # top two, sort by x
    bottom = sorted(pts[2:], key=lambda p: p[0], reverse=True)  # bottom two, right to left
    return [top[0], top[1], bottom[0], bottom[1]]

def quad_to_bbox(quad: list) -> dict:
    pts = np.array(quad, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
