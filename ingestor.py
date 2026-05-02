from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def discover_images(input_dir: str) -> list[str]:
    """Return sorted list of image paths in input_dir."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"Not a directory: {input_dir}")
    
    paths = sorted([
        str(p) for p in input_path.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    
    if not paths:
        raise ValueError(f"No images found in {input_dir}")
    
    return paths
