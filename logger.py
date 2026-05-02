# logger.py
import json
from datetime import datetime
from pathlib import Path

TOKEN_LOG = "logs/token_usage.jsonl"

def log_token_usage(image_path: str, usage_metadata) -> None:
    Path(TOKEN_LOG).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "image": Path(image_path).name,
        "prompt_tokens": usage_metadata.prompt_token_count,
        "candidates_tokens": usage_metadata.candidates_token_count,
        "total_tokens": usage_metadata.total_token_count,
    }
    with open(TOKEN_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
