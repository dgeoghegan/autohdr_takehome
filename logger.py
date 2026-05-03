# logger.py
import json
from datetime import datetime, UTC
from pathlib import Path
from dataclasses import dataclass, field
import time

TOKEN_LOG = "logs/token_usage.jsonl"

@dataclass
class RunStats:
    start_time: float = field(default_factory=time.time)
    total_images: int = 0
    no_tv_detected: int = 0
    confirmation_failed: int = 0
    evaluation_failed: int = 0
    gemini_error: int = 0
    cv2_no_quad: int = 0
    successes: int = 0
    total_tokens: int = 0
    run_id: str = ""

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def log_summary(self) -> None:
        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "runtime_seconds": round(self.elapsed(), 2),
            "total_images": self.total_images,
            "successes": self.successes,
            "no_tv_detected": self.no_tv_detected,
            "confirmation_failed": self.confirmation_failed,
            "evaluation_failed": self.evaluation_failed,
            "gemini_error": self.gemini_error,
            "cv2_no_quad": self.cv2_no_quad,
            "total_tokens": self.total_tokens,
        }
        RUN_LOG = "logs/run_summary.jsonl"
        Path(RUN_LOG).parent.mkdir(parents=True, exist_ok=True)
        with open(RUN_LOG, "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"\nRun summary: {json.dumps(summary, indent=2)}")

def log_token_usage(image_path: str, usage_metadata) -> None:
    Path(TOKEN_LOG).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "image": Path(image_path).name,
        "prompt_tokens": usage_metadata.prompt_token_count,
        "candidates_tokens": usage_metadata.candidates_token_count,
        "total_tokens": usage_metadata.total_token_count,
    }
    with open(TOKEN_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

def log_token_comment(comment: str, usage_metadata, run_id: str = "") -> None:
    Path(TOKEN_LOG).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "comment": comment,
        "prompt_tokens": usage_metadata.prompt_token_count,
        "candidates_tokens": usage_metadata.candidates_token_count,
        "total_tokens": usage_metadata.total_token_count,
    }
    with open(TOKEN_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

def log_image_result(image_path: str, status: str, run_id: str, reason: str = "") -> None:
    RESULT_LOG = "logs/image_results.jsonl"
    Path(RESULT_LOG).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "image": Path(image_path).name,
        "status": status,
        "reason": reason,
    }
    with open(RESULT_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
