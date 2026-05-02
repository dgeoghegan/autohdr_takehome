# mock_gemini.py
from dataclasses import dataclass, field
import json


@dataclass
class MockUsageMetadata:
    prompt_token_count: int = 512
    candidates_token_count: int = 89
    total_token_count: int = 601


@dataclass
class MockGenerateContentResponse:
    text: str = ""
    usage_metadata: MockUsageMetadata = field(default_factory=MockUsageMetadata)


# --- Fixture payloads ---

def _response(detections: list, tv_found: bool) -> MockGenerateContentResponse:
    payload = json.dumps({
        "detections": detections,
        "tv_found": tv_found,
    })
    return MockGenerateContentResponse(text=payload)


FIXTURE_TV_SINGLE = _response(
    detections=[
        {
            "tv_confidence": 0.94,
            "identified_as": "television",
            "box_2d": [200, 100, 600, 800],
            "reasoning": "Flat panel with bezel, screen glare consistent with display",
        }
    ],
    tv_found=True,
)

FIXTURE_TV_MULTIPLE = _response(
    detections=[
        {
            "tv_confidence": 0.91,
            "identified_as": "television",
            "box_2d": [150, 50, 500, 600],
            "reasoning": "Large flat panel mounted on wall",
        },
        {
            "tv_confidence": 0.87,
            "identified_as": "television",
            "box_2d": [200, 620, 550, 950],
            "reasoning": "Second display visible in background",
        },
    ],
    tv_found=True,
)

FIXTURE_NO_TV = _response(detections=[], tv_found=False)

FIXTURE_CONFOUNDER = _response(
    detections=[
        {
            "tv_confidence": 0.31,
            "identified_as": "microwave",
            "box_2d": [400, 500, 600, 750],
            "reasoning": "Rectangular display but small, context suggests kitchen appliance",
        }
    ],
    tv_found=False,
)


# --- Mock client ---

def mock_gemini_vision(prompt: str, image_bytes: bytes, fixture: MockGenerateContentResponse) -> tuple[str, MockUsageMetadata]:
    """
    Drop-in for ask_gemini_vision.
    prompt and image_bytes are accepted but ignored.
    Returns (text, usage_metadata) matching real API contract.
    """
    return fixture.text, fixture.usage_metadata
