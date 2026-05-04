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

FIXTURE_CONFIRM_TV_YES = MockGenerateContentResponse(
    text=json.dumps({
        "is_tv": True,
        "tv_confidence": 0.92,
        "reasoning": "The highlighted region shows a flat panel display with bezel consistent with a television"
    })
)

FIXTURE_CONFIRM_TV_NO = MockGenerateContentResponse(
    text=json.dumps({
        "is_tv": False,
        "tv_confidence": 0.08,
        "reasoning": "The highlighted region appears to be a fireplace opening, not a television screen"
    })
)

FIXTURE_CONFIRM_TV_UNCERTAIN = MockGenerateContentResponse(
    text=json.dumps({
        "is_tv": False,
        "tv_confidence": 0.45,
        "reasoning": "The highlighted region is ambiguous, could be a display but context suggests otherwise"
    })
)

FIXTURE_EVALUATE_SUCCESS = MockGenerateContentResponse(
    text=json.dumps({
        "success": True,
        "tv_confidence": 0.94,
        "reasoning": "Beach scene is correctly placed on the TV screen and fills it naturally"
    })
)

FIXTURE_EVALUATE_FAILURE = MockGenerateContentResponse(
    text=json.dumps({
        "success": False,
        "tv_confidence": 0.88,
        "reasoning": "No beach scene visible on the TV screen in this image"
    })
)

FIXTURE_TV_QUAD = MockGenerateContentResponse(
    text=json.dumps({
        "quad_points": [[80, 60], [80, 940], [920, 940], [920, 60]],
        "confidence": 0.91,
        "reasoning": "Flat panel screen clearly visible, corners identified at bezel inner edge"
    })
)

# --- Mock client ---

def mock_gemini_vision(prompt: str, image_bytes: bytes, fixture: MockGenerateContentResponse) -> tuple[str, MockUsageMetadata]:
    """
    Drop-in for ask_gemini_vision.
    prompt and image_bytes are accepted but ignored.
    Returns (text, usage_metadata) matching real API contract.
    """
    return fixture.text, fixture.usage_metadata

def mock_confirm_tv(prompt: str, image_bytes: bytes, fixture: MockGenerateContentResponse) -> tuple[str, MockUsageMetadata]:
    """
    Drop-in for ask_gemini_vision when confirming a quad is a TV.
    prompt and image_bytes accepted but ignored.
    Returns (text, usage_metadata) matching real API contract.
    """
    return fixture.text, fixture.usage_metadata
