# gemini.py
import os
import mimetypes
from google import genai
from google.genai import types
import warnings
from urllib3.exceptions import NotOpenSSLWarning

GEMINI_MODEL="gemini-2.5-flash"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

class GeminiError(Exception):
    pass

class GeminiAPIError(GeminiError):
    pass

class GeminiParseError(GeminiError):
    pass

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
	raise EnvironmentError("GEMINI_API_KEY not set")

client = genai.Client(api_key=api_key)

def ask_gemini_vision(prompt: str, image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
    except FileNotFoundError:
        raise
    except Exception as e:
        raise GeminiAPIError(f"API call failed: {e}") from e

    text = response.text
    if not text:
        raise GeminiParseError("Empty reponse from Gemini")
    return text
