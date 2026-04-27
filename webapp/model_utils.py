"""
model_utils.py - Banana Analysis using Groq Vision API (Llama 4 Scout)
Replaces the overfitted BananaNet PyTorch model.
Capabilities:
  - Detect if the image contains a banana
  - Classify ripeness on a 1-5 scale (matching training data)
  - Estimate seed count based on visual analysis + training data statistics
"""

import os
import io
import json
import re
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root
from PIL import Image

# ── Training data statistics for seed count estimation ──────────────────────
# Derived from clean_dataset.csv (32 samples)
SEED_STATS = {
    "min": 102,
    "max": 427,
    "mean": 261,
    "by_ripeness": {
        1: {"mean": 257, "min": 146, "max": 373},   # unripe
        2: {"mean": 281, "min": 147, "max": 421},   # slightly unripe
        3: {"mean": 267, "min": 173, "max": 375},   # semi-ripe
        4: {"mean": 295, "min": 102, "max": 392},   # ripe
        5: {"mean": 228, "min": 103, "max": 374},   # overripe
    }
}

RIPENESS_LABELS = {
    1: "Unripe 🟢",
    2: "Slightly Unripe 🟡",
    3: "Semi-Ripe 🟡",
    4: "Ripe ✅",
    5: "Overripe 🟤",
}

RIPENESS_DESCRIPTIONS = {
    1: "Green, firm, and starchy. Not ready to eat.",
    2: "Mostly green with slight yellowing. Nearly ready.",
    3: "Yellow with green tips. Good for cooking.",
    4: "Fully yellow, sweet, and ready to eat.",
    5: "Brown spots or fully brown. Very sweet, best for baking.",
}

# ── Groq API setup ───────────────────────────────────────────────────────────
# NOTE: Client is created lazily so a missing key produces a clean error
# response instead of crashing the Python process at import time (Vercel).
_client = None
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _get_client() -> Groq:
    """Return (and cache) the Groq client, raising clearly if key is absent."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your Vercel project's Environment Variables."
            )
        _client = Groq(api_key=api_key)
    return _client


def _encode_image_b64(image_path: str) -> str:
    """Read image, resize if needed, and return base64-encoded JPEG string."""
    img = Image.open(image_path).convert("RGB")
    # Limit to 1024px on longest side to stay within token limits
    img.thumbnail((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_from_image(image_path: str) -> dict:
    """
    Analyse a banana image using Groq Vision (Llama 4 Scout).

    Returns a dict with:
        is_banana      (bool)
        ripeness       (int 1-5, or None)
        ripeness_label (str)
        ripeness_desc  (str)
        seeds          (int estimated)
        seeds_range    (str e.g. "220 – 310")
        confidence     (str high/medium/low)
        notes          (str extra commentary)
        visual_clues   (str colour/texture observations)
        error          (str only if something went wrong)
    """
    try:
        b64_image = _encode_image_b64(image_path)

        prompt = """You are a banana analysis expert. Carefully look at the image and respond ONLY with a valid JSON object — no markdown fences, no explanation, no extra text.

The JSON must follow this exact schema:
{
  "is_banana": true or false,
  "ripeness": integer 1-5 or null,
  "confidence": "high" or "medium" or "low",
  "visual_clues": "brief description of colour, spots, texture seen in image",
  "notes": "any other relevant observation (variety, freshness, defects, etc.)"
}

Ripeness scale (match to training data labels):
  1 = Unripe         (fully green skin)
  2 = Slightly Unripe (mostly green, starting to turn yellow)
  3 = Semi-Ripe      (yellow with green tips or patches)
  4 = Ripe           (fully yellow, possibly tiny brown flecks)
  5 = Overripe       (heavy brown/black spots or fully brown/black)

If the image does NOT contain a banana:
  - set "is_banana" to false
  - set "ripeness" to null
  - set "confidence" based on how certain you are it is NOT a banana
  - briefly describe what you see in "visual_clues"
  - leave "notes" as an empty string

Return ONLY the JSON. No markdown. No backticks."""

        response = _get_client().chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            temperature=0.1,   # low temperature for deterministic JSON output
            max_tokens=512,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model wraps the JSON anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        is_banana = bool(data.get("is_banana", False))

        if not is_banana:
            return {
                "is_banana": False,
                "ripeness": None,
                "ripeness_label": "N/A",
                "ripeness_desc": "No banana detected in the image.",
                "seeds": None,
                "seeds_range": "N/A",
                "confidence": data.get("confidence", "high"),
                "notes": data.get("notes", ""),
                "visual_clues": data.get("visual_clues", ""),
                "error": None,
            }

        ripeness = data.get("ripeness")
        if ripeness is not None:
            ripeness = max(1, min(5, int(ripeness)))

        seeds, seeds_range = _estimate_seeds(ripeness)

        return {
            "is_banana": True,
            "ripeness": ripeness,
            "ripeness_label": RIPENESS_LABELS.get(ripeness, "Unknown"),
            "ripeness_desc": RIPENESS_DESCRIPTIONS.get(ripeness, ""),
            "seeds": seeds,
            "seeds_range": seeds_range,
            "confidence": data.get("confidence", "medium"),
            "notes": data.get("notes", ""),
            "visual_clues": data.get("visual_clues", ""),
            "error": None,
        }

    except json.JSONDecodeError as e:
        return _error_result(f"Could not parse model response as JSON: {e}. Raw: {raw[:200]}")
    except Exception as e:
        return _error_result(str(e))


def _estimate_seeds(ripeness) -> tuple:
    """Return (mean_estimate, 'min – max') based on ripeness level from training data."""
    if ripeness and ripeness in SEED_STATS["by_ripeness"]:
        stats = SEED_STATS["by_ripeness"][ripeness]
        return stats["mean"], f"{stats['min']} – {stats['max']}"
    return SEED_STATS["mean"], f"{SEED_STATS['min']} – {SEED_STATS['max']}"


def _error_result(msg: str) -> dict:
    return {
        "is_banana": None,
        "ripeness": None,
        "ripeness_label": "Error",
        "ripeness_desc": "",
        "seeds": None,
        "seeds_range": "N/A",
        "confidence": "low",
        "notes": "",
        "visual_clues": "",
        "error": msg,
    }
