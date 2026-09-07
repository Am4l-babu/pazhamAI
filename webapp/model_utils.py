"""
model_utils.py - Hybrid banana analysis: trained model + Groq vision as gate/validator.

Pipeline:
  1. Groq vision (Llama 4 Scout) checks whether the image actually contains a banana.
     If not, we bail out early instead of running the model on garbage input.
  2. Our own trained BananaNet model (exported to ONNX so it stays deploy-light on
     Vercel) predicts seed count + curvature from the image.
  3. Groq is asked a second time to sanity-check the seed count against what it can
     see in the photo. If the count looks like a hallucination (e.g. wildly out of
     range for what's actually in the image), Groq's own best-guess estimate is used
     instead, so the user never sees a nonsense value with no way of knowing it's wrong.

Curvature does NOT come from the trained model. Its curvature head emits a near-constant
(227-252 degrees across all 41 training images, and the same band for random noise), so
the vision pass measures the bend from the photo instead and the model's value is kept
only as a standby for when that pass is unavailable.

The same second call also harvests the deeply unnecessary measurements the rest of
the app is built on (ripeness %, which end is the "top", raw dimensions) so the whole
page costs two API calls rather than nine. Everything derived purely from arithmetic -
shape class, volume, density - is computed here. Spot count and bruise boxes are NOT
asked of the model, which invented plausible-looking coordinates rather than locating
anything; they are measured from pixels in spot_detect.py instead.
"""

import os
import io
import json
import re
import math
import base64

import numpy as np
from PIL import Image
import onnxruntime as ort
from groq import Groq
from dotenv import load_dotenv

from spot_detect import analyse_spots

load_dotenv()  # loads .env from project root

# ── Trained model (ONNX) ─────────────────────────────────────────────────────
_session = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "banana_net.onnx")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def _preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    return arr


def _run_model(image_path: str) -> tuple:
    session = _get_session()
    inp = _preprocess(image_path)
    out = session.run(None, {"input": inp})[0][0]
    seeds = int(round(float(out[0])))
    curvature = round(float(out[1]), 2)
    return seeds, curvature


# ── Groq vision (gate + validator) ───────────────────────────────────────────
_client = None
VISION_MODEL = "qwen/qwen3.8-27b"


def _get_client() -> Groq:
    """Return (and cache) the Groq client, raising clearly if the key is absent."""
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
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _ask_groq(image_b64: str, prompt: str, max_tokens: int = 400) -> dict:
    response = _get_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


GATE_PROMPT = """You are checking whether an uploaded photo actually contains a banana \
(whole, peeled, or cut open — any form counts).

Return ONLY a valid JSON object, no markdown, no explanation outside the JSON:
{"is_banana": true or false, "confidence": "high" or "medium" or "low", "reason": "one short sentence"}
"""


def _build_analysis_prompt(seeds: int) -> str:
    return f"""A specialized computer vision model analyzed this photo of a banana and \
predicted a seed count of {seeds}.

Important context: this is a NENDRAN banana seed-counting model, trained on bananas that \
were physically split open and had every seed counted by hand. For this specific variety \
and method, seed counts of roughly 100-450 (mean around 260) are NORMAL and EXPECTED —
do NOT flag a value in that range as implausible just because ordinary supermarket \
(Cavendish) bananas look seedless from the outside. Seeds are not visible in a photo of \
an unsplit banana, so you cannot verify the exact count visually either way; only flag \
the seed count as implausible if it is a clearly broken value (negative, zero for an \
obviously intact/undamaged banana, or an extreme outlier like several thousand).

If the seed count is genuinely implausible, replace it only if it was a broken value as \
described above; if so, give a best-guess replacement somewhere in the normal 100-450 \
range rather than a low round number like 0.

Separately, measure the following from the photo. Commit to a specific number for every \
field — never return a range, and never refuse because it "cannot be determined exactly". \
A confident estimate is required.

  - curvature_degrees: how far the banana bends away from straight, 0 to 180. A perfectly \
    straight banana is 0; an ordinary supermarket banana is usually 15-45. Judge it from \
    the visible arc of the fruit.
  - ripeness_percent: how ripe the banana is, 0 (completely green) to 100 (fully brown/black).
  - spot_count: how many distinct brown/black speckles are visible on the peel. 0 if none.
  - bruises: up to 6 clearly bruised or blackened patches. Each is a box in NORMALISED \
    image coordinates where x,y is the top-left corner and 0,0 is the top-left of the whole \
    image and 1,1 the bottom-right. Keep every value between 0 and 1. Empty list if unbruised.
  - top_end: which end of the banana in this photo is the stem end (the "top"), as one of \
    "left", "right", "top", "bottom" describing where it sits in the frame.
  - length_cm / width_cm / weight_g: real-world size of the banana. Use the typical size of \
    the visible variety if there is nothing in frame to scale against.

Return ONLY a valid JSON object, no markdown, no explanation outside the JSON:
{{"plausible": true or false,
  "reason": "one short sentence",
  "fallback_seeds": integer or null,
  "curvature_degrees": number,
  "ripeness_percent": number,
  "top_end": "left" or "right" or "top" or "bottom",
  "top_end_reason": "one short sentence",
  "length_cm": number,
  "width_cm": number,
  "weight_g": number}}
"""


# ── Deriving the genuinely unnecessary numbers ───────────────────────────────
SHAPE_CLASSES = ["Straight", "Slightly curved", "Curved", "Extremely curved", "Suspicious"]
SHAPE_BANDS = ((10, "Straight"), (25, "Slightly curved"), (45, "Curved"), (90, "Extremely curved"))

# A banana occupies roughly 78% of the cylinder you could slide it into. Calibrated
# so a typical fruit lands near 0.94 g/cm3, which is why real bananas float.
BANANA_PACKING_FACTOR = 0.78
WATER_DENSITY = 1.0  # g/cm3
SPOT_GROWTH_PER_DAY = 0.35


def _as_float(value, lo: float, hi: float):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return max(lo, min(hi, num))


def _as_int(value, lo: int, hi: int):
    num = _as_float(value, lo, hi)
    return None if num is None else int(round(num))


def _classify_shape(curvature) -> str:
    """Anything bending past a right angle has stopped being a fruit problem."""
    if curvature is None or curvature < 0 or curvature > 90:
        return "Suspicious"
    for limit, label in SHAPE_BANDS:
        if curvature < limit:
            return label
    return "Suspicious"


def _forecast_spots(spot_count, ripeness_percent, days: int = 3) -> list:
    """Spots multiply faster on a banana that is already on its way out."""
    if spot_count is None:
        return []
    ripeness = 50.0 if ripeness_percent is None else ripeness_percent
    rate = SPOT_GROWTH_PER_DAY * (0.5 + ripeness / 100)
    base = max(spot_count, 1)
    return [
        {"day": d, "spots": int(round(base * (1 + rate) ** d))}
        for d in range(1, days + 1)
    ]


def _clean_bruises(raw) -> list:
    """Normalised boxes -> CSS percentages, clipped to the image."""
    if not isinstance(raw, list):
        return []
    cleaned = []
    for item in raw[:6]:
        if not isinstance(item, dict):
            continue
        x = _as_float(item.get("x"), 0, 1)
        y = _as_float(item.get("y"), 0, 1)
        w = _as_float(item.get("w"), 0, 1)
        h = _as_float(item.get("h"), 0, 1)
        if None in (x, y, w, h) or w <= 0 or h <= 0:
            continue
        severity = item.get("severity")
        if severity not in ("light", "moderate", "severe"):
            severity = "moderate"
        cleaned.append({
            "x": round(x * 100, 2),
            "y": round(y * 100, 2),
            "w": round(min(w, 1 - x) * 100, 2),
            "h": round(min(h, 1 - y) * 100, 2),
            "severity": severity,
            "label": f"Bruise {len(cleaned) + 1}",
        })
    return cleaned


def _build_useless_report(check: dict, seeds, curvature, cv: dict) -> dict:
    ripeness = _as_float(check.get("ripeness_percent"), 0, 100)
    spot_count = cv.get("spot_count")
    length_cm = _as_float(check.get("length_cm"), 1, 60)
    width_cm = _as_float(check.get("width_cm"), 0.5, 20)
    weight_g = _as_float(check.get("weight_g"), 10, 1000)

    top_end = check.get("top_end")
    if top_end not in ("left", "right", "top", "bottom"):
        top_end = None

    report = {
        "ripeness_percent": None if ripeness is None else round(ripeness, 1),
        "spot_count": spot_count,
        "spot_forecast": _forecast_spots(spot_count, ripeness),
        "bruises": _clean_bruises(cv.get("bruises")),
        "shape_class": _classify_shape(curvature),
        "shape_classes": SHAPE_CLASSES,
        "top_end": top_end,
        "top_end_reason": str(check.get("top_end_reason") or "").strip()[:200],
        "length_cm": None if length_cm is None else round(length_cm, 1),
        "width_cm": None if width_cm is None else round(width_cm, 1),
        "weight_g": None if weight_g is None else round(weight_g, 1),
        "volume_cm3": None,
        "density": None,
        "floats": None,
        "grams_per_cm": None,
        "seeds_per_cm": None,
    }

    if None not in (length_cm, width_cm, weight_g):
        radius = width_cm / 2
        volume = math.pi * radius * radius * length_cm * BANANA_PACKING_FACTOR
        if volume > 0:
            density = weight_g / volume
            report["volume_cm3"] = round(volume, 1)
            report["density"] = round(density, 3)
            report["floats"] = density < WATER_DENSITY
            report["grams_per_cm"] = round(weight_g / length_cm, 2)
            if seeds is not None:
                report["seeds_per_cm"] = round(seeds / length_cm, 1)

    return report


def predict_from_image(image_path: str) -> dict:
    """
    Analyse a banana image: Groq gate -> trained model -> Groq sanity-check.

    Returns a dict with:
        is_banana   (bool or None on error)
        seeds       (int or None)
        curvature   (float or None)
        source      ("model" | "groq_fallback" | "groq_gate" | None)
        confidence  (str high/medium/low)
        notes       (str, e.g. reason for a fallback override)
        useless     (dict of the extra measurements, or None if we never got that far)
        error       (str only if something went wrong)
    """
    try:
        b64 = _encode_image_b64(image_path)
    except Exception as e:
        return _error_result(f"Could not read the uploaded image: {e}")

    # Step 1: Groq gate — is this even a banana?
    try:
        gate = _ask_groq(b64, GATE_PROMPT)
    except Exception as e:
        # If the gate check itself fails (e.g. API hiccup), don't block the
        # whole feature on it — fall through and let the model try anyway.
        gate = {"is_banana": True, "confidence": "low", "reason": f"gate check unavailable: {e}"}

    if not gate.get("is_banana", True):
        return {
            "is_banana": False,
            "seeds": None,
            "curvature": None,
            "source": "groq_gate",
            "confidence": gate.get("confidence", "medium"),
            "notes": gate.get("reason", "This doesn't look like a banana."),
            "useless": None,
            "error": None,
        }

    # Step 2: our trained model
    try:
        seeds, model_curvature = _run_model(image_path)
    except Exception as e:
        return _error_result(f"Model inference failed: {e}")

    # Step 3: Groq sanity-check — catch a hallucinated / out-of-range prediction,
    # and collect the extra measurements in the same round trip.
    source = "model"
    notes = ""
    try:
        check = _ask_groq(b64, _build_analysis_prompt(seeds), max_tokens=600)
    except Exception:
        # If the analysis call fails, still return the model's raw prediction
        # rather than failing the whole request.
        check = {}

    # Speckles and bruises are counted in pixels, not asked of the model.
    cv = analyse_spots(image_path)

    # Curvature is read straight from the photo: the trained model's curvature head is a
    # near-constant regardless of input, so it only stands in if the vision pass is missing.
    curvature = _as_float(check.get("curvature_degrees"), 0, 180)
    curvature = model_curvature if curvature is None else round(curvature, 2)

    if not check.get("plausible", True):
        fallback_seeds = _as_int(check.get("fallback_seeds"), 0, 5000)
        if fallback_seeds is not None:
            seeds = fallback_seeds
            source = "groq_fallback"
            notes = check.get(
                "reason",
                "The model's estimate looked off, so this value was adjusted after a secondary check."
            )

    return {
        "is_banana": True,
        "seeds": seeds,
        "curvature": curvature,
        "source": source,
        "confidence": gate.get("confidence", "medium"),
        "notes": notes,
        "useless": _build_useless_report(check, seeds, curvature, cv),
        "error": None,
    }


def _error_result(msg: str) -> dict:
    return {
        "is_banana": None,
        "seeds": None,
        "curvature": None,
        "source": None,
        "confidence": "low",
        "notes": "",
        "useless": None,
        "error": msg,
    }
