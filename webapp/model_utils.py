"""
model_utils.py - Hybrid banana analysis: trained model + Groq vision as gate/validator.

Pipeline:
  1. Groq vision (Llama 4 Scout) checks whether the image actually contains a banana.
     If not, we bail out early instead of running the model on garbage input.
  2. Our own trained BananaNet model (exported to ONNX so it stays deploy-light on
     Vercel) predicts seed count + curvature from the image.
  3. Groq is asked a second time to sanity-check the model's numbers against what
     it can see in the photo. If the model's prediction looks like a hallucination
     (e.g. wildly out of range for what's actually in the image), Groq's own
     best-guess estimate is used instead, so the user never sees a nonsense value
     with no way of knowing it's wrong.
"""

import os
import io
import json
import re
import base64

import numpy as np
from PIL import Image
import onnxruntime as ort
from groq import Groq
from dotenv import load_dotenv

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
    curvature = round(float(out[1]), 1)
    return seeds, curvature


# ── Groq vision (gate + validator) ───────────────────────────────────────────
_client = None
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


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


def _ask_groq(image_b64: str, prompt: str) -> dict:
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
        max_tokens=400,
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


def _build_validate_prompt(seeds: int, curvature: float) -> str:
    return f"""A specialized computer vision model analyzed this photo of a banana and predicted:
  - seed count: {seeds}
  - curvature: {curvature} degrees

Important context: this is a NENDRAN banana seed-counting model, trained on bananas that \
were physically split open and had every seed counted by hand. For this specific variety \
and method, seed counts of roughly 100-450 (mean around 260) are NORMAL and EXPECTED —
do NOT flag a value in that range as implausible just because ordinary supermarket \
(Cavendish) bananas look seedless from the outside. Seeds are not visible in a photo of \
an unsplit banana, so you cannot verify the exact count visually either way; only flag \
the seed count as implausible if it is a clearly broken value (negative, zero for an \
obviously intact/undamaged banana, or an extreme outlier like several thousand).

Curvature (the bend angle in degrees, roughly 0-360) IS something you can visually judge \
from the photo's shape — flag it if it clearly contradicts what the banana looks like.

If a value is genuinely implausible:
  - for curvature, give your own best-guess replacement based on the visible bend in the photo.
  - for seeds, only replace it if the original was a broken value as described above; if so, \
    give a best-guess replacement somewhere in the normal 100-450 range rather than a low \
    round number like 0.

Return ONLY a valid JSON object, no markdown, no explanation outside the JSON:
{{"plausible": true or false, "reason": "one short sentence", "fallback_seeds": integer or null, "fallback_curvature": number or null}}
"""


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
            "error": None,
        }

    # Step 2: our trained model
    try:
        seeds, curvature = _run_model(image_path)
    except Exception as e:
        return _error_result(f"Model inference failed: {e}")

    # Step 3: Groq sanity-check — catch a hallucinated / out-of-range prediction
    source = "model"
    notes = ""
    try:
        check = _ask_groq(b64, _build_validate_prompt(seeds, curvature))
        if not check.get("plausible", True):
            fallback_seeds = check.get("fallback_seeds")
            fallback_curvature = check.get("fallback_curvature")
            if fallback_seeds is not None:
                seeds = int(fallback_seeds)
            if fallback_curvature is not None:
                curvature = round(float(fallback_curvature), 1)
            source = "groq_fallback"
            notes = check.get(
                "reason",
                "The model's estimate looked off, so this value was adjusted after a secondary check."
            )
    except Exception:
        # If the sanity-check call fails, still return the model's raw prediction
        # rather than failing the whole request.
        pass

    return {
        "is_banana": True,
        "seeds": seeds,
        "curvature": curvature,
        "source": source,
        "confidence": gate.get("confidence", "medium"),
        "notes": notes,
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
        "error": msg,
    }
