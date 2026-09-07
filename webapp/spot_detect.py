"""
spot_detect.py - Classical computer vision for the speckles and bruises on a banana peel.

This work used to be asked of the vision model, which turned out to invent plausible-looking
boxes rather than locate anything: on a test photo two of three landed on blank background.
Counting dark blobs in pixels is deterministic, runs in milliseconds and needs no API, so it
is done here instead.

The approach: drop the near-white background, take the bright peel as the brightness
reference, mark every pixel meaningfully darker than it, then group those into connected
regions. Small regions are speckles; large ones are bruises.
"""

import numpy as np
from PIL import Image

WORK_MAX_EDGE = 480       # downscale first; nothing here needs full resolution
BG_VALUE_MIN = 0.75       # background is bright...
BG_SAT_MAX = 0.18         # ...and washed out
SPOT_DARKNESS = 0.72      # a spot is this fraction of the peel's reference brightness
MIN_FRUIT_FRACTION = 0.02  # below this the segmentation clearly failed
MAX_FRUIT_FRACTION = 0.97  # above this we never found a background
# A blemish spanning half the frame, or running off the edge of it, is background or
# shadow rather than a bruise. Cluttered scenes fail these, which is the intent: better
# to report nothing than to box the tablecloth.
MAX_BRUISE_AREA = 0.12    # fraction of the fruit
MAX_BRUISE_EDGE = 0.5     # fraction of the image, per side
SEVERE_BELOW = 0.45
MODERATE_BELOW = 0.62
MAX_BRUISES = 6


def _load(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((WORK_MAX_EDGE, WORK_MAX_EDGE), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _components(mask: np.ndarray, min_area: int) -> list:
    """Flood-fill only the set pixels, so cost tracks the spots rather than the image."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    found = []
    for y0, x0 in zip(*(a.tolist() for a in np.nonzero(mask))):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) < min_area:
            continue
        ys = [p[0] for p in pixels]
        xs = [p[1] for p in pixels]
        found.append({
            "area": len(pixels),
            "y0": min(ys), "y1": max(ys) + 1,
            "x0": min(xs), "x1": max(xs) + 1,
            "pixels": pixels,
        })
    return found


def analyse_spots(image_path: str) -> dict:
    """
    Count peel speckles and box the bruises.

    Returns {"spot_count": int|None, "bruises": [{x,y,w,h,severity} normalised 0-1]}.
    spot_count is None when the fruit could not be separated from its background, so the
    caller can say "could not be measured" instead of showing an invented number.
    """
    failed = {"spot_count": None, "bruises": []}
    try:
        rgb = _load(image_path)
    except Exception:
        return failed

    h, w, _ = rgb.shape
    value = rgb.max(axis=2)
    span = value - rgb.min(axis=2)
    sat = np.divide(span, value, out=np.zeros_like(span), where=value > 1e-6)

    fruit = ~((value > BG_VALUE_MIN) & (sat < BG_SAT_MAX))
    fraction = float(fruit.mean())
    if not (MIN_FRUIT_FRACTION < fraction < MAX_FRUIT_FRACTION):
        return failed

    peel = value[fruit]
    reference = float(np.percentile(peel, 75))  # the bright, unblemished yellow
    if reference <= 1e-6:
        return failed

    dark = fruit & (value < SPOT_DARKNESS * reference)
    fruit_area = int(fruit.sum())
    regions = _components(dark, min_area=max(4, int(fruit_area * 0.00008)))
    if not regions:
        return {"spot_count": 0, "bruises": []}

    # Anything past this size stops reading as a speckle and starts reading as a bruise.
    bruise_area = max(60, int(fruit_area * 0.004))
    bruises = []
    for r in sorted(regions, key=lambda r: r["area"], reverse=True):
        if r["area"] < bruise_area or len(bruises) >= MAX_BRUISES:
            continue
        if r["area"] > fruit_area * MAX_BRUISE_AREA:
            continue
        if (r["x1"] - r["x0"]) > w * MAX_BRUISE_EDGE or (r["y1"] - r["y0"]) > h * MAX_BRUISE_EDGE:
            continue
        if r["x0"] == 0 or r["y0"] == 0 or r["x1"] >= w or r["y1"] >= h:
            continue  # runs off the frame, so not a mark on the fruit
        darkness = float(np.mean([value[y, x] for y, x in r["pixels"]]) / reference)
        bruises.append({
            "x": r["x0"] / w,
            "y": r["y0"] / h,
            "w": (r["x1"] - r["x0"]) / w,
            "h": (r["y1"] - r["y0"]) / h,
            "severity": "severe" if darkness < SEVERE_BELOW
                        else "moderate" if darkness < MODERATE_BELOW else "light",
        })

    return {"spot_count": len(regions), "bruises": bruises}
