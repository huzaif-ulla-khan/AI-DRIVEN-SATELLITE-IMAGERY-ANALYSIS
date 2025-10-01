# app/services/predict.py
from PIL import Image
import numpy as np
import random

CLASSES = ["forest", "water", "urban", "barren"]
COLORS = {
    "forest": (34, 139, 34, 150),
    "water": (65, 105, 225, 150),
    "urban": (169, 169, 169, 150),
    "barren": (184, 134, 11, 150),
}

def predict_stub(img: Image.Image):
    """
    Fake predictor that produces:
      - overlay: RGBA mask
      - distribution: % per class
      - confidences: class → [0..1]
    """
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # Assign a class to each quadrant randomly
    quads = CLASSES[:]
    random.shuffle(quads)

    for y in range(h):
        for x in range(w):
            idx = (y > h // 2) * 2 + (x > w // 2)  # 0..3
            cls = quads[idx]
            overlay.putpixel((x, y), COLORS[cls])

    # Random, but normalized distribution & confidences
    distribution = {c: round(random.uniform(15, 35), 2) for c in CLASSES}
    total = sum(distribution.values())
    distribution = {k: round(v * 100 / total, 2) for k, v in distribution.items()}
    confidences = {c: round(random.uniform(0.70, 0.98), 2) for c in CLASSES}

    return {"overlay": overlay, "distribution": distribution, "confidences": confidences}
