# app/services/fetch.py
from PIL import Image, ImageDraw
import numpy as np

def fetch_satellite_image_stub(lat: float, lon: float, radius_km: float, start_date: str, end_date: str):
    """
    Returns a synthetic RGB PIL image (256x256) representing the queried area.
    Replace this with real Earth Engine / Sentinel Hub fetch later.
    """
    w, h = 256, 256
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # green-blue gradient (pretend vegetation/water mixture)
    for y in range(h):
        arr[y, :, 1] = int(100 + 100 * y / h)      # green channel
        arr[y, :, 2] = int(120 + 80 * (1 - y / h)) # blue channel
    img = Image.fromarray(arr, mode="RGB")

    # draw a red ring to hint region of interest
    draw = ImageDraw.Draw(img)
    cx, cy, r = w // 2, h // 2, 30
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 0, 0), width=3)
    return img
