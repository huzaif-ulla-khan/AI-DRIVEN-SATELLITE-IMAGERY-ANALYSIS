# app/services/report.py
import os, io, base64, csv
from PIL import Image

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_csv_report(path: str, payload: dict):
    """
    payload: { lat, lon, radius_km, start_date, end_date, distribution: {class: pct} }
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", payload.get("lat")])
        writer.writerow(["lon", payload.get("lon")])
        writer.writerow(["radius_km", payload.get("radius_km")])
        writer.writerow(["start_date", payload.get("start_date")])
        writer.writerow(["end_date", payload.get("end_date")])
        writer.writerow([])
        writer.writerow(["class", "percent"])
        for k, v in (payload.get("distribution") or {}).items():
            writer.writerow([k, v])

def to_base64_png(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
