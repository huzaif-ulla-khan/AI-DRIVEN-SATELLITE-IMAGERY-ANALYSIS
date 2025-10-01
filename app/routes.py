from flask import Blueprint, render_template, request, url_for, current_app
from datetime import datetime
import os, time

from app.services.fetch import fetch_satellite_image_stub
from app.services.predict import predict_stub
from app.services.report import ensure_dir, save_csv_report, to_base64_png

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    # Gather inputs
    lat = float(request.form.get('lat'))
    lon = float(request.form.get('lon'))
    radius_km = float(request.form.get('radius_km', 5))
    start_date = request.form.get('start_date') or "2024-01-01"
    end_date = request.form.get('end_date') or datetime.utcnow().date().isoformat()

    # 1) Fetch image (stub)
    img = fetch_satellite_image_stub(lat, lon, radius_km, start_date, end_date)

    # 2) Predict (stub)
    pred = predict_stub(img)  # overlay (PIL RGBA), distribution, confidences

    # 3) Save CSV report
    exports_dir = os.path.join(current_app.root_path, "static", "exports")
    ensure_dir(exports_dir)
    ts = int(time.time())
    csv_path = os.path.join(exports_dir, f"report_{ts}.csv")
    save_csv_report(csv_path, {
        "lat": lat, "lon": lon, "radius_km": radius_km,
        "start_date": start_date, "end_date": end_date,
        "distribution": pred["distribution"]
    })

    # 4) Base64 overlay for Leaflet ImageOverlay
    overlay_b64 = to_base64_png(pred["overlay"])

    return render_template(
        "dashboard.html",
        lat=lat, lon=lon, radius_km=radius_km,
        start_date=start_date, end_date=end_date,
        distribution=pred["distribution"],
        confidences=pred["confidences"],
        overlay_b64=overlay_b64,
        csv_url=url_for('static', filename=f"exports/report_{ts}.csv")
    )
