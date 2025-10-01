from flask import Blueprint, render_template, request

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    lat = request.form.get('lat')
    lon = request.form.get('lon')
    radius_km = request.form.get('radius_km', 5)
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    # Phase 3 will: fetch satellite imagery based on these inputs
    return render_template('results.html',
                           lat=lat, lon=lon, radius_km=radius_km,
                           start_date=start_date, end_date=end_date)
