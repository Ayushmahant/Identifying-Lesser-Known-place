from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import math

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset for suggestions
data = pd.read_csv('places.csv')

# Haversine formula to calculate distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/suggest', methods=['GET'])
def suggest_places():
    # Get latitude and longitude from user input
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')

    # Validate latitude and longitude inputs
    if latitude is None or longitude is None:
        return jsonify({'error': 'Latitude and Longitude are required!'}), 400

    try:
        latitude = float(latitude.strip())
        longitude = float(longitude.strip())
    except ValueError:
        return jsonify({'error': 'Invalid Latitude or Longitude!'}), 400

    # Calculate distance and filter places within 100 km
    data['distance'] = data.apply(lambda row: haversine(latitude, longitude, row['latitude'], row['longitude']), axis=1)
    nearby_places = data[data['distance'] <= 100]  # Filter places within 100 km

    # Check if we have any nearby places
    if nearby_places.empty:
        return jsonify({'message': 'No places found within 100 km.'}), 200

    # Prepare features for prediction (only for nearby places)
    features = nearby_places[['latitude', 'longitude', 'Popularity Index']]
    predictions = model.predict(features)

    # Get less visited places (predictions == 1)
    less_visited_places = nearby_places[predictions == 1]

    # If no less visited places, return a message
    if less_visited_places.empty:
        return jsonify({'message': 'No less visited places found within 100 km.'}), 200

    # Return the suggested places
    return jsonify(less_visited_places.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
