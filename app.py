#!/usr/bin/env python3
"""
Flask API for Telematics Risk Modeling.

Converts the original CLI logic into a RESTful API for a modern frontend.
This version includes flask-cors for cross-origin resource sharing, which is
essential for a decoupled architecture (like a React frontend on a different domain).
"""

# --- FLASK IMPORTS ---
from flask import Flask, request, jsonify
from flask_cors import CORS # New Import for CORS

# --- CORE LIBRARIES IMPORTS ---
from typing import List, Optional, Any
import os
import sys
import argparse
import re
import time
import hashlib 
import random 
import string 

# --- DATA AND DB LIBRARIES ---
import pandas as pd
import numpy as np
from pymongo import MongoClient
# from dotenv import load_dotenv # REMOVED: Railway injects environment variables directly
from bson.objectid import ObjectId # Needed for MongoDB _id conversion

# --- SKLEARN FOR MODELING ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load environment variables from the .env file - REMOVING THIS LINE
# load_dotenv()

# --- FLASK APP INSTANCE ---
app = Flask(__name__)

# --- CRITICAL STEP: CONFIGURE CORS ---
# You must update this with your actual GitHub Pages URL later for security.
# For now, we allow all origins ('*') for easy testing, but restrict methods.
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "https://your-frontend-domain.com",
    "http://localhost:50949/"
]}})

# --- CONFIGURATION (UNCHANGED) ---
# NOTE: The default here is for local testing. In Railway, this MUST be overridden
# by the MONGO_URI variable injected by the MongoDB plugin.
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "telematics_risk_db"
RAW_DATA_COLLECTION = "raw_telematics"
CLAIMS_COLLECTION = "claims_history"
USERS_COLLECTION = "users" 
DEVICE_COL_CANDIDATES = ["device_id", "deviceid", "deviceId", "DeviceId", "device", "id", "ID"]
VALUE_COL_CANDIDATES = ["value", "values", "Value", "Values"]
CLAIMS_FILE_NAME = "Claims_History.csv"
FEATURES_FILE_NAME = "Telematics_Risk_Features_FULL.csv"

# --- UTILITY AND CORE LOGIC FUNCTIONS (Modified to fit API needs) ---

def get_mongo_client():
    """Returns a connected MongoClient using the URI."""
    return MongoClient(MONGO_URI)

def hash_password(password: str, salt: str = "secure_salt") -> str:
    """Simulates secure password hashing."""
    pw_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    return hashlib.sha256(pw_bytes + salt_bytes).hexdigest()

def verify_password(stored_hash: str, provided_password: str, salt: str = "secure_salt") -> bool:
    """Verifies a provided password against a stored hash."""
    return stored_hash == hash_password(provided_password, salt)

def get_user_by_credentials(username: str, password: str):
    """Fetches a user document from MongoDB and verifies the password."""
    try:
        with get_mongo_client() as client:
            db = client[DATABASE_NAME]
            collection = db[USERS_COLLECTION]
            user_data = collection.find_one({"username": username})
            
            if user_data:
                if verify_password(user_data['hashed_password'], password):
                    return user_data
            return None
                
    except Exception as e:
        print(f"❌ Error fetching user data from MongoDB: {e}")
        return None

def load_csv(path: str) -> Optional[pd.DataFrame]:
    """Loads the CSV file and handles errors, returns None on failure."""
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as ex:
        print(f"Error reading CSV: {ex}")
        return None

def haversine_distance(lat1, lon1, lat2, lon2, R=6371):
    """Calculates the great-circle distance (in km) between two latitude/longitude points."""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def calculate_all_features(device_id: Optional[str], source_device_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Performs the full feature engineering pipeline.
    Returns the resulting features DataFrame or None on error.
    """
    filter_id = source_device_id if source_device_id else device_id
    df = load_csv("Telematicsdata.csv")
    if df is None: return None

    # Filter by the actual data source ID
    if filter_id:
        df = df[df['deviceId'].astype(str) == str(filter_id)].copy()
        if df.empty: return None
            
    position_df = df[df['variable'] == 'POSITION'].copy()
    if position_df.empty: 
        # Create a zero-feature DataFrame for missing data
        if device_id:
             return pd.DataFrame({'deviceId': [device_id], 'total_distance_km': [0.0], 'hard_brake_rate_per_1000km': [0.0], 'hard_accel_rate_per_1000km': [0.0], 'percent_time_high_speed': [0.0]})
        return None
        
    # ... (Lat/Lon extraction, time prep, distance, speed, acceleration/deceleration logic)
    # Handle the 'value' column extraction (coordinate parsing)
    parts = position_df['value'].astype(str).str.split(r'[,;\s]+', n=2, expand=True)
    position_df.loc[:, 'latitude'] = pd.to_numeric(parts.iloc[:, 0], errors='coerce')
    position_df.loc[:, 'longitude'] = pd.to_numeric(parts.iloc[:, 1], errors='coerce')
    position_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    if position_df.empty:
        if device_id:
             return pd.DataFrame({'deviceId': [device_id], 'total_distance_km': [0.0], 'hard_brake_rate_per_1000km': [0.0], 'hard_accel_rate_per_1000km': [0.0], 'percent_time_high_speed': [0.0]})
        return None
        
    position_df['datetime'] = pd.to_datetime(position_df['timestamp'])
    position_df['time_sec'] = position_df['datetime'].astype(np.int64) // 10**9
    grouped = position_df.groupby('deviceId')
    
    # Distance calculation
    position_df.loc[:, 'latitude_prev'] = grouped['latitude'].shift(1)
    position_df.loc[:, 'longitude_prev'] = grouped['longitude'].shift(1)
    position_df.loc[:, 'distance_km'] = haversine_distance(
        position_df['latitude_prev'], position_df['longitude_prev'],
        position_df['latitude'], position_df['longitude']
    )
    position_df['time_diff_sec'] = grouped['time_sec'].diff() 
    
    # Speed calculation
    position_df['speed_kmh'] = np.where(
        position_df['time_diff_sec'] > 0,
        position_df['distance_km'] / (position_df['time_diff_sec'] / 3600.0),
        0.0 
    )

    # Hard Maneuver calculation
    position_df['speed_ms'] = position_df['speed_kmh'] * (1000 / 3600)
    position_df['delta_speed_ms'] = grouped['speed_ms'].diff()
    position_df['delta_time_sec'] = grouped['time_sec'].diff()
    position_df['acceleration_ms2'] = np.where(
        position_df['delta_time_sec'] > 0,
        position_df['delta_speed_ms'] / position_df['delta_time_sec'],
        0.0
    )
    HARD_BRAKE_THRESHOLD = -4.0
    HARD_ACCEL_THRESHOLD = 3.0
    position_df['is_hard_brake'] = position_df['acceleration_ms2'] <= HARD_BRAKE_THRESHOLD
    position_df['is_hard_accel'] = position_df['acceleration_ms2'] >= HARD_ACCEL_THRESHOLD

    # Aggregation
    grouped_features = position_df.groupby('deviceId').agg(
        total_distance_km=('distance_km', 'sum'),
        total_hard_brakes=('is_hard_brake', 'sum'),
        total_hard_accels=('is_hard_accel', 'sum'),
        total_driving_time_sec=('time_diff_sec', 'sum')
    ).reset_index()
    safe_distance = grouped_features['total_distance_km'].replace(0, 1e-6)
    grouped_features['hard_brake_rate_per_1000km'] = (grouped_features['total_hard_brakes'] / safe_distance) * 1000
    grouped_features['hard_accel_rate_per_1000km'] = (grouped_features['total_hard_accels'] / safe_distance) * 1000

    # Speed Risk
    CLEAN_SPEED_CAP = 200.0
    clean_moving_df = position_df[
        (position_df['speed_kmh'] > 0) & 
        (position_df['speed_kmh'] <= CLEAN_SPEED_CAP)
    ].copy()
    HIGH_SPEED_THRESHOLD = 90.0
    high_speed_time = clean_moving_df[
        clean_moving_df['speed_kmh'] >= HIGH_SPEED_THRESHOLD
    ].groupby('deviceId')['time_diff_sec'].sum().reset_index()
    high_speed_time.columns = ['deviceId', 'time_high_speed_sec']
    total_time_moving = clean_moving_df.groupby('deviceId')['time_diff_sec'].sum().reset_index()
    total_time_moving.columns = ['deviceId', 'total_time_moving_sec']
    speed_risk_df = pd.merge(total_time_moving, high_speed_time, on='deviceId', how='left').fillna(0)
    speed_risk_df['percent_time_high_speed'] = (speed_risk_df['time_high_speed_sec'] / speed_risk_df['total_time_moving_sec'].replace(0, 1e-6)) * 100
    
    # Final Merge and Output Assignment
    final_risk_features = pd.merge(grouped_features, speed_risk_df[['deviceId', 'percent_time_high_speed']], on='deviceId', how='left').fillna(0)

    if filter_id and device_id and (filter_id != device_id):
        # If running for a new user, replace the deviceId in the output
        final_risk_features.loc[:, 'deviceId'] = device_id
        
    return final_risk_features[['deviceId', 'total_distance_km', 'hard_brake_rate_per_1000km', 'hard_accel_rate_per_1000km', 'percent_time_high_speed']]


def get_simulated_premium(features: pd.Series) -> tuple[float, str]:
    """Simulates calling an external insurance API to get a premium estimate."""
    hard_brake_score = np.interp(
        features['hard_brake_rate_per_1000km'], [0, 5, 10, 20], [1, 3, 6, 10]
    )
    high_speed_score = np.interp(
        features['percent_time_high_speed'], [0, 2, 5, 10], [1, 3, 6, 10]
    )
    avg_risk_score = (hard_brake_score + high_speed_score) / 2
    base_premium = 1100.0
    adjustment_factor = (avg_risk_score / 10.0) * 0.7 
    premium = base_premium * (1 + adjustment_factor)
    
    if premium > 1550:
        quote_message = "External Vendor - High Risk Quote."
    elif premium > 1250:
        quote_message = "External Vendor - Standard Quote."
    else:
        quote_message = "External Vendor - Best Rate."
        
    return float(premium), quote_message

def get_premium_estimates_for_api(user_data) -> dict:
    """
    Core logic to calculate estimates, modified to RETURN a dictionary 
    instead of printing results, suitable for API response.
    """
    device_id = user_data['device_id']
    source_device_id = user_data.get('source_device_id', device_id)
    
    # 1. Load Data for Internal Model
    final_risk_features = load_csv(FEATURES_FILE_NAME)
    claims_data = load_csv(CLAIMS_FILE_NAME)

    if final_risk_features is None or claims_data is None:
        return {"error": "Required feature or claims files not found on the server. Please run the batch feature calculation first."}

    # Filter features based on the data source ID
    target_features_df = final_risk_features[final_risk_features['deviceId'].astype(str) == str(source_device_id)].copy()
    
    if target_features_df.empty:
        # Calculate features in real-time if missing from the batch file
        target_features_df = calculate_all_features(device_id=device_id, source_device_id=source_device_id)
        if target_features_df is None or target_features_df.empty:
            return {"error": "Could not generate real-time features. Cannot provide an estimate."}

    target_features = target_features_df.iloc[0]
    
    features_cols = ['hard_brake_rate_per_1000km', 'percent_time_high_speed', 'total_distance_km']
    full_model_data = pd.merge(final_risk_features, claims_data[['deviceId', 'has_claim']], on='deviceId', how='left')
    full_y = full_model_data['has_claim'].fillna(0).astype(int)
    
    if full_y.nunique() <= 1:
        internal_risk_score = 0.5 
        internal_premium = 1000.0
    else:
        # 2. Train and Predict Internal Risk Score
        X_train, _, y_train, _ = train_test_split(full_model_data[features_cols], full_y, test_size=0.01, random_state=42)
        model = LogisticRegression(random_state=42, solver='liblinear')
        model.fit(X_train, y_train)
        X_predict = target_features_df[features_cols] 
        internal_risk_score = model.predict_proba(X_predict)[:, 1][0]
        
        # 3. Calculate Internal Premium
        BASE_PREMIUM = 1000.0
        RISK_MULTIPLIER = 1.0 
        internal_premium = BASE_PREMIUM * (1 + internal_risk_score * RISK_MULTIPLIER)

    # 4. CALL THE SIMULATED EXTERNAL API 
    external_premium, external_message = get_simulated_premium(target_features)
    
    # 5. Prepare the API response structure
    return {
        "user_info": {
            "name": user_data['name'],
            "device_id": device_id,
            "has_claim_history": bool(user_data.get('has_claim', 0))
        },
        "telematics_features": {
            "total_distance_km": float(target_features['total_distance_km']),
            "hard_brake_rate_per_1000km": float(target_features['hard_brake_rate_per_1000km']),
            "hard_accel_rate_per_1000km": float(target_features['hard_accel_rate_per_1000km']),
            "percent_time_high_speed": float(target_features['percent_time_high_speed']),
        },
        "premium_estimates": {
            "internal_model": {
                "risk_score": float(internal_risk_score),
                "annual_premium": float(internal_premium)
            },
            "external_vendor": {
                "annual_premium": float(external_premium),
                "quote_message": external_message
            }
        }
    }


# --- FLASK API ROUTES ---

@app.route('/login', methods=['POST'])
def login_route():
    """Handles user login and returns user data."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400
        
    username = data.get('username')
    password = data.get('password')

    user_data = get_user_by_credentials(username, password)
    
    if user_data:
        # Ensure MongoDB ObjectId is converted to a serializable string
        if '_id' in user_data:
            user_data['_id'] = str(user_data['_id'])
        # Remove the hashed password before sending to the frontend
        if 'hashed_password' in user_data:
            del user_data['hashed_password'] 

        return jsonify({
            "message": "Login successful",
            "user": user_data
        }), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401

@app.route('/features', methods=['POST'])
def get_features_route():
    """Calculates and returns the user's latest telematics features."""
    data = request.get_json()
    if not data or 'device_id' not in data:
        return jsonify({"error": "User data (device_id and source_device_id) is required"}), 400

    device_id = data.get('device_id')
    source_device_id = data.get('source_device_id', device_id)
    
    features_df = calculate_all_features(device_id=device_id, source_device_id=source_device_id)

    if features_df is None or features_df.empty:
        return jsonify({"error": f"Failed to calculate features for device {device_id}."}), 500

    # Convert DataFrame row to dictionary for JSON output
    features_dict = features_df.iloc[0].to_dict()

    # Ensure all numpy float types are converted to standard Python floats for JSON
    for key, value in features_dict.items():
        if isinstance(value, np.float64):
            features_dict[key] = float(value)
            
    return jsonify(features_dict), 200

@app.route('/estimate', methods=['POST'])
def get_estimate_route():
    """Calculates and returns both internal and external premium estimates."""
    user_data = request.get_json()
    if not user_data or 'device_id' not in user_data:
        return jsonify({"error": "User data (including device_id) is required"}), 400

    results = get_premium_estimates_for_api(user_data)

    if "error" in results:
        return jsonify(results), 500
        
    return jsonify(results), 200

# --- MAIN RUN BLOCK ---
# NOTE: This block is primarily for local testing (python app.py). 
# The Docker CMD will use Gunicorn for production instead.

if __name__ == '__main__':
    print("\n----------------------------------------------------")
    print("🚀 Telematics Risk Model API is STARTING...")
    print(f"MongoDB URI: {MONGO_URI}")
    print("Access the API at: http://127.0.0.1:5000/")
    print("----------------------------------------------------\n")
    app.run(debug=True)