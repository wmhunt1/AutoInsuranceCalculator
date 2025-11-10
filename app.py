#!/usr/bin/env python3
"""
Flask API for Telematics Risk Modeling.

CRITICAL UPDATE: Data loading and model training are now performed once on 
application startup, caching the results globally for performance and stability. 
The core estimation logic now uses the cached model or a robust heuristic fallback.
"""

# --- FLASK IMPORTS ---
from flask import Flask, request, jsonify
from flask_cors import CORS 

# --- CORE LIBRARIES IMPORTS ---
from typing import List, Optional, Any
import os
import time
import hashlib 
import random 
import string 
import sys 
import atexit # Added for cleanup if needed, but primarily for robustness

# --- DATA AND DB LIBRARIES ---
import pandas as pd
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId 
from pymongo.errors import ConnectionFailure, OperationFailure

# --- SKLEARN FOR MODELING ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# --- FLASK APP INSTANCE ---
app = Flask(__name__)

# --- CRITICAL STEP: CONFIGURE CORS ---
CORS(app, resources={r"/*": {"origins": [
    "https://wmhunt1.github.io/AutoInsuranceCalculatorUI", 
    "http://localhost:5000",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "https://wmhunt1.github.io" 
], "methods": ["GET", "POST", "OPTIONS"]}})

# --- CONFIGURATION (UPDATED TO USE COLLECTIONS) ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "telematics_risk_db"
USERS_COLLECTION = "users" 

# Collections storing the modeling data
TELEMATICS_COLLECTION = "raw_telematics" 
CLAIMS_COLLECTION = "claims_history" 
FEATURES_COLLECTION = "RiskFeaturesFull"

# --- GLOBAL CACHED DATA AND MODEL (NEW) ---
TRAINING_FEATURES_DF: Optional[pd.DataFrame] = None # Cached features for training set
CLAIMS_DF: Optional[pd.DataFrame] = None            # Cached claims data
RISK_MODEL: Optional[LogisticRegression] = None     # Cached trained model
DATA_LOAD_STATUS: str = "PENDING"                   # Status of the initial load
FEATURES_COLS = ['hard_brake_rate_per_1000km', 'percent_time_high_speed', 'total_distance_km']

# --- UTILITY AND CORE LOGIC FUNCTIONS ---

def get_mongo_client():
    """Returns a connected MongoClient using the URI."""
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

def load_mongo_collection_to_df(collection_name: str) -> Optional[pd.DataFrame]:
    """
    Loads all documents from a specified MongoDB collection into a pandas DataFrame.
    """
    start_time = time.time()
    print(f"🔄 Attempting to load collection: '{collection_name}'...")
    
    try:
        with get_mongo_client() as client:
            client.admin.command('ping')
            db = client[DATABASE_NAME]
            collection = db[collection_name]
            
            cursor = collection.find({})
            data = list(cursor)
            
            if not data:
                print(f"⚠️ Collection '{collection_name}' found, but is empty.")
                return None
            
            df = pd.DataFrame(data)
            
            if '_id' in df.columns:
                df.drop(columns=['_id'], inplace=True)
                
            load_time = time.time() - start_time
            print(f"🎉 Collection '{collection_name}' loaded successfully. Rows: {len(df)}. Time: {load_time:.2f}s")
            return df
            
    except ConnectionFailure as e:
        print(f"❌ CRITICAL CONNECTION FAILURE (Check MONGO_URI): {e}", file=sys.stderr)
        return None
    except OperationFailure as e:
        print(f"❌ MONGODB OPERATION ERROR (Check DB Name/Permissions): {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"❌ General Error loading data from MongoDB collection '{collection_name}': {e}", file=sys.stderr)
        return None
        
# --- USER MANAGEMENT FUNCTIONS (Unchanged, included for completeness) ---

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
        print(f"❌ Error fetching user data from MongoDB: {e}", file=sys.stderr)
        return None

def create_new_user(username: str, password: str, name: str, source_device_id: str, device_id: str):
    """Creates a new user in MongoDB with initial data."""
    try:
        with get_mongo_client() as client:
            db = client[DATABASE_NAME]
            collection = db[USERS_COLLECTION]
            
            if collection.find_one({"username": username}):
                return {"error": "Username already exists."}, None
            
            user_doc = {
                "username": username,
                "name": name,
                "hashed_password": hash_password(password),
                "device_id": device_id,
                "source_device_id": source_device_id, 
                "created_at": time.time()
            }
            
            result = collection.insert_one(user_doc)
            user_doc['_id'] = str(result.inserted_id)
            del user_doc['hashed_password']
            return None, user_doc
            
    except Exception as e:
        print(f"❌ Error creating user in MongoDB: {e}", file=sys.stderr)
        return {"error": f"Database error during creation: {e}"}, None

# --- FEATURE CALCULATION LOGIC (Unchanged, relies on raw telematics load) ---

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
    Performs the full feature engineering pipeline from raw telematics data 
    for a specific device ID (or source ID).
    """
    filter_id = source_device_id if source_device_id else device_id
    
    # NOTE: This still loads raw data synchronously, which is necessary when features aren't pre-calculated.
    df = load_mongo_collection_to_df(TELEMATICS_COLLECTION)
    if df is None: return None

    df['deviceId'] = df['deviceId'].astype(str)
    
    if filter_id:
        df = df[df['deviceId'].astype(str) == str(filter_id)].copy()
        if df.empty: return None
            
    position_df = df[df['variable'] == 'POSITION'].copy()
    
    if position_df.empty: 
        if device_id:
            # Return zero features if no position data is found
            return pd.DataFrame({'deviceId': [device_id], 'total_distance_km': [0.0], 'hard_brake_rate_per_1000km': [0.0], 'hard_accel_rate_per_1000km': [0.0], 'percent_time_high_speed': [0.0]})
        return None
        
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
        final_risk_features.loc[:, 'deviceId'] = device_id
        
    return final_risk_features[['deviceId', 'total_distance_km', 'hard_brake_rate_per_1000km', 'hard_accel_rate_per_1000km', 'percent_time_high_speed']]


# --- STARTUP AND CACHING LOGIC (NEW) ---

def load_and_train_model_on_startup():
    """
    Loads necessary static data from MongoDB and trains the risk model once 
    upon application startup. Caches results globally.
    """
    global TRAINING_FEATURES_DF, CLAIMS_DF, RISK_MODEL, DATA_LOAD_STATUS
    
    # 1. Load Claims Data (Mandatory)
    CLAIMS_DF = load_mongo_collection_to_df(CLAIMS_COLLECTION)
    if CLAIMS_DF is None:
        DATA_LOAD_STATUS = f"FAILURE: Claims data ('{CLAIMS_COLLECTION}') missing or empty. Cannot run model."
        print(f"🚨 {DATA_LOAD_STATUS}", file=sys.stderr)
        return

    CLAIMS_DF['deviceId'] = CLAIMS_DF['deviceId'].astype(str)
    
    # 2. Try to Load Pre-calculated Features (Optional for fallback)
    features_df = load_mongo_collection_to_df(FEATURES_COLLECTION)
    
    if features_df is not None and not features_df.empty:
        features_df['deviceId'] = features_df['deviceId'].astype(str)
        
        # Merge features with claims data for training
        full_model_data = pd.merge(features_df, CLAIMS_DF[['deviceId', 'has_claim']], on='deviceId', how='left')
        
        # Fill NaN claims (assuming no claim if not in claims collection)
        full_y = full_model_data['has_claim'].fillna(0).astype(int)
        
        # Check for sufficient data diversity
        if full_y.nunique() > 1 and len(full_y) > 10:
            try:
                # Train model
                X_train, _, y_train, _ = train_test_split(full_model_data[FEATURES_COLS], full_y, test_size=0.01, random_state=42)
                
                RISK_MODEL = LogisticRegression(random_state=42, solver='liblinear')
                RISK_MODEL.fit(X_train, y_train)
                
                TRAINING_FEATURES_DF = features_df
                DATA_LOAD_STATUS = "SUCCESS: Full model trained and features cached."
                return
                
            except Exception as e:
                DATA_LOAD_STATUS = f"FAILURE: Model training failed ({e}). Using heuristic fallback."
                print(f"🚨 Model training failed on startup: {e}", file=sys.stderr)
        else:
            DATA_LOAD_STATUS = "SUCCESS: Features loaded but model diversity insufficient. Using heuristic."
            TRAINING_FEATURES_DF = features_df
            print("⚠️ Insufficient data diversity for complex model training on startup. Proceeding with heuristic fallback.")
            
    else:
        DATA_LOAD_STATUS = "WARNING: Features collection missing/empty. Using heuristic model and real-time feature calculation."
        print(f"⚠️ Features collection ('{FEATURES_COLLECTION}') not found or empty. Using heuristic fallback.")

    # If model training failed or features were missing, RISK_MODEL remains None, which triggers the heuristic logic.
    RISK_MODEL = None
    TRAINING_FEATURES_DF = features_df 

# --- ESTIMATION LOGIC (Refactored to use cache) ---

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
    Core logic to calculate estimates for API response using cached data/model.
    This function no longer performs heavy synchronous data loading or model training.
    """
    device_id = user_data['device_id']
    source_device_id = user_data.get('source_device_id', device_id)
    name = user_data.get('name', 'User')

    # 1. Get Target Features: Check cache first, otherwise calculate in real-time
    target_features_df = pd.DataFrame() 
    
    if TRAINING_FEATURES_DF is not None:
        # Check cached training features for the target device (uses source_device_id)
        target_features_df = TRAINING_FEATURES_DF[
            TRAINING_FEATURES_DF['deviceId'] == str(source_device_id)
        ].copy()

    if target_features_df.empty:
        # Fallback: Calculate real-time features from raw telematics
        print(f"⚙️ Calculating real-time features for {source_device_id}...")
        target_features_df = calculate_all_features(device_id=device_id, source_device_id=source_device_id)
        
        if target_features_df is None or target_features_df.empty:
            return {"error": "Failed to generate features. Raw Telematics Data may be missing or the device ID is not present."}

    target_features = target_features_df.iloc[0]
    
    # 2. Internal Model Prediction
    if RISK_MODEL is not None:
        # Use the cached Logistic Regression Model
        X_predict = target_features_df[FEATURES_COLS] 
        internal_risk_score = RISK_MODEL.predict_proba(X_predict)[:, 1][0]
        model_source = "LogReg Model (Cached)"
    else:
        # Use the Heuristic Score (Fallback)
        hard_brake_rate = target_features.get('hard_brake_rate_per_1000km', 0)
        high_speed_perc = target_features.get('percent_time_high_speed', 0)
        
        # Heuristic score based on driving metrics: normalize rates to a 0-1 range
        norm_brake = np.clip(hard_brake_rate / 15.0, 0.0, 1.0)
        norm_speed = np.clip(high_speed_perc / 10.0, 0.0, 1.0)
        internal_risk_score = (norm_brake * 0.6) + (norm_speed * 0.4)
        internal_risk_score = float(np.clip(internal_risk_score, 0.1, 0.9))
        model_source = "Heuristic Fallback"

    # 3. Calculate Internal Premium
    BASE_PREMIUM = 1000.0
    RISK_MULTIPLIER = 1.0 
    internal_premium = BASE_PREMIUM * (1 + internal_risk_score * RISK_MULTIPLIER)

    # 4. CALL THE SIMULATED EXTERNAL API 
    external_premium, external_message = get_simulated_premium(target_features)
    
    # 5. Determine Claim History Status
    has_claim_history = 0
    if CLAIMS_DF is not None:
        claim_row = CLAIMS_DF[CLAIMS_DF['deviceId'] == str(source_device_id)]
        if not claim_row.empty:
            has_claim_history = claim_row['has_claim'].iloc[0]
            
    # 6. Prepare the API response structure
    return {
        "user_info": {
            "name": name,
            "device_id": device_id,
            "source_device_id": source_device_id,
            "has_claim_history": bool(has_claim_history),
            "model_source": model_source # Added for transparency
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


# --- FLASK API ROUTES (Unchanged, rely on refactored logic) ---

@app.route('/login', methods=['POST'])
def login_route():
    """Handles user login and returns user data."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        username = data.get('username')
        password = data.get('password')

        user_data = get_user_by_credentials(username, password)
        
        if user_data:
            if '_id' in user_data:
                user_data['_id'] = str(user_data['_id'])
            if 'hashed_password' in user_data:
                del user_data['hashed_password'] 

            return jsonify({
                "message": "Login successful",
                "user": user_data
            }), 200
        else:
            return jsonify({"error": "Invalid username or password"}), 401
            
    except Exception as e:
        print(f"❌ Uncaught error in /login: {e}", file=sys.stderr)
        return jsonify({"status": "error", "code": 500, "message": "Internal server error during login"}), 500

@app.route('/create_user', methods=['POST'])
def create_user_route():
    """Handles new user creation."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        username = data.get('username')
        password = data.get('password')
        name = data.get('name', username)
        device_id = data.get('device_id', 'NEW_USER_' + ''.join(random.choices(string.digits, k=5)))
        source_device_id = data.get('source_device_id', 'driver1') 

        if not all([username, password]):
            return jsonify({"error": "Username and password are required."}), 400

        error, new_user = create_new_user(username, password, name, source_device_id, device_id)
        
        if error:
            status_code = 409 if 'Username already exists' in error['error'] else 500
            return jsonify(error), status_code
        else:
            return jsonify({
                "message": "User created and logged in successfully",
                "user": new_user
            }), 201
            
    except Exception as e:
        print(f"❌ Uncaught error in /create_user: {e}", file=sys.stderr)
        return jsonify({"status": "error", "code": 500, "message": "Internal server error during user creation"}), 500


@app.route('/features', methods=['POST'])
def get_features_route():
    """Calculates and returns the user's latest telematics features."""
    try:
        data = request.get_json()
        if not data or 'device_id' not in data:
            return jsonify({"error": "User data (device_id and source_device_id) is required"}), 400

        device_id = data.get('device_id')
        source_device_id = data.get('source_device_id', device_id)
        
        # This route still performs a real-time calculation because it should always return the freshest data
        features_df = calculate_all_features(device_id=device_id, source_device_id=source_device_id)

        if features_df is None or features_df.empty:
            print(f"❌ Feature calculation failed for device {device_id}. Check logs for data loading errors.")
            return jsonify({"error": f"Failed to calculate features for device {device_id}. Raw Telematics Data may be missing."}), 500

        features_dict = features_df.iloc[0].to_dict()

        for key, value in features_dict.items():
            if isinstance(value, np.float64):
                features_dict[key] = float(value)
                
        return jsonify(features_dict), 200
        
    except Exception as e:
        print(f"❌ Uncaught error in /features route: {e}", file=sys.stderr)
        return jsonify({"status": "error", "code": 500, "message": f"Internal server error while calculating features: {e}"}), 500

@app.route('/estimate', methods=['POST'])
def get_estimate_route():
    """Calculates and returns both internal and external premium estimates."""
    try:
        user_data = request.get_json()
        if not user_data or 'device_id' not in user_data:
            return jsonify({"error": "User data (including device_id) is required"}), 400

        results = get_premium_estimates_for_api(user_data)

        if "error" in results:
            print(f"❌ Estimate generation failed: {results['error']}")
            return jsonify(results), 500
            
        return jsonify(results), 200
        
    except Exception as e:
        print(f"❌ Uncaught error in /estimate route: {e}", file=sys.stderr)
        return jsonify({"status": "error", "code": 500, "message": f"Internal server error while generating estimate: {e}"}), 500

# --- MAIN RUN BLOCK ---
if __name__ == '__main__':
    print("\n----------------------------------------------------")
    print("🚀 Telematics Risk Model API is STARTING...")
    
    # CRITICAL: Load data and train model once on startup
    load_and_train_model_on_startup() 
    
    print(f"MongoDB URI: {MONGO_URI}")
    print(f"Data Load Status: {DATA_LOAD_STATUS}") # Report status
    print("Access the API at: http://127.0.0.1:5000/")
    print("----------------------------------------------------\n")
    app.run(debug=True)