import pandas as pd
from pymongo import MongoClient
from flask import Flask, jsonify, request
import os
import random
import numpy as np

# --- 1. FLASK APPLICATION SETUP ---
app = Flask(__name__)
# Enable pretty printing of JSON responses
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


# --- 2. CONFIGURATION CONSTANTS ---
# NOTE: MONGO_URI should be provided in the environment when running the Flask app
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = "telematics_risk_db"
FEATURES_COLLECTION = "RiskFeaturesFull"  # The collection that is currently missing/skipped
CLAIMS_COLLECTION = "ClaimsHistory"
TELEMATICS_COLLECTION = "TelematicsData"


# --- 3. MONGODB HELPER FUNCTIONS ---

def load_mongo_collection_to_df(collection_name):
    """
    Loads a MongoDB collection into a pandas DataFrame.
    Returns None if the collection is empty, not found, or if an error occurs.
    """
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[collection_name]
        
        # Check if collection is empty (using estimate is fast)
        if collection.estimated_document_count() == 0:
            print(f"Collection '{collection_name}' is empty or not found.")
            client.close()
            return None
        
        # Load all documents from the collection
        df = pd.DataFrame(list(collection.find({})))
        client.close()
        
        if '_id' in df.columns:
             df = df.drop(columns=['_id'])

        return df
    except Exception as e:
        # This catches connection errors or missing collections entirely
        print(f"Error loading collection {collection_name}: {e}")
        return None 

# --- 4. FEATURE CALCULATION (REAL-TIME FALLBACK) ---

def calculate_all_features(device_id, source_device_id):
    """
    Simulated function to calculate features from raw telematics data (TELEMATICS_COLLECTION).
    This runs when pre-calculated features (RiskFeaturesFull) are not available.
    """
    print(f"--- Running REAL-TIME FEATURE CALCULATION for ID: {source_device_id} ---")
    
    raw_data_df = load_mongo_collection_to_df(TELEMATICS_COLLECTION)
    
    if raw_data_df is None:
        print("Raw Telematics Data is unavailable for real-time calculation.")
        return None

    # Filter data for the specific device (assuming 'deviceId' field holds the source_device_id)
    # NOTE: You may need to adjust the column name 'deviceId' to match your actual schema
    target_raw_data = raw_data_df[raw_data_df['deviceId'] == source_device_id]
    
    if target_raw_data.empty:
        print(f"No raw telematics data found for device ID: {source_device_id}")
        return None

    # --- SIMULATION OF FEATURE GENERATION ---
    
    num_records = len(target_raw_data)
    
    # Simulate realistic risk metrics based on records found
    harsh_braking_rate = random.uniform(0.01, 0.25)
    night_driving_ratio = random.uniform(0.05, 0.40)
    total_distance_miles = (num_records / 10) * random.uniform(0.5, 1.5) 
    
    # Simple risk score generation (higher means higher risk)
    risk_score = 100 - np.clip(total_distance_miles / 50, 10, 50) + (harsh_braking_rate * 100) * random.uniform(0.5, 1.5)
    risk_score = int(np.clip(risk_score, 30, 95))
    
    features = {
        'source_device_id': source_device_id,
        'feature_harsh_braking_rate': harsh_braking_rate,
        'feature_night_driving_ratio': night_driving_ratio,
        'TotalDistanceMiles': round(total_distance_miles, 2),
        'ModelRiskScore': risk_score # This is the key output used for premium calculation
    }
    
    return pd.DataFrame([features])


# --- 5. CORE API ESTIMATION FUNCTION (FIXED LOGIC) ---

def get_premium_estimates_for_api(device_id, source_device_id):
    """
    Processes data for a given device ID and returns a premium estimate.
    Includes logic to fall back to real-time calculation if pre-calculated 
    features (RiskFeaturesFull) are missing.
    """
    
    # 1. Load ESSENTIAL Data (Claims History is mandatory)
    claims_data = load_mongo_collection_to_df(CLAIMS_COLLECTION)
    
    if claims_data is None:
        # Failure if essential claims history data is missing
        return {"error": "Required Claims History data collection not found or empty in MongoDB. Cannot run model."}

    # 2. Try to Load Pre-calculated Features (OPTIONAL Primary Source)
    final_risk_features = load_mongo_collection_to_df(FEATURES_COLLECTION)
    target_features_df = pd.DataFrame() # Initialize as empty

    if final_risk_features is not None:
        # If pre-calculated features exist, try to find the target ID in them
        target_features_df = final_risk_features[final_risk_features['source_device_id'] == source_device_id]
        
        # If found, use this data.
        if not target_features_df.empty:
            print(f"Found pre-calculated features for {source_device_id}.")
    else:
        # This branch executes if FEATURES_COLLECTION is missing (the root cause of the original error)
        print(f"Pre-calculated features collection '{FEATURES_COLLECTION}' is missing. Proceeding to real-time calculation.")
        
    if target_features_df.empty:
        # 3. FALLBACK: If data is missing or target ID not found, calculate in real-time
        target_features_df = calculate_all_features(device_id=device_id, source_device_id=source_device_id)
        
        if target_features_df is None or target_features_df.empty:
            # Final failure if real-time calculation also fails due to missing raw data
            return {"error": "Could not generate real-time features. Raw Telematics Data may be missing or the device ID is not present in TelematicsData."}

    # 4. Run Estimation Model
    
    features = target_features_df.iloc[0].to_dict()
    risk_score = features.get('ModelRiskScore', 50) 
    
    # Mock premium calculation based on risk score
    base_premium = 650.00
    if risk_score > 75:
        premium_adjustment = 1.30
        risk_level = "High Risk"
        message_prefix = "Premium reflects high risk based on driving behavior."
    elif risk_score < 50:
        premium_adjustment = 0.75
        risk_level = "Low Risk"
        message_prefix = "Congratulations! You've earned a discount for safe driving."
    else:
        premium_adjustment = 1.0
        risk_level = "Medium Risk"
        message_prefix = "Premium is standard based on average risk."
        
    estimated_premium = round(base_premium * premium_adjustment, 2)
    
    # 5. Return Success Response
    return {
        "success": True,
        "device_id": source_device_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "estimated_premium": estimated_premium,
        "message": message_prefix
    }


# --- 6. FLASK ROUTE ---

@app.route('/api/risk_estimate', methods=['POST'])
def risk_estimate_endpoint():
    """Endpoint for calculating the premium risk estimate."""
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON data in request"}), 400

    # Ensure required fields are present
    device_id = data.get('device_id')
    source_device_id = data.get('source_device_id')

    if not device_id or not source_device_id:
        return jsonify({"error": "Missing 'device_id' or 'source_device_id' in request body."}), 400

    # Call the core processing function
    result = get_premium_estimates_for_api(
        device_id=device_id, 
        source_device_id=source_device_id
    )
    
    # Check if the result is an error message
    if 'error' in result:
        # Use 500 for internal errors (like missing mandatory data)
        return jsonify(result), 500 
    
    return jsonify(result), 200

# --- 7. APPLICATION RUNNER ---
if __name__ == '__main__':
    # Default host and port for local development
    # Ensure you are running MongoDB locally or have MONGO_URI set in your environment
    app.run(debug=True, host='0.0.0.0', port=5000)