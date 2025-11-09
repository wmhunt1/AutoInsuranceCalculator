#!/usr/bin/env python3
"""
Simple console app for Telematics Risk Modeling.

This script implements a two-tiered CLI: a Login menu to select a user and 
a Session menu to run feature engineering and prediction for that user's device.
"""

from typing import List, Optional
import os
import sys
import argparse
import re

# --- THIRD-PARTY LIBRARIES ---
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

# --- SKLEARN FOR MODELING ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# import matplotlib.pyplot as plt # Not needed in CLI version

# Load environment variables from the .env file
load_dotenv()

# --- MongoDB Connection Details ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "telematics_risk_db"
RAW_DATA_COLLECTION = "raw_telematics"
CLAIMS_COLLECTION = "claims_history"
USERS_COLLECTION = "users" # <--- NEW COLLECTION

# --- CONSTANTS ---
DEVICE_COL_CANDIDATES = [
    "device_id", "deviceid", "deviceId", "DeviceId", "device", "id", "ID"
]
TIME_COL_CANDIDATES = [
    "timestamp", "time", "datetime", "date", "ts"
]
VALUE_COL_CANDIDATES = [
    "value", "values", "Value", "Values"
]
CLAIMS_FILE_NAME = "Claims_History.csv"
FEATURES_FILE_NAME = "Telematics_Risk_Features_FULL.csv"


# --- UTILITY FUNCTIONS ---

def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Tries to find the correct column name (case-insensitive) from a list of candidates."""
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def parse_args():
    """Handles command line arguments for file, display limits, and device filtering."""
    p = argparse.ArgumentParser(description="Read Telematicsdata.csv and display rows.")
    p.add_argument("--file", "-f", default="Telematicsdata.csv", help="Path to CSV file")
    p.add_argument("--head", "-n", type=int, default=20, help="Number of rows to show (use 0 for all)")
    p.add_argument("--device", "-d", help="Device id to filter (string match)")
    p.add_argument("--columns", "-c", help="Comma-separated columns to show (or 'all')")
    return p.parse_args()

def load_csv(path: str) -> pd.DataFrame:
    """Loads the CSV file and performs basic error checking."""
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(2)
    try:
        df = pd.read_csv(path)
    except Exception as ex:
        print(f"Error reading CSV: {ex}", file=sys.stderr)
        sys.exit(3)
    if df.empty:
        print("Warning: CSV loaded but contains no rows.", file=sys.stderr)
    return df

def pick_device_column(df: pd.DataFrame) -> Optional[str]:
    """Wrapper to find the device ID column."""
    return find_column(df.columns.tolist(), DEVICE_COL_CANDIDATES)

def filter_by_device(df: pd.DataFrame, device_col: str, device_id: str) -> pd.DataFrame:
    """Filters the DataFrame down to a single device ID."""
    return df[df[device_col].astype(str) == str(device_id)].copy()

def filter_by_variable(df: pd.DataFrame, variable_col: str, variable_name: str) -> pd.DataFrame:
    """Filters the DataFrame to include only rows matching a specific variable name (e.g., 'POSITION')."""
    filtered_df = df[df[variable_col].astype(str).str.upper() == variable_name.upper()].copy()
    return filtered_df

def select_columns(df: pd.DataFrame, cols_arg: Optional[str]) -> List[str]:
    """Selects columns for display based on user input."""
    if not cols_arg or cols_arg.strip().lower() == "all":
        return df.columns.tolist()
    chosen = [c.strip() for c in cols_arg.split(",") if c.strip()]
    missing = [c for c in chosen if c not in df.columns]
    if missing:
        print(f"Columns not found in CSV: {missing}", file=sys.stderr)
        sys.exit(4)
    return chosen

def extract_lat_lon_and_remove_value(df: pd.DataFrame, value_col: str) -> None:
    """
    Splits the 'value' column (e.g., "13.33,74.74") into separate 'latitude' and 'longitude' columns.
    Converts values to numeric, coercing errors to NaN.
    """
    parts = df[value_col].astype(str).str.split(r'[,;\s]+', n=2, expand=True)
    df["latitude"] = pd.to_numeric(parts.iloc[:, 0], errors="coerce")
    if parts.shape[1] > 1:
        df["longitude"] = pd.to_numeric(parts.iloc[:, 1], errors="coerce")
    else:
        df["longitude"] = pd.NA
    try:
        df.drop(columns=[value_col], inplace=True)
    except Exception:
        pass

def retrieve_data(value: str) -> tuple[pd.DataFrame, str | None]:
    """Handles loading, filtering by 'POSITION', and coordinate extraction."""
    args = parse_args()
    df = load_csv(args.file)

    VARIABLE_COL = "variable"
    POSITION_VAR = value

    if VARIABLE_COL in df.columns:
        df = filter_by_variable(df, VARIABLE_COL, POSITION_VAR)
    
    value_col = find_column(df.columns.tolist(), VALUE_COL_CANDIDATES)
    if value_col:
        try:
            extract_lat_lon_and_remove_value(df, value_col)
        except Exception as ex:
            print(f"Warning: failed to extract lat/lon from '{value_col}': {ex}", file=sys.stderr)

    device_col = pick_device_column(df)

    if args.device:
        if not device_col:
            sys.exit(5)
        df = filter_by_device(df, device_col, args.device)
        if df.empty:
            sys.exit(6)

    return df, device_col

def display_data(df: pd.DataFrame, device_col: str | None) -> None:
    """Displays the final DataFrame result to the console."""
    args = parse_args()

    # Information about device IDs for non-filtered views
    if device_col and not args.device:
        unique = df[device_col].dropna().unique()
        sample = list(map(str, unique[:10]))
        print(f"Sample device ids ({min(len(unique),10)} shown): {sample}")
        print("Pass --device <id> to filter the table.\n")
    elif not device_col:
        print("No device column automatically detected (candidates tried):", DEVICE_COL_CANDIDATES)

    cols = select_columns(df, args.columns)
    to_show = df[cols]

    # Use to_string() for full, non-truncated table output
    if args.head == 0:
        try:
            print(to_show.to_string(index=False))
        except Exception:
            print(to_show)
    else:
        print(to_show.head(args.head).to_string(index=False))

def haversine_distance(lat1, lon1, lat2, lon2, R=6371):
    """
    Calculates the great-circle distance (in km) between two latitude/longitude points.
    This function is now designed to take Pandas Series or NumPy arrays (vectorized).
    """
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula core calculation
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


# --- NEW USER MANAGEMENT FUNCTIONS ---

def get_mongo_client():
    """Returns a connected MongoClient using the URI."""
    return MongoClient(MONGO_URI)

def load_sample_users():
    """Defines and loads sample user data into the MongoDB users collection."""
    print("\n--- Loading Sample User Data ---")
    
    # NOTE: You must replace the device_id below with real IDs from your Telematicsdata.csv 
    # if you want to test with real data.
    sample_users = [
        {"user_id": "user_one_id", "name": "Alice Smith", "device_id": "zRYzhAEAHAABAAAKCRtcAAsAtwB1gBAQ"},
        {"user_id": "user_two_id", "name": "Bob Johnson", "device_id": "zRYzhAEAHAABAAAKCRtcAAsANAB0gBAQ"},
        {"user_id": "user_three_id", "name": "Charlie Brown", "device_id": "zRYzhAEAHAABAAAKCRtcAAsAGAB0gBAQ"},
    ]
    try:
        with get_mongo_client() as client:
            db = client[DATABASE_NAME]
            collection = db[USERS_COLLECTION]
            
            # Clear existing data and insert new data
            collection.delete_many({})
            result = collection.insert_many(sample_users)
            
            print(f"✅ Successfully loaded {len(result.inserted_ids)} sample users into '{USERS_COLLECTION}'.")
            print("--- Sample Users Loaded ---")
            for i, user in enumerate(sample_users):
                print(f"[{i+1}] {user['name']} (Device: {user['device_id']})")
                
    except Exception as e:
        print(f"❌ Failed to load sample users. Error: {e}")

def get_user_by_selection(choice: str):
    """Fetches a user document from MongoDB based on the numeric menu choice."""
    try:
        with get_mongo_client() as client:
            db = client[DATABASE_NAME]
            collection = db[USERS_COLLECTION]
            
            # Find the user based on the selected index
            users_list = list(collection.find({}))
            index = int(choice) - 1
            
            if 0 <= index < len(users_list):
                return users_list[index]
            else:
                print("Invalid user selection.")
                return None
                
    except Exception as e:
        print(f"❌ Error fetching user data. Did you run the migration/load users? Error: {e}")
        return None

# --- MONGODB MIGRATION LOGIC ---

def migrate_csv_to_mongodb(csv_file_path: str, collection_name: str, client: MongoClient):
    """Loads a CSV file and inserts its records into the specified MongoDB collection."""
    print(f"\nAttempting to load data from: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}. Skipping migration for this file.")
        return
    data_records = df.to_dict('records')
    db = client[DATABASE_NAME]
    collection = db[collection_name]
    try:
        if data_records:
            result = collection.insert_many(data_records)
            print(f"✅ Successfully inserted {len(result.inserted_ids)} records into '{collection_name}'.")
        else:
            print(f"Warning: CSV file {csv_file_path} was empty.")
    except Exception as e:
        print(f"❌ An error occurred during MongoDB insert for {collection_name}: {e}")

def run_migration():
    """Establishes MongoDB connection and runs the migration for all CSV files."""
    try:
        print(f"\n--- Attempting connection to MongoDB at: {MONGO_URI} ---")
        with get_mongo_client() as client:
            client.admin.command('ping')
            print("✨ Successfully connected to MongoDB.")
            migrate_csv_to_mongodb("Telematicsdata.csv", RAW_DATA_COLLECTION, client)
            migrate_csv_to_mongodb(CLAIMS_FILE_NAME, CLAIMS_COLLECTION, client)
    except Exception as e:
        print(f"🚨 FAILED to connect to MongoDB. Error: {e}")

# --- CORE FEATURE ENGINEERING (Updated to accept device_id) ---

def calculate_all_features(device_id: Optional[str]):
    """
    Performs the full feature engineering pipeline, either for a single device 
    or for ALL devices if device_id is None.
    """
    mode = device_id if device_id else 'ALL'
    print(f"\n--- Running Feature Engineering Pipeline for Device: {mode} ---")
    
    # 1. INITIAL DATA PREPARATION
    df = load_csv("Telematicsdata.csv")
    
    # Filter by device ID immediately if specified
    if device_id:
        df = df[df['deviceId'].astype(str) == str(device_id)].copy()
        if df.empty:
            print(f"Error: No data found for device ID: {device_id}")
            return
            
    position_df = df[df['variable'] == 'POSITION'].copy()
    
    # Check if any POSITION data exists for the filtered device(s)
    if position_df.empty:
        print(f"Warning: No 'POSITION' data found for device {mode}. Cannot calculate movement features.")
        # If running for a single device, display placeholder features
        if device_id:
            final_risk_features = pd.DataFrame({
                'deviceId': [device_id], 
                'total_distance_km': [0.0],
                'hard_brake_rate_per_1000km': [0.0],
                'hard_accel_rate_per_1000km': [0.0],
                'percent_time_high_speed': [0.0]
            })
            print(f"\n✅ Features for Device {device_id} Calculated:")
            print(final_risk_features.head(1).to_string(index=False))
        return
    
    # Handle the 'value' column extraction
    parts = position_df['value'].str.split(',', expand=True)
    
    # Use .loc to ensure the columns are created and assigned correctly.
    position_df.loc[:, 'latitude'] = pd.to_numeric(parts.iloc[:, 0], errors='coerce')
    position_df.loc[:, 'longitude'] = pd.to_numeric(parts.iloc[:, 1], errors='coerce')

    position_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    # Re-check for empty data after dropping NaNs
    if position_df.empty:
        print(f"Warning: No valid coordinate data found after cleaning for device {mode}. Cannot calculate movement features.")
        if device_id:
            final_risk_features = pd.DataFrame({
                'deviceId': [device_id], 
                'total_distance_km': [0.0],
                'hard_brake_rate_per_1000km': [0.0],
                'hard_accel_rate_per_1000km': [0.0],
                'percent_time_high_speed': [0.0]
            })
            print(f"\n✅ Features for Device {device_id} Calculated:")
            print(final_risk_features.head(1).to_string(index=False))
        return

    # Prepare time columns
    position_df['datetime'] = pd.to_datetime(position_df['timestamp'])
    position_df['time_sec'] = position_df['datetime'].astype(np.int64) // 10**9
    
    grouped = position_df.groupby('deviceId')

    # --- FIX APPLIED: Create Lagged Columns first, then calculate vectorially ---
    print("-> Calculating distance using grouped lagged coordinates...")
    # 1. Create lagged columns using groupby().shift(1)
    position_df.loc[:, 'latitude_prev'] = grouped['latitude'].shift(1)
    position_df.loc[:, 'longitude_prev'] = grouped['longitude'].shift(1)
    
    # 2. Calculate distance vectorially across the entire dataframe
    position_df.loc[:, 'distance_km'] = haversine_distance(
        position_df['latitude_prev'], position_df['longitude_prev'],
        position_df['latitude'], position_df['longitude']
    )
    # --------------------------------------------------------------------------
    
    position_df['time_diff_sec'] = grouped['time_sec'].diff() 
    position_df['speed_kmh'] = np.where(
        position_df['time_diff_sec'] > 0,
        position_df['distance_km'] / (position_df['time_diff_sec'] / 3600.0),
        0.0 
    )

    # --- Step [2]: Calculate Maneuver Risk (Hard Events) ---
    print("-> Calculating Hard Braking and Hard Acceleration events...")
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

    # --- Step [1] & [2] Aggregation: Exposure and Maneuver Risk Aggregation ---
    print("-> Aggregating Exposure (Distance) and Maneuver Risk (Event Rates) by device...")
    grouped_features = position_df.groupby('deviceId').agg(
        total_distance_km=('distance_km', 'sum'),
        total_hard_brakes=('is_hard_brake', 'sum'),
        total_hard_accels=('is_hard_accel', 'sum'),
        total_driving_time_sec=('time_diff_sec', 'sum')
    ).reset_index()
    safe_distance = grouped_features['total_distance_km'].replace(0, 1e-6)
    grouped_features['hard_brake_rate_per_1000km'] = (grouped_features['total_hard_brakes'] / safe_distance) * 1000
    grouped_features['hard_accel_rate_per_1000km'] = (grouped_features['total_hard_accels'] / safe_distance) * 1000

    # --- Step [3]: Calculate Speed Risk (Time spent above 90 km/h) ---
    print("-> Calculating Speed Risk (percentage of time above 90 km/h)...")
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
    
    # --- Final Merge and Save/Display ---
    final_risk_features = pd.merge(grouped_features, speed_risk_df[['deviceId', 'percent_time_high_speed']], on='deviceId', how='left')
    
    if device_id:
        print(f"\n✅ Features for Device {device_id} Calculated:")
        # Display only the features for the single device
        print(final_risk_features.head(1).to_string(index=False))
        # Do not save to file when running for a single user session
    else:
        # Only save the full file if we processed ALL devices (Batch Mode)
        final_risk_features.to_csv(FEATURES_FILE_NAME, index=False)
        print(f"\nSuccessfully generated {len(final_risk_features)} device features.")
        print(f"Final feature table saved to: {FEATURES_FILE_NAME}")


# --- CORE PREDICTION (Updated to accept device_id) ---

def calculate_insurance_rate(device_id: str):
    """
    Calculates Insurance Rate by training a model on the batch feature file
    and predicting the risk score for the specified single device.
    """
    
    print(f"\n--- [4] Calculate Insurance Rate (Predicting Risk Score) for Device: {device_id} ---")
    
    # Check dependencies (requires features and claims files for model training)
    if not os.path.exists(FEATURES_FILE_NAME) or not os.path.exists(CLAIMS_FILE_NAME):
        print(f"Error: Required feature or claims files not found.")
        print("Please run **[4] Calculate ALL Features and SAVE (Batch Mode)** first to train the model.")
        return

    # 1. LOAD DATA (Full dataset for training, target features for prediction)
    final_risk_features = pd.read_csv(FEATURES_FILE_NAME)
    claims_data = pd.read_csv(CLAIMS_FILE_NAME)

    # Filter feature set down to the target device
    target_features = final_risk_features[final_risk_features['deviceId'].astype(str) == str(device_id)].copy()
    if target_features.empty:
        print(f"Error: No features found in {FEATURES_FILE_NAME} for device ID: {device_id}")
        print("Did you run [4] Calculate ALL Features after adding your device's data?")
        return

    # --- MODEL PREPARATION & TRAINING (Using full dataset for training) ---
    features = ['hard_brake_rate_per_1000km', 'percent_time_high_speed', 'total_distance_km']
    full_model_data = pd.merge(final_risk_features, claims_data[['deviceId', 'has_claim']], on='deviceId', how='left')
    full_y = full_model_data['has_claim'].fillna(0).astype(int)
    
    if full_y.nunique() <= 1:
        print("Warning: Target variable 'has_claim' is constant across all data. Skipping prediction.")
        return

    print("-> Training Logistic Regression Model on full batch data...")
    X_train, _, y_train, _ = train_test_split(full_model_data[features], full_y, test_size=0.01, random_state=42)
    
    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train, y_train)

    # 2. Predict Risk Probability (Score)
    X_predict = target_features[features]
    risk_probabilities = model.predict_proba(X_predict)[:, 1]
    
    target_features['risk_score'] = risk_probabilities
    
    # 3. Apply a Risk Tier based on the predicted probability
    HIGH_RISK_PROBABILITY_THRESHOLD = 0.55
    target_features['final_rate'] = np.where(
        target_features['risk_score'] >= HIGH_RISK_PROBABILITY_THRESHOLD, 
        'High Premium (Predicted Risk)', 
        'Standard Premium'
    )
    
    print("\n-> Prediction Summary ---")
    print(target_features[[
        'deviceId', 'hard_brake_rate_per_1000km', 
        'percent_time_high_speed', 'risk_score', 'final_rate'
    ]].to_string(index=False))
    print("\nPrediction complete. The 'risk_score' is the predicted probability of a claim.")


# --- CLI EXECUTION LOOPS ---

def session_main(user_data):
    """The main CLI loop run after a user has logged in."""
    user_name = user_data['name']
    device_id = user_data['device_id']
    running = True
    
    print(f"\n🔑 Logged in as: {user_name} (Device ID: {device_id})")
    
    while running:
        
        print("\n=======================================")
        print(f" 👤 Session Menu for: {user_name}")
        print("=======================================")
        print("[1] Calculate Features for MY Device (Real-Time)")
        print("[2] Predict Insurance Rate for MY Device (Model Score)")
        print("---------------------------------------")
        print("[4] Calculate ALL Features and SAVE (Batch Mode - Required for [2])")
        print("[9] Display Test Data (Raw POSITION rows for ALL devices)")
        print("[0] Log Out")
        choice = input("Selection: ").strip().lower()

        match choice:
            case "1":
                calculate_all_features(device_id)
            case "2":
                calculate_insurance_rate(device_id)
            case "4":
                # Original batch mode (no device ID) - runs on all data
                calculate_all_features(device_id=None)
            case "9":
                df, device_col = retrieve_data("POSITION")
                display_data(df, device_col)
            case "0":
                print(f"👋 {user_name} logged out.")
                running = False
            case _:
                print("Invalid selection. Please choose a valid option.")

def login_main():
    """The outermost CLI loop for user selection and application setup."""
    running = True
    while running:
        
        print("\n=======================================")
        print(" 💻 Telematics Risk Model: LOGIN/SETUP")
        print("=======================================")
        print("Select User to Log In:")
        print("[1] User One (Alice Smith)")
        print("[2] User Two (Bob Johnson)")
        print("[3] User Three (Charlie Brown)")
        print("---------------------------------------")
        print("[8] Run MongoDB CSV Migration 🚀 (One-Time)")
        print("[9] Load Sample Users 🧑‍💻 (One-Time)")
        print("[0] Quit Application")
        
        choice = input("Selection: ").strip()

        match choice:
            case "1" | "2" | "3":
                user_data = get_user_by_selection(choice)
                if user_data:
                    # Pass control to the session menu
                    session_main(user_data)
            case "8":
                run_migration()
            case "9":
                load_sample_users()
            case "0":
                print("Exiting the application. Goodbye!")
                running = False
            case _:
                print("Invalid selection. Please choose a valid option.")


if __name__ == "__main__":
    login_main()