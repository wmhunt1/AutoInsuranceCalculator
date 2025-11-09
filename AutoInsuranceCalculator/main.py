#!/usr/bin/env python3
"""
Simple console app for Telematics Risk Modeling.

This script implements a two-tiered CLI: a Login menu to select a user and 
a Session menu to run feature engineering and premium estimation for that user's device.
"""

from typing import List, Optional, Any
import os
import sys
import argparse
import re
import time
import hashlib 
import random 
import string 
import getpass # <-- ADDED: For secure password input

# --- THIRD-PARTY LIBRARIES ---
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

# --- SKLEARN FOR MODELING ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load environment variables from the .env file
load_dotenv()

# --- MongoDB Connection Details ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "telematics_risk_db"
RAW_DATA_COLLECTION = "raw_telematics"
CLAIMS_COLLECTION = "claims_history"
USERS_COLLECTION = "users" 

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


# --- AUTHENTICATION FUNCTIONS ---

def hash_password(password: str, salt: str = "secure_salt") -> str:
    """
    Simulates secure password hashing using SHA256 (for simplicity).
    In a real app, use bcrypt or Argon2.
    """
    # Using SHA256 for a secure one-way hash simulation
    pw_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    hashed = hashlib.sha256(pw_bytes + salt_bytes).hexdigest()
    return hashed

def verify_password(stored_hash: str, provided_password: str, salt: str = "secure_salt") -> bool:
    """
    Verifies a provided password against a stored hash.
    """
    return stored_hash == hash_password(provided_password, salt)

# --- USER CREATION/MANAGEMENT ---

def generate_random_device_id(length=32):
    """Generates a random alphanumeric string to simulate a new device ID."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

def create_new_user():
    """
    Prompts for new user details, generates a device ID, clones 
    the data from a random existing device, and assigns random claims history.
    """
    print("\n--- 📝 New User Registration ---")
    
    # Get user input
    new_username = input("Enter new Username: ").strip().lower()
    new_name = input("Enter full Name: ").strip()
    # Using getpass for the new user's password input
    new_password = getpass.getpass("Enter Password: ").strip() 

    # 1. Generate new device ID
    new_device_id = generate_random_device_id()
    
    # 2. Select a random existing device ID to clone data from
    try:
        df_raw = load_csv("Telematicsdata.csv")
        device_col = pick_device_column(df_raw)
        if not device_col:
            raise ValueError("Could not find device ID column in raw data.")
            
        unique_device_ids = df_raw[device_col].astype(str).unique()
        if len(unique_device_ids) == 0:
            raise ValueError("No unique device IDs found in raw data to clone.")
            
        source_device_id = random.choice(unique_device_ids)
        
    except Exception as e:
        print(f"❌ Error during data preparation for new user: {e}")
        return

    # 3. Generate random claim status (0 or 1)
    random_has_claim = random.choice([0, 1])
    
    # 4. Hash password and prepare user data
    hashed_pw = hash_password(new_password)
    new_user_data = {
        "user_id": f"user_new_{time.time()}",
        "username": new_username,
        "name": new_name,
        "device_id": new_device_id,
        "hashed_password": hashed_pw,
        "source_device_id": source_device_id, # The device whose driving data this user will use
        "has_claim": random_has_claim        # The simulated claims history for this user
    }
    
    # 5. Save to MongoDB
    try:
        with get_mongo_client() as client:
            db = client[DATABASE_NAME]
            collection = db[USERS_COLLECTION]
            
            # Check for existing username
            if collection.find_one({"username": new_username}):
                print(f"❌ Registration failed: Username '{new_username}' already exists.")
                return

            result = collection.insert_one(new_user_data)
            
            print(f"\n✅ User '{new_name}' created successfully.")
            print(f"   - Assigned New Device ID: {new_device_id}")
            print(f"   - **Fetching driving record...** (using data from {source_device_id})")
            print(f"   - **Generating claims history:** {'YES (1)' if random_has_claim == 1 else 'NO (0)'}")
            print("---------------------------------------")
            print(f"You can now log in as: '{new_username}'")
            
    except Exception as e:
        print(f"❌ Failed to create user. Error: {e}")


def get_mongo_client():
    """Returns a connected MongoClient using the URI."""
    return MongoClient(MONGO_URI)

def load_sample_users():
    """Defines and loads sample user data into the MongoDB users collection, including hashed passwords and claim status."""
    print("\n--- Loading Sample User Data ---")
    
    # Load claims data to find out the claims status of the sample users' devices
    claims_map = {}
    try:
        claims_df = load_csv(CLAIMS_FILE_NAME)[['deviceId', 'has_claim']]
        claims_map = claims_df.set_index('deviceId')['has_claim'].to_dict()
    except Exception:
        print("Warning: Claims file not available. Cannot set initial claim status for sample users.")

    # Store plain passwords in a dict for hashing before saving
    user_passwords = {
        "alice_smith": "Password123",
        "bob_johnson": "SecurePass456",
        "charlie_brown": "TestUser789",
    }
    
    # NOTE: These device_ids are derived from your Telematicsdata.csv
    sample_users_data = [
        {"user_id": "user_one_id", "username": "alice_smith", "name": "Alice Smith", "device_id": "zRYzhAEAHAABAAAKCRtcAAsAtwB1gBAQ"},
        {"user_id": "user_two_id", "username": "bob_johnson", "name": "Bob Johnson", "device_id": "zRYzhAEAHAABAAAKCRtcAAsANAB0gBAQ"},
        {"user_id": "user_three_id", "username": "charlie_brown", "name": "Charlie Brown", "device_id": "zRYzhAEAHAABAAAKCRtcAAsAGAB0gBAQ"},
    ]

    # HASH THE PASSWORDS and add them to the data, along with their actual claim status
    for user in sample_users_data:
        plain_pw = user_passwords[user['username']]
        user['hashed_password'] = hash_password(plain_pw)
        user['source_device_id'] = user['device_id'] # Old users use their own device ID as source
        user['has_claim'] = claims_map.get(user['device_id'], 0) # Fetch claim status, default to 0
        
    try:
        with get_mongo_client() as client:
            db = client[DATABASE_NAME]
            collection = db[USERS_COLLECTION]
            
            # Clear existing data and insert new data
            collection.delete_many({})
            result = collection.insert_many(sample_users_data)
            
            print(f"✅ Successfully loaded {len(result.inserted_ids)} sample users into '{USERS_COLLECTION}' with hashed passwords.")
            print("--- Sample Users Loaded (Credentials for testing) ---")
            for user in sample_users_data:
                print(f"  - User: {user['username']} | Pass: {user_passwords[user['username']]} | Claim: {user['has_claim']}")
                
    except Exception as e:
        print(f"❌ Failed to load sample users. Error: {e}")

def get_user_by_credentials(username: str, password: str):
    """Fetches a user document from MongoDB and verifies the password."""
    try:
        with get_mongo_client() as client:
            db = client[DATABASE_NAME]
            collection = db[USERS_COLLECTION]
            
            # 1. Find user by username
            user_data = collection.find_one({"username": username})
            
            if user_data:
                # 2. Verify password against stored hash
                if verify_password(user_data['hashed_password'], password):
                    return user_data
                else:
                    print("Login failed: Invalid password.")
                    return None
            else:
                print(f"Login failed: User '{username}' not found.")
                return None
                
    except Exception as e:
        print(f"❌ Error fetching user data from MongoDB: {e}")
        return None

# --- UTILITY FUNCTIONS (UNCHANGED) ---

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


# --- MONGODB MIGRATION LOGIC (UNCHANGED) ---

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

# --- CORE FEATURE ENGINEERING (UNCHANGED) ---

def calculate_all_features(device_id: Optional[str], source_device_id: Optional[str] = None):
    """
    Performs the full feature engineering pipeline.
    If source_device_id is provided, it uses the data for that device ID
    but assigns the features to the main device_id (for new users).
    """
    
    # Determine which ID to use for filtering (source_device_id if provided, otherwise device_id)
    filter_id = source_device_id if source_device_id else device_id
    mode = filter_id if filter_id else 'ALL'
    
    print(f"\n--- Running Feature Engineering Pipeline using Data from Device: {mode} ---")
    
    # 1. INITIAL DATA PREPARATION
    df = load_csv("Telematicsdata.csv")
    
    # Filter by the actual data source ID
    if filter_id:
        df = df[df['deviceId'].astype(str) == str(filter_id)].copy()
        if df.empty:
            print(f"Error: No data found for device ID: {filter_id}")
            return
            
    position_df = df[df['variable'] == 'POSITION'].copy()
    
    if position_df.empty:
        print(f"Warning: No 'POSITION' data found for device {mode}. Cannot calculate movement features.")
        # Simplified empty feature generation for new users
        if device_id:
            final_risk_features = pd.DataFrame({
                'deviceId': [device_id], 
                'total_distance_km': [0.0],
                'hard_brake_rate_per_1000km': [0.0],
                'hard_accel_rate_per_1000km': [0.0],
                'percent_time_high_speed': [0.0]
            })
            return final_risk_features
        return
    
    # Handle the 'value' column extraction (coordinate parsing)
    parts = position_df['value'].str.split(',', expand=True)
    position_df.loc[:, 'latitude'] = pd.to_numeric(parts.iloc[:, 0], errors='coerce')
    position_df.loc[:, 'longitude'] = pd.to_numeric(parts.iloc[:, 1], errors='coerce')

    position_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    
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
            return final_risk_features
        return

    # Prepare time columns
    position_df['datetime'] = pd.to_datetime(position_df['timestamp'])
    position_df['time_sec'] = position_df['datetime'].astype(np.int64) // 10**9
    
    # Crucially, we group by the original deviceId in the data being processed
    grouped = position_df.groupby('deviceId')

    # ... (Rest of feature calculation logic) ...
    print("-> Calculating distance using grouped lagged coordinates...")
    position_df.loc[:, 'latitude_prev'] = grouped['latitude'].shift(1)
    position_df.loc[:, 'longitude_prev'] = grouped['longitude'].shift(1)
    
    position_df.loc[:, 'distance_km'] = haversine_distance(
        position_df['latitude_prev'], position_df['longitude_prev'],
        position_df['latitude'], position_df['longitude']
    )
    
    position_df['time_diff_sec'] = grouped['time_sec'].diff() 
    position_df['speed_kmh'] = np.where(
        position_df['time_diff_sec'] > 0,
        position_df['distance_km'] / (position_df['time_diff_sec'] / 3600.0),
        0.0 
    )

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
    
    # --- Final Merge and Output Assignment ---
    final_risk_features = pd.merge(grouped_features, speed_risk_df[['deviceId', 'percent_time_high_speed']], on='deviceId', how='left')

    if filter_id and device_id and (filter_id != device_id):
        # If running for a new user (using source data), replace the deviceId in the output
        final_risk_features.loc[:, 'deviceId'] = device_id
        print(f"\n✅ Features Calculated and assigned to New Device ID: {device_id}")
    elif filter_id:
        print(f"\n✅ Features for Device {filter_id} Calculated.")
    else:
        # Batch Mode - runs on all data
        final_risk_features.to_csv(FEATURES_FILE_NAME, index=False)
        print(f"\nSuccessfully generated {len(final_risk_features)} device features.")
        print(f"Final feature table saved to: {FEATURES_FILE_NAME}")

    return final_risk_features

# --- SIMULATION FUNCTION (External API - UNCHANGED) ---

def get_simulated_premium(features: pd.Series) -> tuple[float, str]:
    """
    Simulates calling an external insurance API to get a premium estimate.
    """
    # 1. Simulate proprietary risk mapping (Example Logic)
    hard_brake_score = np.interp(
        features['hard_brake_rate_per_1000km'],
        [0, 5, 10, 20],  # Input range (your feature value)
        [1, 3, 6, 10]    # Output range (the API's required risk score 1-10)
    )
    
    high_speed_score = np.interp(
        features['percent_time_high_speed'],
        [0, 2, 5, 10],   # Input range (your feature value: percent)
        [1, 3, 6, 10]    # Output range (the API's required risk score 1-10)
    )

    avg_risk_score = (hard_brake_score + high_speed_score) / 2
    
    # 2. Simulate Premium Calculation (Example Logic)
    base_premium = 1100.0 # Slightly adjusted base for difference from internal model
    adjustment_factor = (avg_risk_score / 10.0) * 0.7 
    premium = base_premium * (1 + adjustment_factor)
    
    # 3. Simulate API Call Details and Response
    print("\n--- 🌐 Simulating External Insurance API Call ---")
    print(f"   -> Mapping internal features for Device {features['deviceId']}...")
    print(f"      - Hard Brake Score Sent to API: {hard_brake_score:.1f}/10")
    print(f"      - High Speed Score Sent to API: {high_speed_score:.1f}/10")
    print(f"   -> Sending Simulated POST Request to /v1/quote...")
    
    time.sleep(0.5) # Simulate latency
    
    # 4. Determine Quote Message
    if premium > 1550:
        quote_message = "External Vendor - High Risk Quote."
    elif premium > 1250:
        quote_message = "External Vendor - Standard Quote."
    else:
        quote_message = "External Vendor - Best Rate."
        
    print("   <- Received Simulated 200 OK Response.")

    return premium, quote_message

# --- CORE PREMIUM ESTIMATION (UNCHANGED) ---

def get_and_display_all_estimates(user_data):
    """
    Calculates the internal risk score/premium and calls the simulated external API.
    """
    device_id = user_data['device_id']
    source_device_id = user_data.get('source_device_id', device_id) # Use the source ID if it exists

    print(f"\n--- [2] Generating Internal and External Premium Estimates for Device: {device_id} ---")
    
    # 1. Check dependencies
    if not os.path.exists(FEATURES_FILE_NAME) or not os.path.exists(CLAIMS_FILE_NAME):
        print(f"Error: Required feature or claims files not found.")
        print("Please run **[4] Calculate ALL Features and SAVE (Batch Mode)** first to ensure the training data is ready.")
        return

    # 2. Load and Prepare Data for Internal Model
    final_risk_features = pd.read_csv(FEATURES_FILE_NAME)
    claims_data = pd.read_csv(CLAIMS_FILE_NAME)

    # Note: Target features must be filtered by the logged-in user's source device_id (if a new user)
    target_features_df = final_risk_features[final_risk_features['deviceId'].astype(str) == str(source_device_id)].copy()
    
    if target_features_df.empty:
        # If the batch file doesn't contain the data, calculate it in real-time
        print(f"Warning: Batch feature data missing for source ID {source_device_id}. Calculating real-time features...")
        target_features_df = calculate_all_features(device_id=device_id, source_device_id=source_device_id)
        if target_features_df is None or target_features_df.empty:
            print("Fatal Error: Could not generate real-time features. Cannot provide an estimate.")
            return

    target_features = target_features_df.iloc[0]
    
    # Temporarily correct the deviceId for display if it's a new user
    target_features.loc['deviceId'] = device_id 

    features_cols = ['hard_brake_rate_per_1000km', 'percent_time_high_speed', 'total_distance_km']
    full_model_data = pd.merge(final_risk_features, claims_data[['deviceId', 'has_claim']], on='deviceId', how='left')
    full_y = full_model_data['has_claim'].fillna(0).astype(int)
    
    if full_y.nunique() <= 1:
        print("Warning: Target variable 'has_claim' is constant across all data. Internal prediction skipped.")
        internal_risk_score = 0.5 # Neutral score fallback
        internal_premium = 1000.0 # Neutral premium fallback
    else:
        # 3. Train and Predict Internal Risk Score (Logistic Regression)
        print("-> Training Logistic Regression Model on full batch data for Internal Estimate...")
        X_train, _, y_train, _ = train_test_split(full_model_data[features_cols], full_y, test_size=0.01, random_state=42)
        
        model = LogisticRegression(random_state=42, solver='liblinear')
        model.fit(X_train, y_train)

        # Predict Risk Probability (Score) - use the data row from the source ID
        X_predict = target_features_df[features_cols] 
        internal_risk_score = model.predict_proba(X_predict)[:, 1][0]
        
        # 4. Calculate Internal Premium
        BASE_PREMIUM = 1000.0
        RISK_MULTIPLIER = 1.0 
        internal_premium = BASE_PREMIUM * (1 + internal_risk_score * RISK_MULTIPLIER)

    # 5. CALL THE SIMULATED EXTERNAL API 
    external_premium, external_message = get_simulated_premium(target_features)
    
    # 6. Display Results
    user_has_claim = user_data.get('has_claim', 'N/A')
    
    print("\n=============================================")
    print("      *** FINAL PREMIUM ESTIMATES ***")
    print("=============================================")
    print(f"Driver Name: {user_data['name']}")
    print(f"Device ID: {device_id}")
    print(f"Actual Claim History: {'YES (1)' if user_has_claim == 1 else 'NO (0)'} 🧐")
    print(f"Driving Exposure (Distance): {target_features['total_distance_km']:.0f} km")
    print("---------------------------------------------")

    # Internal Estimate
    print("1. INTERNAL MODEL ESTIMATE")
    print(f"   - Predicted Claim Risk Score: {internal_risk_score:.4f}")
    print(f"   - Estimated Annual Premium: ${internal_premium:,.2f}")
    print("---------------------------------------------")

    # External Estimate
    print("2. EXTERNAL VENDOR ESTIMATE (Simulated API)")
    print(f"   - Estimated Annual Premium: ${external_premium:,.2f}")
    print(f"   - Vendor Quote Message: {external_message}")
    print("=============================================")


# --- CLI EXECUTION LOOPS (UPDATED) ---

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
        print("[2] **Generate ALL Premium Estimates (Internal & External)** 💰")
        print("---------------------------------------")
        print("[4] Calculate ALL Features and SAVE (Batch Mode - Required for [2])")
        print("[9] Display Test Data (Raw POSITION rows for ALL devices)")
        print("[0] Log Out")
        choice = input("Selection: ").strip().lower()

        match choice:
            case "1":
                calculate_all_features(device_id, user_data.get('source_device_id', device_id))
            case "2":
                get_and_display_all_estimates(user_data)
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
        print("Select Action:")
        print("[L] Login to Account")
        print("[C] Create New User 🧑‍🔬")
        print("---------------------------------------")
        print("[8] Run MongoDB CSV Migration 🚀 (One-Time)")
        print("[9] Load Sample Users 🧑‍💻 (One-Time)")
        print("[0] Quit Application")
        
        choice = input("Selection: ").strip().lower()

        match choice:
            case "l":
                username = input("Enter Username: ").strip().lower()
                # Use getpass.getpass() to securely read the password
                password = getpass.getpass("Enter Password: ").strip() 
                user_data = get_user_by_credentials(username, password)
                if user_data:
                    # Pass control to the session menu
                    session_main(user_data)
            case "c":
                create_new_user()
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