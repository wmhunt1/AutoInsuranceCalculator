#!/usr/bin/env python3
"""
Simple console app for Telematics Risk Modeling.

This script loads data, calculates risk features (Exposure, Maneuver, Speed), 
and uses a Logistic Regression model (with placeholder claims data) to predict 
a risk score and premium tier. It also contains functionality to migrate 
CSV files to MongoDB, keeping credentials secure via a .env file.
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
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# --- SKLEARN FOR MODELING ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load environment variables (must happen early)
load_dotenv()

# --- MongoDB Connection Details ---
# Reads the URI from the .env file. Falls back to local host if not found.
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "telematics_risk_db"
RAW_DATA_COLLECTION = "raw_telematics"
CLAIMS_COLLECTION = "claims_history"

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


# --- UTILITY FUNCTIONS (Unchanged for brevity) ---

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

# --- CORE FEATURE ENGINEERING ---

def calculate_all_features():
    """
    Performs the full feature engineering pipeline:
    [1] Calculate Exposure (Distance/Time), Maneuver Risk, and Speed Risk.
    Saves the final aggregated features per device to Telematics_Risk_Features_FULL.csv.
    """
    print("\n--- Running Feature Engineering Pipeline ---")
    
    # 1. INITIAL DATA PREPARATION
    df = load_csv("Telematicsdata.csv")
    position_df = df[df['variable'] == 'POSITION'].copy()
    parts = position_df['value'].str.split(',', expand=True)
    position_df['latitude'] = pd.to_numeric(parts[0], errors='coerce')
    position_df['longitude'] = pd.to_numeric(parts[1], errors='coerce')
    position_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    position_df['datetime'] = pd.to_datetime(position_df['timestamp'])
    position_df['time_sec'] = position_df['datetime'].astype(np.int64) // 10**9

    grouped = position_df.groupby('deviceId')

    # Calculate Distance, Time Diff, and Speed
    position_df['distance_km'] = grouped.apply(lambda x: haversine_distance(
        x['latitude'].shift(1), x['longitude'].shift(1),
        x['latitude'], x['longitude']
    )).reset_index(level=0, drop=True)
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
    
    # --- Final Merge and Save ---
    final_risk_features = pd.merge(grouped_features, speed_risk_df[['deviceId', 'percent_time_high_speed']], on='deviceId', how='left')
    
    final_risk_features.to_csv(FEATURES_FILE_NAME, index=False)
    
    print(f"\nSuccessfully generated {len(final_risk_features)} device features.")
    print(f"Final feature table saved to: {FEATURES_FILE_NAME}")
    print("\n--- Summary of Final Features (First 3 Devices) ---")
    print(final_risk_features.head(3).to_string(index=False))


def calculate_insurance_rate():
    """
    [4] Calculate Insurance Rate - Implements Logistic Regression using features 
    and the Claims_History.csv target variable.
    """
    print("\n--- [4] Calculate Insurance Rate (Predicting Risk Score) ---")
    
    # Check if features file exists (must run [1] first)
    if not os.path.exists(FEATURES_FILE_NAME):
        print(f"Error: Risk features file '{FEATURES_FILE_NAME}' not found. Please run [1] Calculate All Features first.")
        return
        
    # Check if claims file exists (must be generated/provided)
    if not os.path.exists(CLAIMS_FILE_NAME):
        print(f"Error: Claims history file '{CLAIMS_FILE_NAME}' not found. Please generate it first.")
        return

    # 1. LOAD DATA
    final_risk_features = pd.read_csv(FEATURES_FILE_NAME)
    claims_data = pd.read_csv(CLAIMS_FILE_NAME)

    # 2. MERGE: Combine Features (X) with Target (Y)
    print("-> Merging features with claims data to get the target variable (Y)...")
    model_data = pd.merge(final_risk_features, claims_data[['deviceId', 'has_claim']], on='deviceId', how='left')
    # Fill NaN claims with 0 (assuming devices with no match had no claim in the period)
    model_data['has_claim'] = model_data['has_claim'].fillna(0).astype(int)

    # --- MODEL PREPARATION ---
    # Define Features (X) and Target (Y)
    features = ['hard_brake_rate_per_1000km', 'percent_time_high_speed', 'total_distance_km']
    
    # Check for features availability
    missing_features = [f for f in features if f not in model_data.columns]
    if missing_features:
        print(f"Error: Missing features in model data: {missing_features}. Cannot run prediction.")
        return

    X = model_data[features]
    y = model_data['has_claim']

    # Handle cases where all target values are the same (prevents model crash)
    if y.nunique() <= 1:
        print("Warning: Target variable 'has_claim' is constant. Cannot train model. Skipping prediction.")
        return

    # Split the data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # --- MODEL TRAINING AND PREDICTION ---
    print(f"-> Training Logistic Regression Model on {len(X_train)} training records...")
    
    # 1. Train the Logistic Regression model
    model = LogisticRegression(random_state=42, solver='liblinear') # using liblinear for stability on small datasets
    model.fit(X_train, y_train)

    # 2. Predict Risk Probability (Score) for ALL data points
    # [:, 1] extracts the probability of the positive class (i.e., probability of a claim = 1)
    risk_probabilities = model.predict_proba(X)[:, 1]
    model_data['risk_score'] = risk_probabilities
    
    # 3. Apply a Risk Tier based on the predicted probability
    HIGH_RISK_PROBABILITY_THRESHOLD = 0.55 # Example cutoff
    model_data['final_rate'] = np.where(
        model_data['risk_score'] >= HIGH_RISK_PROBABILITY_THRESHOLD, 
        'High Premium (Predicted Risk)', 
        'Standard Premium'
    )
    
    print("\n-> Prediction Summary (Highest Risk Devices) ---")
    prediction_summary = model_data[[
        'deviceId', 'has_claim', 'hard_brake_rate_per_1000km', 
        'percent_time_high_speed', 'risk_score', 'final_rate'
    ]].sort_values(by='risk_score', ascending=False)
    
    print(prediction_summary.head(10).to_string(index=False))
    print("\nPrediction complete. The 'risk_score' is the predicted probability of a claim.")

# --- MONGODB MIGRATION LOGIC ---

def migrate_csv_to_mongodb(csv_file_path: str, collection_name: str, client: MongoClient):
    """
    Loads a CSV file and inserts its records into the specified MongoDB collection.
    """
    print(f"\nAttempting to load data from: {csv_file_path}")
    
    # 1. Load CSV data into a Pandas DataFrame
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}. Skipping migration for this file.")
        return

    # 2. Convert the DataFrame to a list of Python dictionaries (JSON-like documents)
    data_records = df.to_dict('records')
    
    # 3. Get the database and collection
    db = client[DATABASE_NAME]
    collection = db[collection_name]
    
    # 4. Insert the data
    try:
        if data_records:
            # Clear collection before inserting to ensure a clean run
            # WARNING: Uncommenting this will delete ALL existing data in the collection
            # collection.delete_many({}) 
            result = collection.insert_many(data_records)
            print(f"✅ Successfully inserted {len(result.inserted_ids)} records into '{collection_name}'.")
        else:
            print(f"Warning: CSV file {csv_file_path} was empty.")
    except Exception as e:
        print(f"❌ An error occurred during MongoDB insert for {collection_name}: {e}")

def run_migration():
    """Establishes MongoDB connection and runs the migration for all CSV files."""
    try:
        # Use a 'with' statement to ensure the connection is closed automatically
        print(f"\n--- Attempting connection to MongoDB at: {MONGO_URI} ---")
        with MongoClient(MONGO_URI) as client:
            # The server_api is often required for modern Atlas clusters
            # client = MongoClient(MONGO_URI, server_api=ServerApi('1')) 
            
            client.admin.command('ping')
            print("✨ Successfully connected to MongoDB.")
            
            # 1. Migrate the large raw data file
            migrate_csv_to_mongodb("Telematicsdata.csv", RAW_DATA_COLLECTION, client)
            
            # 2. Migrate the smaller claims history file
            migrate_csv_to_mongodb(CLAIMS_FILE_NAME, CLAIMS_COLLECTION, client)
            
    except Exception as e:
        print(f"🚨 FAILED to connect to MongoDB. Please check your MONGO_URI, network access, and firewall settings. Error: {e}")


# --- MAIN CLI EXECUTION ---

def main():
    """The main command line interface loop."""
    running = True
    while running:
        
        print("\n=======================================")
        print(" Welcome to the Telematics Risk Model CLI")
        print("=======================================")
        print("[1] Calculate All Features (Exposure, Maneuver, Speed)")
        print("[2] Calculate Insurance Rate (Run Logistic Regression Model)")
        print("[3] MIGRATE CSV DATA TO MONGODB (One-time setup)")
        print("[9] Display Test Data (Raw POSITION rows)")
        print("[0] Quit")
        choice = input("Selection: ").strip().lower()

        match choice:
            case "1":
                calculate_all_features()
            case "2":
                calculate_insurance_rate()
            case "3":
                run_migration()
            case "9":
                df, device_col = retrieve_data("POSITION")
                display_data(df, device_col)
            case "0":
                print("Exiting the application. Goodbye!")
                running = False
            case _:
                print("Invalid selection. Please choose a valid option.")


if __name__ == "__main__":
    main()