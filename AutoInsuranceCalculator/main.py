#!/usr/bin/env python3
"""
Simple console app to read Telematicsdata.csv and display its contents.

Usage:
  python main.py
  python main.py --file Telematicsdata.csv --head 50
  python main.py --device 12345 --columns speed,accel
"""

from typing import List, Optional
import os
import sys
import argparse
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# --- UTILITY FUNCTIONS ---

def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def parse_args():
    p = argparse.ArgumentParser(description="Read Telematicsdata.csv and display rows.")
    p.add_argument("--file", "-f", default="Telematicsdata.csv", help="Path to CSV file")
    p.add_argument("--head", "-n", type=int, default=20, help="Number of rows to show (use 0 for all)")
    p.add_argument("--device", "-d", help="Device id to filter (string match)")
    p.add_argument("--columns", "-c", help="Comma-separated columns to show (or 'all')")
    return p.parse_args()

def load_csv(path: str) -> pd.DataFrame:
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
    return find_column(df.columns.tolist(), DEVICE_COL_CANDIDATES)

def filter_by_device(df: pd.DataFrame, device_col: str, device_id: str) -> pd.DataFrame:
    # Compare as string to be robust to numeric/string mismatch
    return df[df[device_col].astype(str) == str(device_id)].copy()

def filter_by_variable(df: pd.DataFrame, variable_col: str, variable_name: str) -> pd.DataFrame:
    """Filters the DataFrame to include only rows where the variable column matches a specific name."""
    filtered_df = df[df[variable_col].astype(str).str.upper() == variable_name.upper()].copy()
    return filtered_df

def select_columns(df: pd.DataFrame, cols_arg: Optional[str]) -> List[str]:
    if not cols_arg:
        return df.columns.tolist()
    if cols_arg.strip().lower() == "all":
        return df.columns.tolist()
    chosen = [c.strip() for c in cols_arg.split(",") if c.strip()]
    missing = [c for c in chosen if c not in df.columns]
    if missing:
        print(f"Columns not found in CSV: {missing}", file=sys.stderr)
        sys.exit(4)
    return chosen

def extract_lat_lon_and_remove_value(df: pd.DataFrame, value_col: str) -> None:
    """
    Extract the first two tokens from the value column into 'latitude' and 'longitude'.
    Modifies df in place.
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
    """Loads, filters, and prepares the DataFrame for display."""
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
    """Displays the processed DataFrame, potentially filtered and sampled."""
    args = parse_args()

    if device_col and not args.device:
        unique = df[device_col].dropna().unique()
        sample = list(map(str, unique[:10]))
        print(f"Sample device ids ({min(len(unique),10)} shown): {sample}")
        print("Pass --device <id> to filter the table.\n")
    elif not device_col:
        print("No device column automatically detected (candidates tried):", DEVICE_COL_CANDIDATES)

    cols = select_columns(df, args.columns)
    to_show = df[cols]

    if args.head == 0:
        try:
            print(to_show.to_string(index=False))
        except Exception:
            print(to_show)
    else:
        print(to_show.head(args.head).to_string(index=False))

def haversine_distance(lat1, lon1, lat2, lon2, R=6371):
    """
    Calculate the great-circle distance (km) between two points on the Earth.
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

# --- NEW ANALYSIS FUNCTIONS ---

def calculate_all_features():
    """
    Calculates Exposure, Maneuver Risk, and Speed Risk and saves the final features.
    This function replaces the placeholder steps [1], [2], and [3].
    """
    print("\n--- Running Feature Engineering Pipeline (Steps [1], [2], [3]) ---")
    
    df = pd.read_csv("Telematicsdata.csv")
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

    # --- Step [1] & [2] Aggregation: Calculate Exposure and Event Counts/Rates ---
    print("-> Aggregating Exposure and Maneuver Risk by device...")
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
    print("-> Calculating Speed Risk (time above 90 km/h)...")
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
    
    # Merge time spent at high speed with total driving time
    total_time_moving = clean_moving_df.groupby('deviceId')['time_diff_sec'].sum().reset_index()
    total_time_moving.columns = ['deviceId', 'total_time_moving_sec']

    speed_risk_df = pd.merge(total_time_moving, high_speed_time, on='deviceId', how='left').fillna(0)
    speed_risk_df['percent_time_high_speed'] = (speed_risk_df['time_high_speed_sec'] / speed_risk_df['total_time_moving_sec'].replace(0, 1e-6)) * 100
    
    # --- Final Merge and Save ---
    final_risk_features = pd.merge(grouped_features, speed_risk_df[['deviceId', 'percent_time_high_speed']], on='deviceId', how='left')
    
    file_name = "Telematics_Risk_Features_FULL.csv"
    final_risk_features.to_csv(file_name, index=False)
    
    print(f"\nSuccessfully generated {len(final_risk_features)} device features.")
    print(f"Final feature table saved to: {file_name}")
    print("\n--- Summary of Final Features (First 3 Devices) ---")
    print(final_risk_features.head(3).to_string(index=False))


def calculate_insurance_rate():
    """
    [4] Calculate Insurance Rate - Placeholder for final prediction step.
    This function would typically load the features generated by calculate_all_features(),
    load a pre-trained risk model, and predict the final insurance premium or rate for each device.
    """
    print("\n--- [4] Calculate Insurance Rate ---")
    
    # Placeholder: Load the features (assuming they were just generated)
    try:
        final_risk_features = pd.read_csv("Telematics_Risk_Features_FULL.csv")
    except FileNotFoundError:
        print("Error: Risk features not yet calculated. Please run [1] first.")
        return

    print(f"Loaded {len(final_risk_features)} device features for prediction.")
    
    # --- PLACEHOLDER FOR MODEL PREDICTION ---
    # Example: A simple score based on the highest risk feature (Hard Acceleration Rate)
    # The higher the rate, the higher the imaginary risk score.
    
    final_risk_features['risk_score'] = final_risk_features['hard_accel_rate_per_1000km'] * 100
    final_risk_features['final_rate'] = np.where(
        final_risk_features['risk_score'] > 50, 
        'High Premium', 
        'Standard Premium'
    )
    # --- END PLACEHOLDER ---

    print("\n--- Example Insurance Risk Prediction ---")
    prediction_summary = final_risk_features[['deviceId', 'hard_accel_rate_per_1000km', 'risk_score', 'final_rate']].sort_values(by='risk_score', ascending=False)
    print(prediction_summary.to_string(index=False))
    print("\nPrediction complete. This step requires a trained machine learning model in a real application.")


def main():
    running = True
    while running:
        
        print("\n=======================================")
        print(" Welcome to the Telematics Risk Model CLI")
        print("=======================================")
        print("[1] Calculate All Features (Exposure, Maneuver, Speed)")
        print("[2] Calculate Insurance Rate (Prediction Placeholder)")
        #print("[9] Display Test Data (Raw POSITION rows)")
        print("[0] Quit")
        choice = input("Selection: ").strip().lower()

        match choice:
            case "1":
                calculate_all_features()
            case "2":
                calculate_insurance_rate()
            # case "9":
            #     # Original test data display function
            #     df, device_col = retrieve_data("POSITION")
            #     display_data(df, device_col)
            case "0":
                print("Exiting the application. Goodbye!")
                running = False
            case _:
                print("Invalid selection. Please choose a valid option.")


if __name__ == "__main__":
    main()