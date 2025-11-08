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

DEVICE_COL_CANDIDATES = [
    "device_id", "deviceid", "deviceId", "DeviceId", "device", "id", "ID"
]
TIME_COL_CANDIDATES = [
    "timestamp", "time", "datetime", "date", "ts"
]
VALUE_COL_CANDIDATES = [
    "value", "values", "Value", "Values"
]


def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def parse_args():
    p = argparse.ArgumentParser(description="Read Telematicsdata.csv and display rows.")
    p.add_argument("--file", "-f", default="Telematicsdata.csv", help="Path to CSV file")
    # This line ensures the display limit is 20 by default.
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


def show_overview(df: pd.DataFrame):
    print(f"Loaded CSV: {len(df)} rows x {len(df.columns)} columns\n")
    print("Columns and dtypes:")
    for c, dt in df.dtypes.items():
        print(f"  {c}: {dt}")
    print()


def filter_by_device(df: pd.DataFrame, device_col: str, device_id: str) -> pd.DataFrame:
    # Compare as string to be robust to numeric/string mismatch
    return df[df[device_col].astype(str) == str(device_id)].copy()

def filter_by_variable(df: pd.DataFrame, variable_col: str, variable_name: str) -> pd.DataFrame:
    """Filters the DataFrame to include only rows where the variable column matches a specific name."""
    #print(f"Filtering data to only include rows where {variable_col} is '{variable_name}'...")
    
    # Use boolean indexing to keep only the rows that match the desired variable name
    filtered_df = df[df[variable_col].astype(str).str.upper() == variable_name.upper()].copy()
    
    #print(f"Filtered rows: {len(filtered_df)} remaining out of {len(df)} initial rows.")
    
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
    Tokens are split on commas, semicolons or whitespace. Non-convertible entries become NaN.
    Modifies df in place by adding 'latitude' and 'longitude' columns and removing the original value column.
    """
    # split into at most 3 parts, take first two tokens
    parts = df[value_col].astype(str).str.split(r'[,;\s]+', n=2, expand=True)
    # parts[0] -> first token (now latitude), parts[1] -> second token (now longitude)
    df["latitude"] = pd.to_numeric(parts.iloc[:, 0], errors="coerce")
    if parts.shape[1] > 1:
        df["longitude"] = pd.to_numeric(parts.iloc[:, 1], errors="coerce")
    else:
        df["longitude"] = pd.NA
    # remove original values column so it won't be displayed
    try:
        df.drop(columns=[value_col], inplace=True)
    except Exception:
        # ignore if drop fails for any reason
        pass

def retrieve_data(value: str) -> tuple[pd.DataFrame, str | None]:
    """
    Loads, filters, and prepares the DataFrame for display.

    Args:
        value: The value to filter by in the VARIABLE_COL.

    Returns:
        A tuple containing the processed DataFrame and the detected device column name (or None).
    """
    args = parse_args()
    df = load_csv(args.file)

    VARIABLE_COL = "variable"
    POSITION_VAR = value

    if VARIABLE_COL in df.columns:
        df = filter_by_variable(df, VARIABLE_COL, POSITION_VAR)
    # else:
    #     print(f"Warning: Cannot filter by variable. Column '{VARIABLE_COL}' not found.")

    # attempt to find a 'value' column and extract latitude/longitude
    value_col = find_column(df.columns.tolist(), VALUE_COL_CANDIDATES)
    if value_col:
        try:
            extract_lat_lon_and_remove_value(df, value_col)
            #print(f"Extracted latitude/longitude from column '{value_col}' into 'latitude' and 'longitude' and removed '{value_col}' from display.\n")
        except Exception as ex:
            print(f"Warning: failed to extract lat/lon from '{value_col}': {ex}", file=sys.stderr)

    device_col = pick_device_column(df)

    if args.device:
        if not device_col:
            #print("Cannot filter by device: no device column detected.", file=sys.stderr)
            sys.exit(5)
        df = filter_by_device(df, device_col, args.device)
        if df.empty:
            #print(f"No rows found for device '{args.device}'.", file=sys.stderr)
            sys.exit(6)
        #print(f"Filtered to device '{args.device}': {len(df)} rows\n")

    return df, device_col

def display_data(df: pd.DataFrame, device_col: str | None) -> None:
    """
    Displays the processed DataFrame, potentially filtered and sampled.

    Args:
        df: The DataFrame to display.
        device_col: The name of the detected device column (or None).
    """
    args = parse_args() # Need to call parse_args here to get display-related arguments

    if device_col:
        #print(f"Detected device column: {device_col}")

        if not args.device:
            # show sample device values if device column exists and no specific device was requested
            unique = df[device_col].dropna().unique()
            sample = list(map(str, unique[:10]))
            #print(f"Sample device ids ({min(len(unique),10)} shown): {sample}")
            #print("Pass --device <id> to filter the table.\n")
    else:
        print("No device column automatically detected (candidates tried):", DEVICE_COL_CANDIDATES)

    cols = select_columns(df, args.columns)
    to_show = df[cols]

    # This conditional logic ensures the display limit is applied.
    # Since args.head defaults to 20 (from parse_args), the output will be limited to 20 rows
    # unless the user explicitly passes --head 0 or another number.
    if args.head == 0:
        # show entire selection
        try:
            print(to_show.to_string(index=False))
        except Exception:
            # fallback to pandas default print
            print(to_show)
    else:
        # This will show the first 'args.head' rows (defaulting to 20)
        print(to_show.head(args.head).to_string(index=False))

def haversine_distance(lat1, lon1, lat2, lon2, R=6371):
    """
    Calculate the great-circle distance (km) between two points on the Earth
    (specified in decimal degrees).
    R is the Earth's radius in kilometers.
    """
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def speed_from_distance_time():
    df = pd.read_csv("Telematicsdata.csv")
    # 1. Filter for POSITION and extract coordinates
    position_df = df[df['variable'] == 'POSITION'].copy()
    parts = position_df['value'].str.split(',', expand=True)
    position_df['latitude'] = pd.to_numeric(parts[0], errors='coerce')
    position_df['longitude'] = pd.to_numeric(parts[1], errors='coerce')

    # Drop rows where coordinates could not be extracted (should be few/none after filtering)
    position_df.dropna(subset=['latitude', 'longitude'], inplace=True)

    # 2. Prepare Time Data
    # Convert timestamp string to datetime objects
    position_df['datetime'] = pd.to_datetime(position_df['timestamp'])
    # Convert datetime to total seconds (or total milliseconds) for easy subtraction
    position_df['time_sec'] = position_df['datetime'].astype(np.int64) // 10**9

    # 3. Calculate Features Grouped by Device
    # Group by deviceId to ensure calculations only happen within one vehicle's trip sequence
    grouped = position_df.groupby('deviceId')

    # Calculate distance (km)
    position_df['distance_km'] = grouped.apply(lambda x: haversine_distance(
        x['latitude'].shift(1), x['longitude'].shift(1),
        x['latitude'], x['longitude']
    )).reset_index(level=0, drop=True)

    # Calculate time difference (hours)
    position_df['time_diff_hrs'] = grouped['time_sec'].diff() / 3600 # 3600 seconds in an hour

    # Calculate speed (km/h)
    # Use np.where to handle division by zero (where distance is non-zero and time_diff > 0)
    # Use 0 for speed where distance is 0 (stationary) or time_diff is 0, and NaN for impossible calculations
    position_df['speed_kmh'] = np.where(
        position_df['time_diff_hrs'] > 0,
        position_df['distance_km'] / position_df['time_diff_hrs'],
        0.0 # If time diff is zero or less, speed is considered 0 or invalid (NaN in diff will remain NaN)
    )

    # 4. Display the results
    # Select key columns for review
    analysis_df = position_df[['deviceId', 'timestamp', 'latitude', 'longitude', 'distance_km', 'time_diff_hrs', 'speed_kmh']]

    #stationary_df = position_df[position_df['speed_kmh'] == 0].copy()

    moving_df = analysis_df[position_df['speed_kmh'] > 0].copy().head(100)

    #print("--- Records with Calculated Distance and Speed ---")
    #print(moving_df.to_string(index=False))

    # 1. Filter the data to include only realistic moving records
    # Moving: speed_kmh > 0
    # Realistic: speed_kmh <= 200 (km/h cap to exclude GPS errors)
    CLEAN_SPEED_CAP = 200.0 
    clean_moving_df = position_df[
        (position_df['speed_kmh'] > 0) & 
        (position_df['speed_kmh'] <= CLEAN_SPEED_CAP)
    ].copy()

    # 2. Calculate the average speed from the cleaned records
    cleaned_average_speed_kmh = clean_moving_df['speed_kmh'].mean()
    total_moving_records_uncapped = len(position_df[position_df['speed_kmh'] > 0])
    total_clean_moving_records = len(clean_moving_df)
    outliers_removed = total_moving_records_uncapped - total_clean_moving_records

    print(f"Total moving records before cleaning: {total_moving_records_uncapped}")
    print(f"Outliers (speed > {CLEAN_SPEED_CAP} km/h) removed: {outliers_removed}")
    print(f"Total clean moving records used: {total_clean_moving_records}")
    print(f"Average speed of cleaned moving records: {cleaned_average_speed_kmh:.2f} km/h")

def main():
    running = True
    while running:
        
        print("Welcome to the Telematics Data Viewer!")
        print("---------------------------------------")
        print("This application allows you to read and display data from a Telematicsdata.csv file.")
        print("[1] Display Data")
        print("[2] Calculate Average Speed")
        print("[0] Exit")
        choice = input("Selection: ").strip().lower()

        match choice:
            case "1":
                df, device_col = retrieve_data("POSITION")
                display_data(df, device_col)
            case "2":
                speed_from_distance_time()
            case "0":
                print("Exiting the application. Goodbye!")
                running = False
            case _:
                print("Invalid selection. Please choose a valid option.\n")


if __name__ == "__main__":
    main()