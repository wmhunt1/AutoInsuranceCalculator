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
    print(f"Filtering data to only include rows where {variable_col} is '{variable_name}'...")
    
    # Use boolean indexing to keep only the rows that match the desired variable name
    filtered_df = df[df[variable_col].astype(str).str.upper() == variable_name.upper()].copy()
    
    print(f"Filtered rows: {len(filtered_df)} remaining out of {len(df)} initial rows.")
    
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


def main():
    args = parse_args()
    df = load_csv(args.file)

    VARIABLE_COL = "variable" 
    POSITION_VAR = "POSITION"

    if VARIABLE_COL in df.columns:
        df = filter_by_variable(df, VARIABLE_COL, POSITION_VAR)
    else:
        print(f"Warning: Cannot filter by variable. Column '{VARIABLE_COL}' not found.")

    # attempt to find a 'value' column and extract latitude/longitude before showing overview
    value_col = find_column(df.columns.tolist(), VALUE_COL_CANDIDATES)
    if value_col:
        try:
            extract_lat_lon_and_remove_value(df, value_col)
            print(f"Extracted latitude/longitude from column '{value_col}' into 'latitude' and 'longitude' and removed '{value_col}' from display.\n")
        except Exception as ex:
            print(f"Warning: failed to extract lat/lon from '{value_col}': {ex}", file=sys.stderr)

    show_overview(df)

    device_col = pick_device_column(df)
    if device_col:
        print(f"Detected device column: {device_col}")
    else:
        print("No device column automatically detected (candidates tried):", DEVICE_COL_CANDIDATES)

    if args.device:
        if not device_col:
            print("Cannot filter by device: no device column detected.", file=sys.stderr)
            sys.exit(5)
        df = filter_by_device(df, device_col, args.device)
        if df.empty:
            print(f"No rows found for device '{args.device}'.", file=sys.stderr)
            sys.exit(6)
        print(f"Filtered to device '{args.device}': {len(df)} rows\n")
    else:
        # show sample device values if device column exists
        if device_col:
            unique = df[device_col].dropna().unique()
            sample = list(map(str, unique[:10]))
            print(f"Sample device ids ({min(len(unique),10)} shown): {sample}")
            print("Pass --device <id> to filter the table.\n")

    cols = select_columns(df, args.columns)
    to_show = df[cols]

    if args.head == 0:
        # show entire selection
        try:
            print(to_show.to_string(index=False))
        except Exception:
            # fallback to pandas default print
            print(to_show)
    else:
        print(to_show.head(args.head).to_string(index=False))


if __name__ == "__main__":
    main()