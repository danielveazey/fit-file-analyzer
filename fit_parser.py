# fit_parser.py
import fitparse
import pandas as pd
from datetime import timezone
import logging
import numpy as np
import io

logger = logging.getLogger(__name__)

# Constants
SEMICIRCLE_TO_DEGREE = 180.0 / (2**31)
MOVING_THRESHOLD_MS = 0.5 # Approx 1.1 mph / 1.8 km/h - speed below this is considered stopped

def parse_fit_file(file_path_or_buffer):
    """
    Parses a FIT file and returns a Pandas DataFrame and summary statistics.
    """
    data = []
    start_time = None
    end_time = None

    required_fields = {'timestamp'}
    optional_fields = {
        'position_lat', 'position_long', 'distance', 'enhanced_speed', 'speed',
        'enhanced_altitude', 'altitude', 'heart_rate', 'cadence', 'power',
        'temperature'
    }
    available_fields = set()
    required_fields_found = {'timestamp': False, 'distance': False, 'speed': False}

    try:
        fitfile = fitparse.FitFile(file_path_or_buffer)
        messages = list(fitfile.get_messages('record')) # Get all record messages

        if not messages: logger.warning("No 'record' messages found."); return pd.DataFrame(), None
        logger.info(f"Found {len(messages)} 'record' messages.")

        # Determine available fields from *all* records for robustness
        all_record_fields = set().union(*(set(f.name for f in msg if hasattr(f, 'name')) for msg in messages))
        logger.info(f"Available fields in records: {all_record_fields}")

        # Determine fields to actually extract
        fields_to_extract = required_fields.union(optional_fields.intersection(all_record_fields))
        logger.info(f"Fields to extract: {fields_to_extract}")

        # Log if essential summary fields are missing
        for key in ['timestamp', 'distance', 'speed']:
            # Speed requires either speed or enhanced_speed
            field_present = key in fields_to_extract if key != 'speed' else \
                           ('speed' in fields_to_extract or 'enhanced_speed' in fields_to_extract)
            if field_present: required_fields_found[key] = True
            else: logger.warning(f"{key.capitalize()} field missing in FIT records! Related summaries may be unavailable.")

        # Process Records
        for record in messages:
            row = {}
            if not (record.mesg_type and record.mesg_type.name == 'record'): continue
            record_data = record.get_values()

            # Timestamp is critical
            ts_val = record_data.get('timestamp');
            if ts_val is None: continue # Skip records without timestamp
            row['timestamp'] = ts_val.replace(tzinfo=timezone.utc)
            if start_time is None: start_time = row['timestamp']
            end_time = row['timestamp'] # Continuously update end time

            # Extract other fields if available and requested
            if 'position_lat' in fields_to_extract: row['latitude'] = record_data.get('position_lat')
            if 'position_long' in fields_to_extract: row['longitude'] = record_data.get('position_long')
            if 'distance' in fields_to_extract: row['distance'] = record_data.get('distance')
            if 'enhanced_speed' in fields_to_extract: row['speed_ms'] = record_data.get('enhanced_speed')
            elif 'speed' in fields_to_extract: row['speed_ms'] = record_data.get('speed')
            if 'enhanced_altitude' in fields_to_extract: row['altitude'] = record_data.get('enhanced_altitude')
            elif 'altitude' in fields_to_extract: row['altitude'] = record_data.get('altitude')
            if 'heart_rate' in fields_to_extract: row['heart_rate'] = record_data.get('heart_rate')
            if 'cadence' in fields_to_extract: row['cadence'] = record_data.get('cadence')
            if 'power' in fields_to_extract: row['power'] = record_data.get('power')
            if 'temperature' in fields_to_extract: row['temperature'] = record_data.get('temperature')

            data.append(row)

        if not data: logger.warning("No records with timestamps processed."); return pd.DataFrame(), None

        # Create DataFrame
        df = pd.DataFrame(data)
        logger.info(f"Created initial DataFrame with {len(df)} rows.")

        # --- Post-processing ---
        # Convert timestamp, filter NAs
        try:
             df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.tz_convert(None)
             initial_len = len(df)
             df = df.dropna(subset=['timestamp'])
             if len(df) < initial_len: logger.warning(f"Dropped {initial_len - len(df)} rows due to invalid timestamp.")
             if df.empty: logger.error("No valid timestamps found after conversion."); return pd.DataFrame(), None
        except Exception as e: logger.error(f"Timestamp conversion failed: {e}", exc_info=True); return pd.DataFrame(), None

        # Sort by timestamp
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        # Convert lat/lon
        if 'latitude' in df.columns: df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce') * SEMICIRCLE_TO_DEGREE
        if 'longitude' in df.columns: df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce') * SEMICIRCLE_TO_DEGREE
        # Basic validation (could add dropna here too)
        if 'latitude' in df.columns: df.loc[(df['latitude'] > 90) | (df['latitude'] < -90), 'latitude'] = None
        if 'longitude' in df.columns: df.loc[(df['longitude'] > 181) | (df['longitude'] < -181), 'longitude'] = None


        # Convert other numerics, coerce errors to NaN
        numeric_cols = ['speed_ms', 'altitude', 'distance', 'heart_rate', 'cadence', 'power', 'temperature']
        for col in numeric_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        # Optional: Convert suitable columns to nullable Integers
        for col in ['heart_rate', 'cadence', 'temperature']:
             if col in df.columns and df[col].notna().any():
                try: df[col] = df[col].astype('Int64')
                except Exception as e: logger.warning(f"Could not convert column {col} to Int64: {e}")


        # Calculate derived columns
        if 'timestamp' in df.columns and not df.empty : df['elapsed_time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        else: df['elapsed_time_s'] = pd.NA
        if 'speed_ms' in df.columns and df['speed_ms'].notna().any(): df['speed_kmh'] = df['speed_ms'] * 3.6


        # --- Calculate Summary Statistics ---
        summary = { k: None for k in ['start_time', 'total_time', 'total_distance', 'avg_speed', 'avg_heart_rate', 'avg_cadence', 'avg_power', 'max_hr', 'max_cadence', 'max_power', 'min_temp_c', 'max_temp_c', 'avg_temp_c', 'total_elevation_gain_m', 'moving_time_seconds'] }

        if start_time and end_time:
            summary['start_time'] = start_time.replace(tzinfo=None)
            calc_total_time = (end_time - start_time).total_seconds()
            summary['total_time'] = calc_total_time if calc_total_time >= 0 else None
        if summary['total_time'] is None: logger.warning("Total time calculation failed or was negative.")

        # Get last valid distance
        if 'distance' in df.columns and df['distance'].notna().any():
            final_dist = df['distance'].dropna().iloc[-1]; summary['total_distance'] = final_dist / 1000.0 if pd.notna(final_dist) else None
        elif required_fields_found['distance']: logger.warning("Distance field present but no valid data points found.")

        # Averages
        if 'speed_kmh' in df.columns and df['speed_kmh'].notna().any(): summary['avg_speed'] = df['speed_kmh'].mean(skipna=True)
        if 'heart_rate' in df.columns and df['heart_rate'].notna().any(): summary['avg_heart_rate'] = df['heart_rate'].mean(skipna=True)
        if 'cadence' in df.columns and df['cadence'].notna().any(): summary['avg_cadence'] = df['cadence'].mean(skipna=True)
        if 'power' in df.columns and df['power'].notna().any(): summary['avg_power'] = df['power'].mean(skipna=True)

        # Max values
        if 'heart_rate' in df.columns and df['heart_rate'].notna().any(): summary['max_hr'] = df['heart_rate'].max(skipna=True)
        if 'cadence' in df.columns and df['cadence'].notna().any(): summary['max_cadence'] = df['cadence'].max(skipna=True)
        if 'power' in df.columns and df['power'].notna().any(): summary['max_power'] = df['power'].max(skipna=True)

        # Temperature
        if 'temperature' in df.columns and df['temperature'].notna().any():
            summary['min_temp_c'] = df['temperature'].min(skipna=True)
            summary['max_temp_c'] = df['temperature'].max(skipna=True)
            summary['avg_temp_c'] = df['temperature'].mean(skipna=True)

        # Elevation Gain
        summary['total_elevation_gain_m'] = 0.0 # Default
        if 'altitude' in df.columns and df['altitude'].notna().any():
            alt_series = df['altitude'].dropna(); gain = alt_series.diff().clip(lower=0).sum() if len(alt_series) > 1 else 0.0; summary['total_elevation_gain_m'] = float(gain) if pd.notna(gain) else 0.0

        # Moving Time
        summary['moving_time_seconds'] = 0.0 # Default
        if 'speed_ms' in df.columns and df['speed_ms'].notna().any() and 'timestamp' in df.columns and len(df) > 1:
             time_diff = df['timestamp'].diff().dt.total_seconds().iloc[1:]; is_moving = (df['speed_ms'].iloc[1:] > MOVING_THRESHOLD_MS); moving_time = time_diff[is_moving].sum(); summary['moving_time_seconds'] = float(moving_time) if pd.notna(moving_time) else 0.0

        # Final Formatting Pass
        for key, val in summary.items():
            if pd.isna(val): summary[key] = None
            elif key != 'start_time': # Don't process timestamp
                try: # Try converting to float first for rounding/general numeric handling
                     summary[key] = round(float(val), 1) # Round numeric values
                except (ValueError, TypeError): summary[key] = None # If not convertible to float, set None


        logger.info(f"Parsing successful. Summary Dict: {summary}")
        # log_stream = io.StringIO(); df.info(buf=log_stream); logger.info(f"DataFrame Info:\n{log_stream.getvalue()}") # Removed verbose logging
        return df, summary

    # --- Error Handling ---
    except fitparse.FitParseError as e: logger.error(f"FIT Parse Error: {e}", exc_info=True); return pd.DataFrame(), None
    except Exception as e: logger.error(f"Unexpected FIT Parsing Error: {e}", exc_info=True); return pd.DataFrame(), None