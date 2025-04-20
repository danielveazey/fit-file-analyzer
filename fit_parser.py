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
MOVING_THRESHOLD_MS = 0.5 # Speed below this (m/s) is considered stopped

def parse_fit_file(file_path_or_buffer):
    """
    Parses FIT file:
    1. Extracts definitions (HR zones, Power zones) and summaries (session, lap).
    2. Extracts ALL fields from 'record' messages into a DataFrame.
    3. Calculates summary statistics and time in zones based on the record data.
    """
    records_data = []
    hr_zones = {}   # Store HR zone upper bounds {zone_index: high_bpm}
    power_zones = {} # Store Power zone upper bounds {zone_index: high_watts}
    file_summary = {} # Store general file info if needed
    session_summary = {} # Store session message data
    lap_summaries = [] # Store lap message data
    start_time = None
    end_time = None
    last_calories = None # Keep track of the latest calorie value

    try:
        fitfile = fitparse.FitFile(file_path_or_buffer)
        all_messages = list(fitfile.get_messages()) # Read all messages once

        # --- First Pass: Extract Definitions and High-Level Summaries ---
        logger.info(f"Found {len(all_messages)} total messages. Extracting definitions and non-record data...")
        for message in all_messages:
            msg_type = message.name

            # Store HR zone definitions
            if msg_type == 'hr_zone':
                msg_dict = message.get_values()
                zone_index = msg_dict.get('message_index') # Typically 0-based zone number
                high_bpm = msg_dict.get('high_bpm')
                if zone_index is not None and high_bpm is not None:
                    hr_zones[zone_index] = high_bpm
                    logger.debug(f"Found HR Zone {zone_index}: Upper Bound {high_bpm} bpm")

            # Store Power zone definitions
            elif msg_type == 'power_zone':
                msg_dict = message.get_values()
                zone_index = msg_dict.get('message_index')
                high_watts = msg_dict.get('high_watts')
                if zone_index is not None and high_watts is not None:
                    power_zones[zone_index] = high_watts
                    logger.debug(f"Found Power Zone {zone_index}: Upper Bound {high_watts} W")

            # Potentially store other message types like 'session', 'lap', 'file_id'
            elif msg_type == 'session':
                session_summary = message.get_values() # Overwrite if multiple sessions exist (usually only one)
            elif msg_type == 'lap':
                lap_summaries.append(message.get_values())
            elif msg_type == 'file_id':
                file_summary = message.get_values()

            # ** Store record messages separately for detailed DataFrame **
            elif msg_type == 'record':
                records_data.append(message.get_values()) # Store the raw dictionary

        # Sort zones by index for correct processing later
        hr_zones = dict(sorted(hr_zones.items()))
        power_zones = dict(sorted(power_zones.items()))
        logger.info(f"Found {len(hr_zones)} HR Zones, {len(power_zones)} Power Zones defined.")
        logger.info(f"Found {len(lap_summaries)} Lap messages.")
        if session_summary: logger.info("Found Session message.")
        if file_summary: logger.info("Found File ID message.")


        # --- Process Record Data ---
        if not records_data:
            logger.warning("No 'record' messages found for detailed analysis.")
            # Still possible to return summary based on 'session'/'lap' messages if needed
            # For consistency, return empty DF and None summary if no records
            return pd.DataFrame(), None

        logger.info(f"Processing {len(records_data)} 'record' messages for DataFrame, extracting *all* fields.")

        processed_records = []

        for record_dict in records_data:
            row = {}
            ts_val = record_dict.get('timestamp')
            if ts_val is None:
                logger.debug("Skipping record message without timestamp.")
                continue # Must have timestamp

            # Process Timestamp first
            row['timestamp'] = ts_val.replace(tzinfo=timezone.utc)
            if start_time is None: start_time = row['timestamp']
            end_time = row['timestamp'] # Keep updating end time

            # --- Dynamically process ALL fields in the record ---
            temp_speed = None
            temp_altitude = None
            has_enhanced_speed = False
            has_enhanced_altitude = False

            for field_name, value in record_dict.items():
                if value is None: # Skip fields with None value
                     continue

                # Skip timestamp as it's already handled
                if field_name == 'timestamp':
                    continue

                # Handle Lat/Lon potential renaming and conversion later
                elif field_name == 'position_lat':
                    row['latitude_raw'] = value # Store raw semicircles
                elif field_name == 'position_long':
                    row['longitude_raw'] = value # Store raw semicircles

                # Handle Speed - Prefer enhanced, store intermediate
                elif field_name == 'enhanced_speed':
                    temp_speed = float(value)
                    has_enhanced_speed = True
                elif field_name == 'speed' and not has_enhanced_speed:
                    try: temp_speed = float(value)
                    except (ValueError, TypeError): pass # Ignore if cannot convert

                # Handle Altitude - Prefer enhanced, store intermediate
                elif field_name == 'enhanced_altitude':
                    temp_altitude = float(value)
                    has_enhanced_altitude = True
                elif field_name == 'altitude' and not has_enhanced_altitude:
                    try: temp_altitude = float(value)
                    except (ValueError, TypeError): pass # Ignore if cannot convert

                # Store Calories separately for last known value logic
                elif field_name == 'calories':
                    row['calories'] = value # Keep original value in DF column
                    try:
                        last_calories = int(value) # Update last known integer value for summary
                    except (ValueError, TypeError):
                        pass # Ignore non-integer calorie values for summary

                # Store all other fields directly
                else:
                    row[field_name] = value

            # --- Post-processing for combined fields (Speed/Altitude) ---
            if temp_speed is not None:
                row['speed_ms'] = temp_speed # Final speed in m/s
            if temp_altitude is not None:
                row['altitude'] = temp_altitude # Final altitude in m

            processed_records.append(row)

        if not processed_records:
            logger.error("No records processed successfully (likely all missing timestamps).")
            return pd.DataFrame(), None

        # --- DataFrame Creation and Post-Processing ---
        df = pd.DataFrame(processed_records)
        # df.info() # Debug: Show DataFrame structure and types
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns from records.")

        # Timestamp conversion (critical)
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            # Convert to timezone naive UTC for calculations, handle potential errors
            df['timestamp'] = df['timestamp'].dt.tz_convert(None)
            df = df.dropna(subset=['timestamp']) # Drop rows where timestamp conversion failed
        except Exception as e:
            logger.error(f"Timestamp conversion failed post-DF: {e}", exc_info=True)
            return pd.DataFrame(), None
        if df.empty:
            logger.error("No valid timestamps after conversion.")
            return pd.DataFrame(), None
        df = df.sort_values(by='timestamp').reset_index(drop=True) # Ensure sort before calculations

        # Lat/Lon conversion & validation (using the raw fields)
        if 'latitude_raw' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude_raw'], errors='coerce') * SEMICIRCLE_TO_DEGREE
            df.loc[~df['latitude'].between(-90, 90, inclusive='both'), 'latitude'] = None # inclusive='both' is default but explicit
            df = df.drop(columns=['latitude_raw'], errors='ignore') # Drop raw column
        if 'longitude_raw' in df.columns:
            df['longitude'] = pd.to_numeric(df['longitude_raw'], errors='coerce') * SEMICIRCLE_TO_DEGREE
            df.loc[~df['longitude'].between(-180, 180, inclusive='both'), 'longitude'] = None # Standard range for longitude
            df = df.drop(columns=['longitude_raw'], errors='ignore') # Drop raw column


        # --- Convert known numeric columns, coerce errors ---
        # Define a list of columns commonly expected to be numeric
        # Add other common numeric fields found in FIT files if needed
        common_numeric_cols = [
            'speed_ms', 'altitude', 'distance', 'heart_rate', 'cadence', 'power',
            'temperature', 'calories', 'grade', 'left_right_balance',
            'left_torque_effectiveness', 'right_torque_effectiveness',
            'left_pedal_smoothness', 'right_pedal_smoothness', 'combined_pedal_smoothness',
            'gps_accuracy' # Add any other frequently numeric fields you encounter
        ]
        for col in df.columns:
            # Attempt conversion if column is potentially numeric OR is in our common list
            # This tries to catch numeric columns even if not explicitly listed
            is_potentially_numeric = pd.api.types.is_numeric_dtype(df[col]) or \
                                     pd.api.types.is_timedelta64_dtype(df[col]) or \
                                     col in common_numeric_cols

            if is_potentially_numeric:
                 # Only attempt conversion if not already a suitable numeric type (float, int)
                 # This avoids unnecessary operations and potential type changes (e.g., int to float)
                 if not (pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col])):
                     logger.debug(f"Attempting numeric conversion for column: {col}")
                     original_type = df[col].dtype
                     df[col] = pd.to_numeric(df[col], errors='coerce')
                     # Log if type changed or conversion failed (all NaNs)
                     if df[col].dtype != original_type:
                         if df[col].isna().all() and not df[col].empty:
                              logger.warning(f"Column '{col}' became all NaNs after numeric conversion from {original_type}.")
                         else:
                              logger.debug(f"Converted column '{col}' from {original_type} to {df[col].dtype}.")
                 else:
                     logger.debug(f"Column '{col}' is already numeric ({df[col].dtype}), skipping conversion.")

        # --- Calculate derived columns ---
        if not df.empty :
            df['elapsed_time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        else:
            df['elapsed_time_s'] = pd.NA # Assign pandas NA

        if 'speed_ms' in df.columns and df['speed_ms'].notna().any():
            df['speed_kmh'] = df['speed_ms'] * 3.6


        # --- Calculate Summary Statistics ---
        # Initialize summary dictionary (same keys as before for consistency)
        summary = { k: None for k in [ 'start_time', 'total_time', 'total_distance', 'avg_speed', 'avg_heart_rate', 'avg_cadence', 'avg_power', 'max_hr', 'max_cadence', 'max_power', 'min_temp_c', 'max_temp_c', 'avg_temp_c', 'total_elevation_gain_m', 'moving_time_seconds', 'total_calories'] }

        # Time
        if start_time and end_time:
            # Use the converted, timezone-naive timestamps from the DF if available and valid
            if not df.empty and 'timestamp' in df.columns:
                first_ts = df['timestamp'].iloc[0]
                last_ts = df['timestamp'].iloc[-1]
                if pd.notna(first_ts) and pd.notna(last_ts):
                    summary['start_time'] = first_ts # Already timezone naive
                    duration_seconds = (last_ts - first_ts).total_seconds()
                    summary['total_time'] = duration_seconds if duration_seconds >= 0 else None
                    if summary['total_time'] is None:
                        logger.warning("Calculated total time negative, setting to None.")
                else:
                     logger.warning("Could not determine valid start/end time from DataFrame.")
            else: # Fallback to originally captured start/end times (less precise)
                summary['start_time'] = start_time.replace(tzinfo=None)
                duration_seconds = (end_time - start_time).total_seconds()
                summary['total_time'] = duration_seconds if duration_seconds >= 0 else None
                if summary['total_time'] is None:
                      logger.warning("Calculated total time negative (fallback), setting to None.")

        # Distance & Speed Averages
        if 'distance' in df.columns and df['distance'].notna().any():
            final_dist = df['distance'].dropna().iloc[-1] if not df['distance'].dropna().empty else None
            summary['total_distance'] = final_dist / 1000.0 if pd.notna(final_dist) else None
        # Use calculated speed_kmh if available
        if 'speed_kmh' in df.columns and df['speed_kmh'].notna().any():
            summary['avg_speed'] = df['speed_kmh'].mean(skipna=True)
        # Fallback: calculate avg speed from total distance / total time if speed data missing
        elif summary['total_distance'] is not None and summary['total_time'] is not None and summary['total_time'] > 0:
             summary['avg_speed'] = (summary['total_distance'] / (summary['total_time'] / 3600.0))
             logger.info("Calculated avg speed from distance/time due to missing speed data.")


        # HR, Cad, Power Avg/Max
        for key in ['heart_rate', 'cadence', 'power']:
            if key in df.columns and df[key].notna().any():
                # Ensure column is numeric before calculating mean/max
                if pd.api.types.is_numeric_dtype(df[key]):
                    summary[f'avg_{key}'] = df[key].mean(skipna=True)
                    if key == 'heart_rate':
                        summary['max_hr'] = df[key].max(skipna=True)
                    else:
                        summary[f'max_{key}'] = df[key].max(skipna=True)
                else:
                     logger.warning(f"Column '{key}' exists but is not numeric. Cannot calculate stats.")

        # Temp Min/Max/Avg
        if 'temperature' in df.columns and df['temperature'].notna().any():
             if pd.api.types.is_numeric_dtype(df['temperature']):
                 summary['min_temp_c'] = df['temperature'].min(skipna=True)
                 summary['max_temp_c'] = df['temperature'].max(skipna=True)
                 summary['avg_temp_c'] = df['temperature'].mean(skipna=True)
             else:
                  logger.warning("Column 'temperature' exists but is not numeric. Cannot calculate stats.")


        # Elevation Gain
        summary['total_elevation_gain_m'] = 0.0
        if 'altitude' in df.columns and df['altitude'].notna().any():
             if pd.api.types.is_numeric_dtype(df['altitude']):
                 alt_series = df['altitude'].dropna()
                 gain = alt_series.diff().clip(lower=0).sum() if len(alt_series) > 1 else 0.0
                 summary['total_elevation_gain_m'] = float(gain) if pd.notna(gain) else 0.0
             else:
                  logger.warning("Column 'altitude' exists but is not numeric. Cannot calculate gain.")

        # Moving Time
        summary['moving_time_seconds'] = 0.0
        if 'speed_ms' in df.columns and df['speed_ms'].notna().any() and len(df) > 1:
             if pd.api.types.is_numeric_dtype(df['speed_ms']):
                 time_diff = df['timestamp'].diff().dt.total_seconds().iloc[1:] # Exclude first row's NaN diff
                 is_moving = (df['speed_ms'].iloc[1:] > MOVING_THRESHOLD_MS)
                 # Ensure indices align before boolean indexing
                 if len(time_diff) == len(is_moving):
                      moving_time = time_diff[is_moving].sum()
                      summary['moving_time_seconds'] = float(moving_time) if pd.notna(moving_time) else 0.0
                 else:
                      logger.error("Length mismatch calculating moving time.")
             else:
                  logger.warning("Column 'speed_ms' exists but is not numeric. Cannot calculate moving time.")


        # Calories - Use the last tracked value
        summary['total_calories'] = last_calories # Assign last seen integer value

        # --- Time in Zone Calculation ---
        MAX_HR_ZONES = 6
        MAX_PWR_ZONES = 7
        for i in range(MAX_HR_ZONES): summary[f'time_in_hr_zone_{i}'] = 0.0
        for i in range(MAX_PWR_ZONES): summary[f'time_in_pwr_zone_{i}'] = 0.0

        if len(df) > 1:
            # Calculate time differences *between* consecutive records
            time_diff = df['timestamp'].diff().dt.total_seconds() # Series with time deltas, NaN for first

            # Assign time delta to the *interval ending* at the timestamp
            # For zone calculations, we typically consider the value *at the end* of the interval
            # Example: time spent between row 0 and row 1 is time_diff[1]
            #          value at end of this interval is df['heart_rate'][1]

            # Calculate Time in HR Zones
            if hr_zones and 'heart_rate' in df.columns and df['heart_rate'].notna().any():
                if pd.api.types.is_numeric_dtype(df['heart_rate']):
                    logger.info("Calculating time in HR zones...")
                    lower_bound = 0
                    for i in range(MAX_HR_ZONES):
                        is_in_zone = pd.Series([False] * len(df)) # Initialize boolean Series
                        if i in hr_zones: # Zone with defined upper bound
                            upper_bound = hr_zones[i]
                            is_in_zone = (df['heart_rate'] > lower_bound) & (df['heart_rate'] <= upper_bound)
                            lower_bound = upper_bound
                        elif i == len(hr_zones) and i > 0: # Last zone (if zones are defined) -> infinity
                             # Check previous lower bound exists
                             prev_zone_idx = i - 1
                             if prev_zone_idx in hr_zones:
                                  lower_bound = hr_zones[prev_zone_idx]
                                  is_in_zone = (df['heart_rate'] > lower_bound)
                             else: # Should not happen if zones sorted, but safe check
                                 logger.warning(f"Cannot determine lower bound for final HR Zone {i}, skipping.")
                                 continue
                        else: # Gap in zone definitions? Or fewer zones than MAX_HR_ZONES? Skip.
                             logger.debug(f"HR Zone definition for index {i} missing or invalid, skipping.")
                             continue # Skip this zone index entirely

                        # Sum time diff where the *endpoint* of the interval falls in the zone
                        # Use .iloc[1:] to align with time_diff which starts at index 1
                        valid_indices = is_in_zone.iloc[1:].index[is_in_zone.iloc[1:]]
                        time_in_this_zone = time_diff.loc[valid_indices].sum()
                        summary[f'time_in_hr_zone_{i}'] = float(time_in_this_zone) if pd.notna(time_in_this_zone) else 0.0

                    for i in range(MAX_HR_ZONES): logger.debug(f"Time in HR Zone {i}: {summary.get(f'time_in_hr_zone_{i}', 0.0):.1f}s")
                else:
                     logger.warning("Column 'heart_rate' exists but is not numeric. Cannot calculate HR zones.")


            # Calculate Time in Power Zones (similar logic)
            if power_zones and 'power' in df.columns and df['power'].notna().any():
                 if pd.api.types.is_numeric_dtype(df['power']):
                     logger.info("Calculating time in Power zones...")
                     lower_bound = 0
                     for i in range(MAX_PWR_ZONES):
                          is_in_zone = pd.Series([False] * len(df)) # Initialize boolean Series
                          if i in power_zones:
                              upper_bound = power_zones[i]
                              is_in_zone = (df['power'] > lower_bound) & (df['power'] <= upper_bound)
                              lower_bound = upper_bound
                          elif i == len(power_zones) and i > 0: # Last zone -> infinity
                              prev_zone_idx = i - 1
                              if prev_zone_idx in power_zones:
                                   lower_bound = power_zones[prev_zone_idx]
                                   is_in_zone = (df['power'] > lower_bound)
                              else: logger.warning(f"Cannot determine lower bound for final Power Zone {i}, skipping."); continue
                          else: logger.debug(f"Power Zone definition for index {i} missing or invalid, skipping."); continue

                          valid_indices = is_in_zone.iloc[1:].index[is_in_zone.iloc[1:]]
                          time_in_this_zone = time_diff.loc[valid_indices].sum()
                          summary[f'time_in_pwr_zone_{i}'] = float(time_in_this_zone) if pd.notna(time_in_this_zone) else 0.0

                     for i in range(MAX_PWR_ZONES): logger.debug(f"Time in Power Zone {i}: {summary.get(f'time_in_pwr_zone_{i}', 0.0):.1f}s")
                 else:
                     logger.warning("Column 'power' exists but is not numeric. Cannot calculate Power zones.")


        # --- Final Formatting Pass ---
        for key, val in summary.items():
            if pd.isna(val):
                summary[key] = None
            elif key != 'start_time':
                try:
                    f_val = float(val)
                    # Decide rounding based on key
                    if key in ['total_time', 'moving_time_seconds'] or 'time_in_' in key:
                        summary[key] = round(f_val, 0) # Round times to nearest second for display
                    elif key == 'total_calories':
                        summary[key] = int(f_val) if f_val >= 0 else 0 # Calories as non-negative integer
                    else:
                        summary[key] = round(f_val, 1) # Most others to 1 decimal
                except (ValueError, TypeError):
                    summary[key] = None # Set to None if conversion fails

        logger.info(f"Parsing successful. Final Summary: { {k: v for k, v in summary.items() if not k.startswith('time_in')} }") # Log summary without zones for brevity
        return df, summary

    # --- Error Handling ---
    except fitparse.FitParseError as e:
        logger.error(f"FIT Parse Error: {e}", exc_info=True)
        return pd.DataFrame(), None
    except Exception as e:
        logger.error(f"Unexpected FIT Parsing Error: {e}", exc_info=True)
        return pd.DataFrame(), None

