# fit_parser.py
import fitparse
import pandas as pd
from datetime import timezone, timedelta # Added timedelta
import logging
import numpy as np
import io
import pytz # Added pytz
from timezonefinder import TimezoneFinder # Added timezonefinder

logger = logging.getLogger(__name__)

# Initialize TimezoneFinder - this can take a second or two the first time
tf = TimezoneFinder()

# Constants
SEMICIRCLE_TO_DEGREE = 180.0 / (2**31)
MOVING_THRESHOLD_MS = 0.5 # Speed below this (m/s) is considered stopped

def parse_fit_file(file_path_or_buffer):
    """
    Parses FIT file:
    1. Extracts definitions (HR zones, Power zones) and summaries (session, lap).
    2. Extracts ALL fields from 'record' messages into a DataFrame.
    3. Determines local time zone from initial coordinates.
    4. Calculates summary statistics (using local time where appropriate) and time in zones.
    """
    records_data = []
    hr_zones = {}
    power_zones = {}
    file_summary = {}
    session_summary = {}
    lap_summaries = []
    start_time_utc = None # Store the first timestamp encountered (UTC)
    end_time_utc = None   # Store the last timestamp encountered (UTC)
    last_calories = None
    local_timezone_str = None # To store the determined timezone string
    first_valid_lat = None
    first_valid_lon = None

    try:
        fitfile = fitparse.FitFile(file_path_or_buffer)
        all_messages = list(fitfile.get_messages()) # Read all messages once

        # --- First Pass: Extract Definitions and High-Level Summaries ---
        logger.info(f"Found {len(all_messages)} total messages. Extracting definitions and non-record data...")
        for message in all_messages:
            msg_type = message.name
            msg_dict = message.get_values()

            # Store HR zone definitions
            if msg_type == 'hr_zone':
                zone_index = msg_dict.get('message_index')
                high_bpm = msg_dict.get('high_bpm')
                if zone_index is not None and high_bpm is not None:
                    hr_zones[zone_index] = high_bpm
                    logger.debug(f"Found HR Zone {zone_index}: Upper Bound {high_bpm} bpm")

            # Store Power zone definitions
            elif msg_type == 'power_zone':
                zone_index = msg_dict.get('message_index')
                high_watts = msg_dict.get('high_watts')
                if zone_index is not None and high_watts is not None:
                    power_zones[zone_index] = high_watts
                    logger.debug(f"Found Power Zone {zone_index}: Upper Bound {high_watts} W")

            elif msg_type == 'session':
                session_summary = msg_dict
            elif msg_type == 'lap':
                lap_summaries.append(msg_dict)
            elif msg_type == 'file_id':
                file_summary = msg_dict

            elif msg_type == 'record':
                records_data.append(msg_dict) # Store raw dictionary

                # --- Attempt to get first valid coordinates for timezone lookup ---
                if first_valid_lat is None or first_valid_lon is None:
                    lat = msg_dict.get('position_lat')
                    lon = msg_dict.get('position_long')
                    if lat is not None and lon is not None:
                        try:
                            # Convert potential semicircles immediately for lookup
                            lat_deg = float(lat) * SEMICIRCLE_TO_DEGREE
                            lon_deg = float(lon) * SEMICIRCLE_TO_DEGREE
                            # Basic validation
                            if -90 <= lat_deg <= 90 and -180 <= lon_deg <= 180:
                                first_valid_lat = lat_deg
                                first_valid_lon = lon_deg
                                logger.info(f"Found first valid coordinates for TZ lookup: Lat={first_valid_lat:.5f}, Lon={first_valid_lon:.5f}")
                        except (ValueError, TypeError):
                            pass # Ignore if conversion fails


        # Sort zones by index
        hr_zones = dict(sorted(hr_zones.items()))
        power_zones = dict(sorted(power_zones.items()))
        logger.info(f"Found {len(hr_zones)} HR Zones, {len(power_zones)} Power Zones defined.")
        logger.info(f"Found {len(lap_summaries)} Lap messages.")
        if session_summary: logger.info("Found Session message.")
        if file_summary: logger.info("Found File ID message.")


        # --- Determine Time Zone ---
        if first_valid_lat is not None and first_valid_lon is not None:
            local_timezone_str = tf.timezone_at(lng=first_valid_lon, lat=first_valid_lat)
            if local_timezone_str:
                logger.info(f"Determined time zone: {local_timezone_str}")
            else:
                logger.warning(f"Could not determine time zone for Lat={first_valid_lat:.5f}, Lon={first_valid_lon:.5f} (possibly over water?). Using UTC.")
                local_timezone_str = 'UTC' # Fallback to UTC
        else:
            logger.warning("No valid coordinates found in records. Using UTC as time zone.")
            local_timezone_str = 'UTC' # Fallback to UTC


        # --- Process Record Data ---
        if not records_data:
            logger.warning("No 'record' messages found for detailed analysis.")
            return pd.DataFrame(), None, None # Return None for timezone too

        logger.info(f"Processing {len(records_data)} 'record' messages for DataFrame...")
        processed_records = []

        for record_dict in records_data:
            row = {}
            ts_val = record_dict.get('timestamp')
            if ts_val is None: continue

            # Store UTC timestamp internally
            ts_utc = ts_val.replace(tzinfo=timezone.utc)
            row['timestamp'] = ts_utc # Store the raw datetime obj with UTC tzinfo
            if start_time_utc is None: start_time_utc = ts_utc
            end_time_utc = ts_utc # Keep updating last UTC time

            # Process all other fields dynamically
            temp_speed, temp_altitude = None, None
            has_enhanced_speed, has_enhanced_altitude = False, False

            for field_name, value in record_dict.items():
                if value is None: continue
                if field_name == 'timestamp': continue

                if field_name == 'position_lat': row['latitude_raw'] = value
                elif field_name == 'position_long': row['longitude_raw'] = value
                elif field_name == 'enhanced_speed': temp_speed, has_enhanced_speed = float(value), True
                elif field_name == 'speed' and not has_enhanced_speed:
                    try: temp_speed = float(value)
                    except (ValueError, TypeError): pass
                elif field_name == 'enhanced_altitude': temp_altitude, has_enhanced_altitude = float(value), True
                elif field_name == 'altitude' and not has_enhanced_altitude:
                    try: temp_altitude = float(value)
                    except (ValueError, TypeError): pass
                elif field_name == 'calories':
                    row['calories'] = value
                    try: last_calories = int(value)
                    except (ValueError, TypeError): pass
                else: row[field_name] = value

            if temp_speed is not None: row['speed_ms'] = temp_speed
            if temp_altitude is not None: row['altitude'] = temp_altitude

            processed_records.append(row)

        if not processed_records:
            logger.error("No records processed successfully.")
            return pd.DataFrame(), None, None

        # --- DataFrame Creation and Post-Processing ---
        df = pd.DataFrame(processed_records)
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns.")

        # Timestamp conversion: Convert to datetime objects, keep as UTC, then make NAIVE
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            df = df.dropna(subset=['timestamp']) # Drop invalid timestamps BEFORE making naive
            df['timestamp'] = df['timestamp'].dt.tz_convert(None) # Make naive UTC for internal calculations
        except Exception as e:
            logger.error(f"Timestamp conversion failed post-DF: {e}", exc_info=True)
            return pd.DataFrame(), None, None
        if df.empty:
            logger.error("No valid timestamps after conversion.")
            return pd.DataFrame(), None, None
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        # Lat/Lon conversion
        if 'latitude_raw' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude_raw'], errors='coerce') * SEMICIRCLE_TO_DEGREE
            df.loc[~df['latitude'].between(-90, 90, inclusive='both'), 'latitude'] = None
            df = df.drop(columns=['latitude_raw'], errors='ignore')
        if 'longitude_raw' in df.columns:
            df['longitude'] = pd.to_numeric(df['longitude_raw'], errors='coerce') * SEMICIRCLE_TO_DEGREE
            df.loc[~df['longitude'].between(-180, 180, inclusive='both'), 'longitude'] = None
            df = df.drop(columns=['longitude_raw'], errors='ignore')

        # --- Convert known numeric columns ---
        common_numeric_cols = [
            'speed_ms', 'altitude', 'distance', 'heart_rate', 'cadence', 'power',
            'temperature', 'calories', 'grade', 'left_right_balance',
            'left_torque_effectiveness', 'right_torque_effectiveness',
            'left_pedal_smoothness', 'right_pedal_smoothness', 'combined_pedal_smoothness',
            'gps_accuracy'
        ]
        for col in df.columns:
            if col not in ['timestamp', 'latitude', 'longitude']: # Avoid re-converting already handled cols
                 is_potentially_numeric = pd.api.types.is_numeric_dtype(df[col]) or col in common_numeric_cols
                 if is_potentially_numeric:
                     if not (pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col])):
                         original_type = df[col].dtype
                         df[col] = pd.to_numeric(df[col], errors='coerce')
                         if df[col].dtype != original_type and df[col].isna().all() and not df[col].empty:
                              logger.warning(f"Column '{col}' became all NaNs after numeric conversion from {original_type}.")


        # --- Calculate derived columns ---
        if not df.empty:
            df['elapsed_time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        else:
            df['elapsed_time_s'] = pd.NA

        if 'speed_ms' in df.columns and df['speed_ms'].notna().any():
            df['speed_kmh'] = df['speed_ms'] * 3.6


        # --- Calculate Summary Statistics ---
        summary = { k: None for k in [ 'start_time', 'total_time', 'total_distance', 'avg_speed', 'avg_heart_rate', 'avg_cadence', 'avg_power', 'max_hr', 'max_cadence', 'max_power', 'min_temp_c', 'max_temp_c', 'avg_temp_c', 'total_elevation_gain_m', 'moving_time_seconds', 'total_calories', 'timezone_str'] } # Added timezone_str

        # Store timezone
        summary['timezone_str'] = local_timezone_str

        # Time: Use DataFrame times (naive UTC) for duration, convert start time to local for summary
        first_ts_utc_naive = df['timestamp'].iloc[0] if not df.empty else None
        last_ts_utc_naive = df['timestamp'].iloc[-1] if not df.empty else None

        if first_ts_utc_naive is not None and last_ts_utc_naive is not None:
            duration_seconds = (last_ts_utc_naive - first_ts_utc_naive).total_seconds()
            summary['total_time'] = duration_seconds if duration_seconds >= 0 else None
            if summary['total_time'] is None: logger.warning("Total time negative.")

            # Convert start time to local time for display
            try:
                start_time_local = pytz.utc.localize(first_ts_utc_naive).astimezone(pytz.timezone(local_timezone_str))
                summary['start_time'] = start_time_local # Store the *local* time object
            except Exception as tz_err:
                logger.error(f"Failed to convert start time to local timezone '{local_timezone_str}': {tz_err}. Storing naive UTC.", exc_info=True)
                summary['start_time'] = first_ts_utc_naive # Fallback to naive UTC
                summary['timezone_str'] = 'UTC' # Reset timezone if conversion failed

        else: # Fallback if DataFrame times are invalid
             logger.warning("Using originally captured UTC times for summary (less precise).")
             if start_time_utc and end_time_utc:
                  duration_seconds = (end_time_utc - start_time_utc).total_seconds()
                  summary['total_time'] = duration_seconds if duration_seconds >= 0 else None
                  try:
                       start_time_local = start_time_utc.astimezone(pytz.timezone(local_timezone_str))
                       summary['start_time'] = start_time_local
                  except Exception as tz_err:
                       logger.error(f"Failed to convert fallback start time to local timezone '{local_timezone_str}': {tz_err}. Storing UTC.", exc_info=True)
                       summary['start_time'] = start_time_utc.replace(tzinfo=None) # Store naive UTC
                       summary['timezone_str'] = 'UTC'


        # Distance & Speed Averages (logic unchanged, based on df columns)
        if 'distance' in df.columns and df['distance'].notna().any():
            final_dist = df['distance'].dropna().iloc[-1] if not df['distance'].dropna().empty else None
            summary['total_distance'] = final_dist / 1000.0 if pd.notna(final_dist) else None
        if 'speed_kmh' in df.columns and df['speed_kmh'].notna().any():
            summary['avg_speed'] = df['speed_kmh'].mean(skipna=True)
        elif summary['total_distance'] is not None and summary['total_time'] is not None and summary['total_time'] > 0:
             summary['avg_speed'] = (summary['total_distance'] / (summary['total_time'] / 3600.0))


        # HR, Cad, Power Avg/Max (logic unchanged)
        for key in ['heart_rate', 'cadence', 'power']:
            if key in df.columns and df[key].notna().any():
                if pd.api.types.is_numeric_dtype(df[key]):
                    summary[f'avg_{key}'] = df[key].mean(skipna=True)
                    if key == 'heart_rate': summary['max_hr'] = df[key].max(skipna=True)
                    else: summary[f'max_{key}'] = df[key].max(skipna=True)

        # Temp Min/Max/Avg (logic unchanged)
        if 'temperature' in df.columns and df['temperature'].notna().any():
             if pd.api.types.is_numeric_dtype(df['temperature']):
                 summary['min_temp_c'] = df['temperature'].min(skipna=True)
                 summary['max_temp_c'] = df['temperature'].max(skipna=True)
                 summary['avg_temp_c'] = df['temperature'].mean(skipna=True)

        # Elevation Gain (logic unchanged)
        summary['total_elevation_gain_m'] = 0.0
        if 'altitude' in df.columns and df['altitude'].notna().any():
             if pd.api.types.is_numeric_dtype(df['altitude']):
                 alt_series = df['altitude'].dropna()
                 gain = alt_series.diff().clip(lower=0).sum() if len(alt_series) > 1 else 0.0
                 summary['total_elevation_gain_m'] = float(gain) if pd.notna(gain) else 0.0

        # Moving Time (logic unchanged, uses df timestamps)
        summary['moving_time_seconds'] = 0.0
        if 'speed_ms' in df.columns and df['speed_ms'].notna().any() and len(df) > 1:
             if pd.api.types.is_numeric_dtype(df['speed_ms']):
                 time_diff = df['timestamp'].diff().dt.total_seconds().iloc[1:]
                 is_moving = (df['speed_ms'].iloc[1:] > MOVING_THRESHOLD_MS)
                 if len(time_diff) == len(is_moving):
                      moving_time = time_diff[is_moving].sum()
                      summary['moving_time_seconds'] = float(moving_time) if pd.notna(moving_time) else 0.0

        # Calories
        summary['total_calories'] = last_calories

        # --- Time in Zone Calculation (logic unchanged, uses df timestamps) ---
        MAX_HR_ZONES = 6
        MAX_PWR_ZONES = 7
        for i in range(MAX_HR_ZONES): summary[f'time_in_hr_zone_{i}'] = 0.0
        for i in range(MAX_PWR_ZONES): summary[f'time_in_pwr_zone_{i}'] = 0.0

        if len(df) > 1:
            time_diff = df['timestamp'].diff().dt.total_seconds()

            # Calculate Time in HR Zones
            if hr_zones and 'heart_rate' in df.columns and df['heart_rate'].notna().any():
                if pd.api.types.is_numeric_dtype(df['heart_rate']):
                    logger.info("Calculating time in HR zones...")
                    lower_bound = 0
                    for i in range(MAX_HR_ZONES):
                        is_in_zone = pd.Series([False] * len(df))
                        if i in hr_zones:
                            upper_bound = hr_zones[i]
                            is_in_zone = (df['heart_rate'] > lower_bound) & (df['heart_rate'] <= upper_bound)
                            lower_bound = upper_bound
                        elif i == len(hr_zones) and i > 0:
                             prev_zone_idx = i - 1
                             if prev_zone_idx in hr_zones: lower_bound = hr_zones[prev_zone_idx]
                             else: continue
                             is_in_zone = (df['heart_rate'] > lower_bound)
                        else: continue

                        valid_indices = is_in_zone.iloc[1:].index[is_in_zone.iloc[1:]]
                        time_in_this_zone = time_diff.loc[valid_indices].sum()
                        summary[f'time_in_hr_zone_{i}'] = float(time_in_this_zone) if pd.notna(time_in_this_zone) else 0.0

                    for i in range(MAX_HR_ZONES): logger.debug(f"Time in HR Zone {i}: {summary.get(f'time_in_hr_zone_{i}', 0.0):.1f}s")

            # Calculate Time in Power Zones
            if power_zones and 'power' in df.columns and df['power'].notna().any():
                 if pd.api.types.is_numeric_dtype(df['power']):
                     logger.info("Calculating time in Power zones...")
                     lower_bound = 0
                     for i in range(MAX_PWR_ZONES):
                          is_in_zone = pd.Series([False] * len(df))
                          if i in power_zones:
                              upper_bound = power_zones[i]
                              is_in_zone = (df['power'] > lower_bound) & (df['power'] <= upper_bound)
                              lower_bound = upper_bound
                          elif i == len(power_zones) and i > 0:
                              prev_zone_idx = i - 1
                              if prev_zone_idx in power_zones: lower_bound = power_zones[prev_zone_idx]
                              else: continue
                              is_in_zone = (df['power'] > lower_bound)
                          else: continue

                          valid_indices = is_in_zone.iloc[1:].index[is_in_zone.iloc[1:]]
                          time_in_this_zone = time_diff.loc[valid_indices].sum()
                          summary[f'time_in_pwr_zone_{i}'] = float(time_in_this_zone) if pd.notna(time_in_this_zone) else 0.0

                     for i in range(MAX_PWR_ZONES): logger.debug(f"Time in Power Zone {i}: {summary.get(f'time_in_pwr_zone_{i}', 0.0):.1f}s")

        # --- Final Formatting Pass ---
        # Handle start_time separately as it's now a timezone-aware object
        start_time_obj = summary.get('start_time')
        if isinstance(start_time_obj, pd.Timestamp):
             # Keep it as a Timestamp object for potential future use,
             # formatting will happen in app.py
             pass
        elif start_time_obj is not None: # If it's not None but not a Timestamp (e.g., fallback UTC naive)
             try: # Attempt to convert back just in case
                  summary['start_time'] = pd.Timestamp(start_time_obj)
             except: summary['start_time'] = None # Fallback to None if conversion fails


        for key, val in summary.items():
            if key == 'start_time' or key == 'timezone_str': continue # Skip special fields
            if pd.isna(val):
                summary[key] = None
            else:
                try:
                    f_val = float(val)
                    if key in ['total_time', 'moving_time_seconds'] or 'time_in_' in key:
                        summary[key] = round(f_val, 0)
                    elif key == 'total_calories':
                        summary[key] = int(f_val) if f_val >= 0 else 0
                    else:
                        summary[key] = round(f_val, 1)
                except (ValueError, TypeError):
                    summary[key] = None

        logger.info(f"Parsing successful. TZ={summary['timezone_str']}. Summary: { {k: v for k, v in summary.items() if not k.startswith('time_in')} }")
        # Return df, summary, and the timezone string separately if needed elsewhere, though it's now IN the summary
        return df, summary # Timezone is now part of the summary dict

    # --- Error Handling ---
    except fitparse.FitParseError as e:
        logger.error(f"FIT Parse Error: {e}", exc_info=True)
        return pd.DataFrame(), None
    except Exception as e:
        logger.error(f"Unexpected FIT Parsing Error: {e}", exc_info=True)
        return pd.DataFrame(), None