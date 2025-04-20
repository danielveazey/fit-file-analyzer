# fit_parser.py
import fitparse
import pandas as pd
from datetime import timezone, timedelta
import logging
import numpy as np
import io
import pytz
from timezonefinder import TimezoneFinder

logger = logging.getLogger(__name__)
tf = TimezoneFinder()

# Constants
SEMICIRCLE_TO_DEGREE = 180.0 / (2**31)
MOVING_THRESHOLD_MS = 0.5

# Use 0-based index keys consistent with fitparse message_index
MAX_HR_ZONES = 6
MAX_PWR_ZONES = 7

def parse_fit_file(file_path_or_buffer, user_zones=None):
    """
    Parses FIT file, prioritizing user_zones for calculations if provided.
    Args:
        file_path_or_buffer: Path or buffer for the FIT file.
        user_zones (dict, optional): Dict with 'hr' and 'power' keys,
                                     e.g., {'hr': {0: 90, 1: 120,...}, 'power': {0: 100,...}}.
                                     Keys are 0-based zone indices, values are upper bounds.
                                     Defaults to None.
    Returns:
        tuple: (pandas.DataFrame, dict) -> (DataFrame with all record data, summary dictionary)
               Returns (pd.DataFrame(), None) on failure.
    """
    records_data = []; file_hr_zones = {}; file_power_zones = {}
    file_summary = {}; session_summary = {}; lap_summaries = []
    start_time_utc = None; end_time_utc = None; last_calories = None
    local_timezone_str = None; first_valid_lat = None; first_valid_lon = None

    if user_zones is None: user_zones = {'hr': {}, 'power': {}}

    try:
        fitfile = fitparse.FitFile(file_path_or_buffer)
        all_messages = list(fitfile.get_messages())

        # --- First Pass: Definitions, Summaries, First Coords ---
        logger.info(f"Found {len(all_messages)} messages. Extracting definitions...")
        for message in all_messages:
            msg_type = message.name; msg_dict = message.get_values()
            if msg_type == 'hr_zone':
                zone_idx = msg_dict.get('message_index'); high_bpm = msg_dict.get('high_bpm')
                if zone_idx is not None and high_bpm is not None: file_hr_zones[zone_idx] = high_bpm
            elif msg_type == 'power_zone':
                zone_idx = msg_dict.get('message_index'); high_watts = msg_dict.get('high_watts')
                if zone_idx is not None and high_watts is not None: file_power_zones[zone_idx] = high_watts
            elif msg_type == 'session': session_summary = msg_dict
            elif msg_type == 'lap': lap_summaries.append(msg_dict)
            elif msg_type == 'file_id': file_summary = msg_dict
            elif msg_type == 'record':
                records_data.append(msg_dict)
                if first_valid_lat is None or first_valid_lon is None:
                    lat = msg_dict.get('position_lat'); lon = msg_dict.get('position_long')
                    if lat is not None and lon is not None:
                        try:
                            lat_deg, lon_deg = float(lat) * SEMICIRCLE_TO_DEGREE, float(lon) * SEMICIRCLE_TO_DEGREE
                            if -90 <= lat_deg <= 90 and -180 <= lon_deg <= 180:
                                first_valid_lat, first_valid_lon = lat_deg, lon_deg
                                logger.info(f"First coords for TZ: Lat={lat_deg:.5f}, Lon={lon_deg:.5f}")
                        except (ValueError, TypeError):
                            pass

        file_hr_zones = dict(sorted(file_hr_zones.items()))
        file_power_zones = dict(sorted(file_power_zones.items()))
        logger.info(f"File zones: HR={len(file_hr_zones)}, Power={len(file_power_zones)}")

        # --- Determine Time Zone ---
        if first_valid_lat is not None and first_valid_lon is not None:
            local_timezone_str = tf.timezone_at(lng=first_valid_lon, lat=first_valid_lat) or 'UTC'
            logger.info(f"Determined time zone: {local_timezone_str}")
        else: logger.warning("No valid coords found. Using UTC."); local_timezone_str = 'UTC'

        # --- Process Record Data ---
        if not records_data: logger.warning("No 'record' messages."); return pd.DataFrame(), None
        logger.info(f"Processing {len(records_data)} 'record' messages...")
        processed_records = []
        for record_dict in records_data:
            row = {}; ts_val = record_dict.get('timestamp')
            if ts_val is None: continue
            ts_utc = ts_val.replace(tzinfo=timezone.utc); row['timestamp'] = ts_utc
            if start_time_utc is None: start_time_utc = ts_utc
            end_time_utc = ts_utc # Keep updating

            temp_speed, temp_altitude = None, None; has_enh_spd, has_enh_alt = False, False
            for field, value in record_dict.items():
                if value is None or field == 'timestamp': continue
                if field == 'position_lat': row['latitude_raw'] = value
                elif field == 'position_long': row['longitude_raw'] = value
                elif field == 'enhanced_speed':
                    try: temp_speed = float(value); has_enh_spd = True
                    except (ValueError, TypeError): pass
                elif field == 'speed' and not has_enh_spd:
                    try: temp_speed = float(value)
                    except (ValueError, TypeError): pass
                elif field == 'enhanced_altitude':
                    try: temp_altitude = float(value); has_enh_alt = True
                    except (ValueError, TypeError): pass
                elif field == 'altitude' and not has_enh_alt:
                    try: temp_altitude = float(value)
                    except (ValueError, TypeError): pass
                elif field == 'calories':
                    row['calories'] = value
                    try: last_calories = int(value)
                    except (ValueError, TypeError): pass
                else: row[field] = value
            if temp_speed is not None: row['speed_ms'] = temp_speed
            if temp_altitude is not None: row['altitude'] = temp_altitude
            processed_records.append(row)

        if not processed_records: logger.error("No records processed."); return pd.DataFrame(), None

        # --- DataFrame Creation & Post-Processing ---
        df = pd.DataFrame(processed_records); logger.info(f"Created DF {len(df)}r x {len(df.columns)}c.")
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            df = df.dropna(subset=['timestamp']); df['timestamp'] = df['timestamp'].dt.tz_convert(None)
        except Exception as e: logger.error(f"Timestamp conversion failed: {e}"); return pd.DataFrame(), None
        if df.empty: logger.error("No valid timestamps."); return pd.DataFrame(), None
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        if 'latitude_raw' in df.columns: df['latitude'] = pd.to_numeric(df['latitude_raw'], errors='coerce') * SEMICIRCLE_TO_DEGREE; df.loc[~df['latitude'].between(-90, 90), 'latitude'] = None; df.drop(columns=['latitude_raw'], errors='ignore', inplace=True)
        if 'longitude_raw' in df.columns: df['longitude'] = pd.to_numeric(df['longitude_raw'], errors='coerce') * SEMICIRCLE_TO_DEGREE; df.loc[~df['longitude'].between(-180, 180), 'longitude'] = None; df.drop(columns=['longitude_raw'], errors='ignore', inplace=True)

        common_numeric_cols = ['speed_ms', 'altitude', 'distance', 'heart_rate', 'cadence', 'power','temperature', 'calories', 'grade', 'gps_accuracy'] # etc.
        for col in df.columns:
            if col not in ['timestamp', 'latitude', 'longitude']:
                 is_num = pd.api.types.is_numeric_dtype(df[col]) or col in common_numeric_cols
                 if is_num and not (pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col])):
                      df[col] = pd.to_numeric(df[col], errors='coerce')

        if not df.empty: df['elapsed_time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        else: df['elapsed_time_s'] = pd.NA
        if 'speed_ms' in df.columns and df['speed_ms'].notna().any(): df['speed_kmh'] = df['speed_ms'] * 3.6

        # --- Calculate Summary Statistics ---
        summary = { k: None for k in [ 'start_time', 'total_time', 'total_distance', 'avg_speed', 'avg_heart_rate', 'avg_cadence', 'avg_power', 'max_hr', 'max_cadence', 'max_power', 'min_temp_c', 'max_temp_c', 'avg_temp_c', 'total_elevation_gain_m', 'moving_time_seconds', 'total_calories', 'timezone_str'] }
        summary['timezone_str'] = local_timezone_str
        first_ts_utc_naive = df['timestamp'].iloc[0] if not df.empty else None; last_ts_utc_naive = df['timestamp'].iloc[-1] if not df.empty else None
        if first_ts_utc_naive and last_ts_utc_naive:
            duration = (last_ts_utc_naive - first_ts_utc_naive).total_seconds(); summary['total_time'] = duration if duration >= 0 else None
            try: start_local = pytz.utc.localize(first_ts_utc_naive).astimezone(pytz.timezone(local_timezone_str)); summary['start_time'] = start_local
            except Exception: logger.error(f"TZ convert failed for {local_timezone_str}. Storing naive UTC."); summary['start_time'] = first_ts_utc_naive; summary['timezone_str'] = 'UTC'
        if 'distance' in df.columns and df['distance'].notna().any(): final_dist = df['distance'].dropna().iloc[-1] if not df['distance'].dropna().empty else None; summary['total_distance'] = final_dist / 1000.0 if pd.notna(final_dist) else None
        if 'speed_kmh' in df.columns and df['speed_kmh'].notna().any(): summary['avg_speed'] = df['speed_kmh'].mean(skipna=True)
        elif summary['total_distance'] and summary['total_time'] and summary['total_time'] > 0: summary['avg_speed'] = (summary['total_distance'] / (summary['total_time'] / 3600.0))
        for key in ['heart_rate', 'cadence', 'power']:
            if key in df.columns and df[key].notna().any() and pd.api.types.is_numeric_dtype(df[key]):
                summary[f'avg_{key}'] = df[key].mean(skipna=True); max_key = f'max_{key}' if key != 'heart_rate' else 'max_hr'; summary[max_key] = df[key].max(skipna=True)
        if 'temperature' in df.columns and df['temperature'].notna().any() and pd.api.types.is_numeric_dtype(df['temperature']): summary['min_temp_c']=df['temperature'].min(skipna=True); summary['max_temp_c']=df['temperature'].max(skipna=True); summary['avg_temp_c']=df['temperature'].mean(skipna=True)
        summary['total_elevation_gain_m'] = 0.0
        if 'altitude' in df.columns and df['altitude'].notna().any() and pd.api.types.is_numeric_dtype(df['altitude']): alt_series = df['altitude'].dropna(); gain = alt_series.diff().clip(lower=0).sum() if len(alt_series) > 1 else 0.0; summary['total_elevation_gain_m'] = float(gain) if pd.notna(gain) else 0.0

        # Moving Time Calculation (Corrected Indentation)
        summary['moving_time_seconds'] = 0.0
        if 'speed_ms' in df.columns and df['speed_ms'].notna().any() and pd.api.types.is_numeric_dtype(df['speed_ms']) and len(df) > 1:
            time_diff_mov = df['timestamp'].diff().dt.total_seconds().iloc[1:]
            is_moving = (df['speed_ms'].iloc[1:] > MOVING_THRESHOLD_MS)
            # Check length match before summing
            if len(time_diff_mov) == len(is_moving):
                moving_time = time_diff_mov[is_moving].sum()
                summary['moving_time_seconds'] = float(moving_time) if pd.notna(moving_time) else 0.0
            else:
                logger.error("Moving time calculation failed: length mismatch.")
        # End Moving Time Correction

        summary['total_calories'] = last_calories

        # --- Time in Zone Calculation ---
        hr_zones_to_use = user_zones.get('hr') if user_zones.get('hr') else file_hr_zones
        pwr_zones_to_use = user_zones.get('power') if user_zones.get('power') else file_power_zones
        source_hr = "User" if user_zones.get('hr') else ("File" if file_hr_zones else "None"); source_pwr = "User" if user_zones.get('power') else ("File" if file_power_zones else "None")
        summary['hr_zone_source'] = source_hr; summary['pwr_zone_source'] = source_pwr
        logger.info(f"Using {source_hr} HR zones."); logger.info(f"Using {source_pwr} Power zones.")
        for i in range(MAX_HR_ZONES): summary[f'time_in_hr_zone_{i}'] = 0.0
        for i in range(MAX_PWR_ZONES): summary[f'time_in_pwr_zone_{i}'] = 0.0

        if len(df) > 1:
            time_diff = df['timestamp'].diff().dt.total_seconds()
            if hr_zones_to_use and 'heart_rate' in df.columns and df['heart_rate'].notna().any() and pd.api.types.is_numeric_dtype(df['heart_rate']):
                logger.info(f"Calculating time in {source_hr} HR zones..."); lower_bound = 0; num_defined_hr_zones = len(hr_zones_to_use)
                for i in range(num_defined_hr_zones + 1):
                    is_in_zone = pd.Series([False] * len(df)); zone_idx = i
                    if zone_idx < num_defined_hr_zones: upper_bound = hr_zones_to_use[zone_idx]; is_in_zone = (df['heart_rate'] > lower_bound) & (df['heart_rate'] <= upper_bound); lower_bound = upper_bound
                    elif zone_idx == num_defined_hr_zones: lower_bound = hr_zones_to_use[zone_idx - 1] if zone_idx > 0 else 0; is_in_zone = (df['heart_rate'] > lower_bound)
                    else: continue
                    valid_indices = is_in_zone.iloc[1:].index[is_in_zone.iloc[1:]]; time_in_this_zone = time_diff.loc[valid_indices].sum(); summary_key = f'time_in_hr_zone_{zone_idx}'
                    if summary_key in summary: summary[summary_key] = float(time_in_this_zone) if pd.notna(time_in_this_zone) else 0.0
                for i in range(MAX_HR_ZONES): logger.debug(f"Time in HR Zone {i}: {summary.get(f'time_in_hr_zone_{i}', 0.0):.1f}s")
            if pwr_zones_to_use and 'power' in df.columns and df['power'].notna().any() and pd.api.types.is_numeric_dtype(df['power']):
                 logger.info(f"Calculating time in {source_pwr} Power zones..."); lower_bound = 0; num_defined_pwr_zones = len(pwr_zones_to_use)
                 for i in range(num_defined_pwr_zones + 1):
                      is_in_zone = pd.Series([False] * len(df)); zone_idx = i
                      if zone_idx < num_defined_pwr_zones: upper_bound = pwr_zones_to_use[zone_idx]; is_in_zone = (df['power'] > lower_bound) & (df['power'] <= upper_bound); lower_bound = upper_bound
                      elif zone_idx == num_defined_pwr_zones: lower_bound = pwr_zones_to_use[zone_idx - 1] if zone_idx > 0 else 0; is_in_zone = (df['power'] > lower_bound)
                      else: continue
                      valid_indices = is_in_zone.iloc[1:].index[is_in_zone.iloc[1:]]; time_in_this_zone = time_diff.loc[valid_indices].sum(); summary_key = f'time_in_pwr_zone_{zone_idx}'
                      if summary_key in summary: summary[summary_key] = float(time_in_this_zone) if pd.notna(time_in_this_zone) else 0.0
                 for i in range(MAX_PWR_ZONES): logger.debug(f"Time in Power Zone {i}: {summary.get(f'time_in_pwr_zone_{i}', 0.0):.1f}s")

        # --- Final Formatting ---
        start_time_obj = summary.get('start_time')
        for key, val in summary.items():
            if key in ['start_time', 'timezone_str', 'hr_zone_source', 'pwr_zone_source']: continue
            if pd.isna(val): summary[key] = None
            else:
                try:
                    f_val = float(val);
                    if key in ['total_time', 'moving_time_seconds'] or 'time_in_' in key: summary[key] = round(f_val, 0)
                    elif key == 'total_calories': summary[key] = int(f_val) if f_val >= 0 else 0
                    else: summary[key] = round(f_val, 1)
                except (ValueError, TypeError): summary[key] = None

        logger.info(f"Parsing successful. TZ={summary['timezone_str']}. Summary: { {k: v for k, v in summary.items() if not k.startswith('time_in')} }")
        return df, summary

    except fitparse.FitParseError as e: logger.error(f"FIT Parse Error: {e}"); return pd.DataFrame(), None
    except Exception as e: logger.error(f"Unexpected FIT Parsing Error: {e}", exc_info=True); return pd.DataFrame(), None