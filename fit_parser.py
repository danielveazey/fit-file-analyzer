# fit_parser.py
import fitparse
import pandas as pd
from datetime import timezone
import logging
import numpy as np # Import numpy for isnan check if needed

logger = logging.getLogger(__name__)

# Constants for coordinate conversion
SEMICIRCLE_TO_DEGREE = 180.0 / (2**31)

def parse_fit_file(file_path_or_buffer):
    """
    Parses a FIT file and returns a Pandas DataFrame and summary statistics.

    Args:
        file_path_or_buffer: Path to the .fit file or a file-like object.

    Returns:
        A tuple containing:
        - pd.DataFrame: DataFrame with ride data.
        - dict: Summary statistics including total_elevation_gain_m.
    """
    data = []
    summary = {}
    start_time = None
    end_time = None
    total_distance = 0.0 # Meters initially

    required_fields = {'timestamp'}
    optional_fields = {
        'position_lat', 'position_long', 'distance', 'enhanced_speed', 'speed',
        'enhanced_altitude', 'altitude', 'heart_rate', 'cadence', 'power',
        'temperature' # Add if you want temperature
    }
    available_fields = set()

    try:
        fitfile = fitparse.FitFile(file_path_or_buffer)
        messages = list(fitfile.get_messages('record'))

        if not messages:
             logger.warning("No 'record' messages found.")
             return pd.DataFrame(), None

        logger.info(f"Found {len(messages)} 'record' messages.")

        for msg in messages[:15]: # Check first 15 records for fields
             available_fields.update(f.name for f in msg if hasattr(f, 'name'))
        logger.info(f"Available fields detected: {available_fields}")

        fields_to_extract = required_fields.union(optional_fields.intersection(available_fields))
        logger.info(f"Fields to extract: {fields_to_extract}")

        for record in messages:
            row = {}
            if record.mesg_type and record.mesg_type.name == 'record':
                record_data = record.get_values()
                if 'timestamp' not in record_data or record_data['timestamp'] is None:
                    continue
                row['timestamp'] = record_data['timestamp'].replace(tzinfo=timezone.utc)
                if start_time is None: start_time = row['timestamp']
                end_time = row['timestamp']

                # --- Safely extract coords (Corrected Multi-line Try/Except) ---
                row['latitude'] = None
                if 'position_lat' in fields_to_extract and 'position_lat' in record_data:
                    lat_val = record_data['position_lat']
                    if lat_val is not None:
                        try:
                            lat = float(lat_val) * SEMICIRCLE_TO_DEGREE
                            row['latitude'] = lat if -90.0 <= lat <= 90.0 else None
                        except (TypeError, ValueError):
                            logger.debug(f"Non-numeric latitude ignored: {lat_val}")
                            pass # Keep row['latitude'] as None

                row['longitude'] = None
                if 'position_long' in fields_to_extract and 'position_long' in record_data:
                    lon_val = record_data['position_long']
                    if lon_val is not None:
                         try:
                             lon = float(lon_val) * SEMICIRCLE_TO_DEGREE
                             row['longitude'] = lon if -181.0 <= lon <= 181.0 else None # Allow wrap
                         except (TypeError, ValueError):
                             logger.debug(f"Non-numeric longitude ignored: {lon_val}")
                             pass # Keep row['longitude'] as None


                # --- Extract other fields safely (Corrected Multi-line Try/Except) ---
                speed_val = record_data.get('enhanced_speed') if 'enhanced_speed' in fields_to_extract else record_data.get('speed')
                row['speed_ms'] = None # Default
                if speed_val is not None:
                    try:
                        row['speed_ms'] = float(speed_val)
                    except (TypeError, ValueError):
                         logger.debug(f"Non-numeric speed ignored: {speed_val}")
                         pass # Keep row['speed_ms'] as None

                alt_val = record_data.get('enhanced_altitude') if 'enhanced_altitude' in fields_to_extract else record_data.get('altitude')
                row['altitude'] = None # Default
                if alt_val is not None:
                     try:
                         row['altitude'] = float(alt_val)
                     except (TypeError, ValueError):
                         logger.debug(f"Non-numeric altitude ignored: {alt_val}")
                         pass # Keep row['altitude'] as None

                dist_val = record_data.get('distance')
                row['distance'] = None # Default
                if dist_val is not None:
                    try:
                         row['distance'] = float(dist_val)
                         # Update total distance based on valid numeric distance points processed so far
                         total_distance = max(total_distance, row['distance'])
                    except (TypeError, ValueError):
                        logger.debug(f"Non-numeric distance ignored: {dist_val}")
                        pass # Keep row['distance'] as None


                for field in ['heart_rate', 'cadence', 'power', 'temperature']:
                    val = record_data.get(field)
                    row[field] = None # Default
                    if field in fields_to_extract and val is not None:
                        try:
                            # Adjust type conversion based on field
                            if field in ['heart_rate', 'cadence', 'temperature']:
                                row[field] = int(val) # Usually integers
                            elif field == 'power':
                                row[field] = float(val) # Can be float
                            else:
                                row[field] = float(val) # Default fallback
                        except (TypeError, ValueError):
                            logger.debug(f"Non-numeric {field} ignored: {val}")
                            pass # Keep row[field] as None


                data.append(row)

        if not data:
            logger.warning("No valid records extracted.")
            return pd.DataFrame(), None

        df = pd.DataFrame(data)

        # --- Post-processing ---
        if 'timestamp' in df.columns:
             try: df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.tz_convert(None); df = df.dropna(subset=['timestamp'])
             except Exception: logger.error("Failed to convert timestamp column."); return pd.DataFrame(), None
        else: logger.error("Timestamp column missing after processing."); return pd.DataFrame(), None

        numeric_cols = ['latitude', 'longitude', 'speed_ms', 'altitude', 'distance', 'heart_rate', 'cadence', 'power', 'temperature']
        for col in numeric_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

        if not df.empty and 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
             df = df.sort_values(by='timestamp').reset_index(drop=True)
             df['elapsed_time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        else: df['elapsed_time_s'] = pd.NA

        if 'speed_ms' in df.columns: df['speed_kmh'] = df['speed_ms'] * 3.6


        # --- Calculate Summary Statistics ---
        if start_time and end_time:
            summary['start_time'] = start_time.replace(tzinfo=None)
            summary['total_time'] = (end_time - start_time).total_seconds()

            final_distance = df['distance'].dropna().iloc[-1] if 'distance' in df.columns and not df['distance'].dropna().empty else None
            summary['total_distance'] = final_distance / 1000.0 if final_distance is not None else None

            summary['avg_speed'] = df['speed_kmh'].mean(skipna=True) if 'speed_kmh' in df.columns else None
            summary['avg_heart_rate'] = df['heart_rate'].mean(skipna=True) if 'heart_rate' in df.columns else None
            summary['avg_cadence'] = df['cadence'].mean(skipna=True) if 'cadence' in df.columns else None
            summary['avg_power'] = df['power'].mean(skipna=True) if 'power' in df.columns else None

            # Calculate Total Elevation Gain (in Meters)
            summary['total_elevation_gain_m'] = None
            if 'altitude' in df.columns and df['altitude'].notna().any():
                altitude_series = df['altitude'].dropna()
                if len(altitude_series) > 1:
                    elevation_diffs = altitude_series.diff()
                    total_gain = elevation_diffs[elevation_diffs > 0].sum()
                    summary['total_elevation_gain_m'] = float(total_gain) if pd.notna(total_gain) else 0.0
                else: # Only one valid altitude point
                    summary['total_elevation_gain_m'] = 0.0
            else: # Altitude column missing or all NaN
                 summary['total_elevation_gain_m'] = 0.0 # Assign 0 if no altitude data? Or None? Let's use 0.0


            logger.info(f"Calculated Elevation Gain: {summary['total_elevation_gain_m']} m")

            # Format numeric summaries
            numeric_summary_keys = ['avg_speed', 'avg_heart_rate', 'avg_cadence', 'avg_power', 'total_elevation_gain_m']
            for key in numeric_summary_keys:
                 val = summary.get(key)
                 if val is not None and pd.notna(val):
                     # Round gain to 1 decimal place (meters)
                     summary[key] = round(val, 1)
                 else:
                     summary[key] = None

            logger.info(f"Parsing successful. Summary: {summary}")

        else:
             logger.warning("Could not determine start or end time.")
             summary = None

        logger.info(f"DataFrame created with columns: {df.columns.tolist()}")
        logger.info(f"DataFrame shape: {df.shape}")

        return df, summary

    except fitparse.FitParseError as e:
        logger.error(f"Error parsing FIT file: {e}")
        return pd.DataFrame(), None
    except Exception as e:
        logger.error(f"An unexpected error occurred during FIT parsing: {e}", exc_info=True)
        return pd.DataFrame(), None