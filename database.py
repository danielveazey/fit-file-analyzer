# database.py
import sqlite3
import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import pytz

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "rides.db"

DB_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# --- Constants for Zone storage ---
MAX_HR_ZONES_USER = 6
MAX_PWR_ZONES_USER = 7

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10,
                               detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        conn.row_factory = sqlite3.Row
        logger.debug("DB connection established.")
        return conn
    except sqlite3.Error as e:
        logger.error(f"DB connection error: {e}", exc_info=True)
        return None

def init_db():
    """Initializes the database schema, adding timezone_str and user_profile if they don't exist."""
    conn = get_db_connection()
    if conn is None: logger.error("DB init failed: No connection."); return
    try:
        cursor = conn.cursor()
        cursor.execute(""" PRAGMA journal_mode=WAL; """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rides (
                id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL, total_time_seconds REAL, total_distance_km REAL,
                avg_speed_kmh REAL, avg_heart_rate REAL, avg_cadence REAL, avg_power REAL,
                total_elevation_gain_m REAL, max_hr REAL, max_cadence REAL, max_power REAL,
                moving_time_seconds REAL, min_temp_c REAL, max_temp_c REAL, avg_temp_c REAL,
                total_calories REAL,
                time_in_hr_zone_0 REAL, time_in_hr_zone_1 REAL, time_in_hr_zone_2 REAL,
                time_in_hr_zone_3 REAL, time_in_hr_zone_4 REAL, time_in_hr_zone_5 REAL,
                time_in_pwr_zone_0 REAL, time_in_pwr_zone_1 REAL, time_in_pwr_zone_2 REAL,
                time_in_pwr_zone_3 REAL, time_in_pwr_zone_4 REAL, time_in_pwr_zone_5 REAL,
                time_in_pwr_zone_6 REAL,
                data_path TEXT UNIQUE NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                timezone_str TEXT
            )""")
        try: cursor.execute("ALTER TABLE rides ADD COLUMN timezone_str TEXT")
        except sqlite3.OperationalError: pass

        hr_zone_cols = ", ".join([f"hr_zone_{i}_upper INTEGER" for i in range(1, MAX_HR_ZONES_USER)])
        pwr_zone_cols = ", ".join([f"pwr_zone_{i}_upper INTEGER" for i in range(1, MAX_PWR_ZONES_USER)])
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS user_profile (
                id INTEGER PRIMARY KEY, {hr_zone_cols}, {pwr_zone_cols}
            )""")
        cursor.execute("INSERT OR IGNORE INTO user_profile (id) VALUES (1)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rides_start_time ON rides (start_time);")
        conn.commit()
        logger.info("DB schema initialized/verified (including user_profile).")
    except sqlite3.Error as e:
        logger.error(f"DB init error: {e}", exc_info=True)
    finally:
        if conn: conn.close(); logger.debug("DB connection closed after init.")

# --- User Profile Functions ---
def get_user_zones():
    """Retrieves user-defined zones (0-based keys)."""
    conn = get_db_connection()
    if conn is None: return {'hr': {}, 'power': {}}
    user_zones = {'hr': {}, 'power': {}}
    try:
        cursor = conn.cursor(); cursor.execute("SELECT * FROM user_profile WHERE id = 1")
        profile = cursor.fetchone()
        if profile:
            profile_dict = dict(profile)
            for i in range(1, MAX_HR_ZONES_USER):
                db_col = f"hr_zone_{i}_upper"; val = profile_dict.get(db_col)
                if val is not None: user_zones['hr'][i - 1] = int(val)
            for i in range(1, MAX_PWR_ZONES_USER):
                db_col = f"pwr_zone_{i}_upper"; val = profile_dict.get(db_col)
                if val is not None: user_zones['power'][i - 1] = int(val)
            logger.info(f"Loaded user zones: HR={len(user_zones['hr'])}, Power={len(user_zones['power'])}")
        else: logger.warning("No user profile found (id=1).")
    except sqlite3.Error as e: logger.error(f"DB error getting user zones: {e}", exc_info=True)
    finally:
        if conn: conn.close()
    return user_zones

def save_user_zones(hr_zones_dict, pwr_zones_dict):
    """Saves user zones (expects 1-based keys from UI)."""
    conn = get_db_connection()
    if conn is None: return False
    update_params = {'id': 1}; set_clauses = []
    for i in range(1, MAX_HR_ZONES_USER):
        db_col = f"hr_zone_{i}_upper"; param_name = f"hr{i}"; value = hr_zones_dict.get(i)
        try: update_params[param_name] = int(value) if value is not None else None
        except (ValueError, TypeError): update_params[param_name] = None
        set_clauses.append(f"{db_col} = :{param_name}")
    for i in range(1, MAX_PWR_ZONES_USER):
        db_col = f"pwr_zone_{i}_upper"; param_name = f"pwr{i}"; value = pwr_zones_dict.get(i)
        try: update_params[param_name] = int(value) if value is not None else None
        except (ValueError, TypeError): update_params[param_name] = None
        set_clauses.append(f"{db_col} = :{param_name}")

    sql = f"UPDATE user_profile SET {', '.join(set_clauses)} WHERE id = :id"
    logger.debug(f"Executing SQL: {sql}")
    try:
        cursor = conn.cursor(); cursor.execute(sql, update_params); conn.commit()
        logger.info(f"User zones saved. Rows affected: {cursor.rowcount}"); return True
    except sqlite3.Error as e: logger.error(f"DB error saving user zones: {e}", exc_info=True); return False
    finally:
        if conn: conn.close()

# --- Ride Functions ---
def add_ride(summary_data, dataframe):
    conn = get_db_connection()
    if conn is None: logger.error("Add Ride: Failed DB connection."); return None
    filename = summary_data.get('filename'); start_time_obj = summary_data.get('start_time')
    if not filename: logger.error("Add Ride: Filename missing."); conn.close(); return None
    if start_time_obj is None or pd.isna(start_time_obj): logger.error("Add Ride: Start time missing/invalid."); conn.close(); return None
    try:
        if isinstance(start_time_obj, pd.Timestamp):
             start_time_utc_naive = start_time_obj.tz_convert('UTC').tz_localize(None) if start_time_obj.tzinfo else start_time_obj
        else: start_time_utc_naive = pd.Timestamp(start_time_obj).tz_localize('UTC').tz_convert('UTC').tz_localize(None)
        if pd.isna(start_time_utc_naive): raise ValueError("Timestamp NaT after UTC conversion.")
        timestamp_str = start_time_utc_naive.strftime('%Y%m%d_%H%M%S')
    except Exception as e: logger.error(f"Add Ride: Error converting start time '{start_time_obj}' for filename: {e}"); conn.close(); return None

    safe_filename_base = "".join(c if c.isalnum() else "_" for c in Path(filename).stem); parquet_filename = f"ride_{timestamp_str}_{safe_filename_base}.parquet"; data_path = DATA_DIR / parquet_filename; data_path_str = str(data_path)
    if dataframe is None or dataframe.empty: logger.warning("DataFrame empty."); conn.close(); return None
    try: DATA_DIR.mkdir(parents=True, exist_ok=True); dataframe.to_parquet(data_path, index=False); logger.info(f"DF saved: {data_path}")
    except Exception as e:
        logger.error(f"Parquet save error {data_path}: {e}", exc_info=True)
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Removed partial Parquet {data_path}.")
            except OSError as os_e: logger.error(f"Error removing partial Parquet {data_path}: {os_e}")
        conn.close(); return None

    db_column_map = { # Simplified map + dynamic zone keys
        'filename': 'filename', 'start_time': 'start_time', 'timezone_str': 'timezone_str',
        'total_time': 'total_time_seconds', 'total_distance': 'total_distance_km',
        'avg_speed': 'avg_speed_kmh', 'avg_heart_rate': 'avg_heart_rate',
        'avg_cadence': 'avg_cadence', 'avg_power': 'avg_power',
        'total_elevation_gain_m': 'total_elevation_gain_m', 'max_hr': 'max_hr',
        'max_cadence': 'max_cadence', 'max_power': 'max_power',
        'moving_time_seconds': 'moving_time_seconds', 'min_temp_c': 'min_temp_c',
        'max_temp_c': 'max_temp_c', 'avg_temp_c': 'avg_temp_c',
        'total_calories': 'total_calories'
    }
    # Add zone keys dynamically (Assuming MAX_HR_ZONES and MAX_PWR_ZONES are defined)
    for i in range(MAX_HR_ZONES_USER): db_column_map[f'time_in_hr_zone_{i}'] = f'time_in_hr_zone_{i}'
    for i in range(MAX_PWR_ZONES_USER): db_column_map[f'time_in_pwr_zone_{i}'] = f'time_in_pwr_zone_{i}'


    params_for_sql = {}; numeric_conversion_failures = []
    for summary_key, db_col in db_column_map.items():
        value = summary_data.get(summary_key)
        if pd.isna(value): params_for_sql[db_col] = None
        elif db_col == 'start_time':
            try:
                start_time_for_db = pd.Timestamp(value);
                if start_time_for_db.tzinfo is not None: start_time_for_db = start_time_for_db.tz_convert('UTC').tz_localize(None)
                params_for_sql[db_col] = start_time_for_db.isoformat(sep=' ', timespec='seconds')
            except Exception as ts_e: logger.warning(f"DB Convert start_time '{value}' failed: {ts_e}."); params_for_sql[db_col] = None
        elif db_col in ['filename', 'timezone_str']: params_for_sql[db_col] = str(value) if value else ('UTC' if db_col == 'timezone_str' else None)
        else:
            try: params_for_sql[db_col] = float(value)
            except (ValueError, TypeError): logger.warning(f"DB Convert '{summary_key}' ('{value}') to float failed."); params_for_sql[db_col] = None; numeric_conversion_failures.append(summary_key)
    params_for_sql['data_path'] = data_path_str

    # Ensure only valid columns are included in the INSERT statement
    # Get columns currently in the rides table
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(rides)")
        existing_db_cols = {row['name'] for row in cursor.fetchall()}
    except sqlite3.Error as e:
        logger.error(f"Failed to get table info for rides: {e}"); conn.close(); return None
    finally:
        if cursor: cursor.close()

    # Filter params_for_sql to include only existing columns (+ data_path)
    filtered_params = {k: v for k, v in params_for_sql.items() if k in existing_db_cols}
    if 'data_path' not in filtered_params and 'data_path' in params_for_sql: # Ensure data_path is always included if calculated
        filtered_params['data_path'] = params_for_sql['data_path']

    valid_db_columns = list(filtered_params.keys()) # Use keys from filtered dict
    placeholders = ', '.join(':' + k for k in valid_db_columns)
    sql = f"INSERT INTO rides ({', '.join(valid_db_columns)}) VALUES ({placeholders})"


    logger.debug(f"Executing SQL: {sql}")
    ride_id = None; cursor = None
    try:
        cursor = conn.cursor(); cursor.execute(sql, filtered_params); conn.commit(); ride_id = cursor.lastrowid # Use filtered_params
        if ride_id is not None: logger.info(f"Ride summary added. ID: {ride_id}"); return ride_id
        else:
            logger.error("DB Insert OK but no ride ID.")
            if data_path.exists():
                try: os.remove(data_path); logger.info(f"Removed {data_path} (no DB ID).")
                except OSError as os_e: logger.error(f"Error removing {data_path}: {os_e}")
            return None
    except sqlite3.IntegrityError as e:
        logger.warning(f"IntegrityError {data_path_str}: {e}. Likely duplicate.")
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Removed duplicate {data_path_str}.")
            except OSError as os_e: logger.error(f"Error removing {data_path_str}: {os_e}", exc_info=True)
        return None
    except sqlite3.Error as e:
        logger.error(f"DB error adding summary: {e}", exc_info=True); logger.error(f"Failed SQL: {sql}")
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Removed {data_path_str} on DB Error.")
            except OSError as os_e: logger.error(f"Error removing {data_path_str}: {os_e}", exc_info=True)
        return None
    finally:
        if cursor: cursor.close()
        if conn: conn.close(); logger.debug("DB connection closed after add_ride.")

def get_rides():
    conn = get_db_connection()
    if conn is None: return []
    try:
        cursor = conn.cursor()
        # Ensure timezone_str is selected
        cursor.execute("SELECT id, filename, start_time, timezone_str FROM rides ORDER BY start_time DESC")
        rides = cursor.fetchall()
        # Convert rows to dicts, handle potential missing timezone_str gracefully
        results = []
        for row in rides:
            row_dict = dict(row)
            if 'timezone_str' not in row_dict:
                row_dict['timezone_str'] = None # Add None if column was missing
            results.append(row_dict)
        return results
    except sqlite3.Error as e:
        # Fallback query if timezone_str column does not exist
        if "no such column: timezone_str" in str(e):
             logger.warning("timezone_str column missing, fetching without it.");
             cursor.execute("SELECT id, filename, start_time FROM rides ORDER BY start_time DESC")
             rides = cursor.fetchall()
             # Add 'timezone_str': None manually to each dict
             return [dict(row) | {'timezone_str': None} for row in rides] if rides else []
        else:
             logger.error(f"DB error fetching rides list: {e}", exc_info=True)
             return []
    finally:
        if conn: conn.close()


def get_ride_summary(ride_id):
    conn = get_db_connection()
    if conn is None: return None
    try:
        cursor = conn.cursor()
        # Use PRAGMA to check if column exists before querying
        cursor.execute("PRAGMA table_info(rides)")
        columns = [info['name'] for info in cursor.fetchall()]
        select_cols = "*"
        if 'timezone_str' not in columns:
            # Build select list excluding timezone_str and add NULL alias
            cursor.execute("PRAGMA table_info(rides)") # Re-fetch without altering the cache
            cols_without_tz = [f"'{col['name']}'" for col in cursor.fetchall()]
            select_cols = ", ".join(cols_without_tz) + ", NULL as timezone_str"
            logger.warning("timezone_str column missing in DB, fetching summary with NULL placeholder.")

        cursor.execute(f"SELECT {select_cols} FROM rides WHERE id = ?", (ride_id,))
        ride_summary_row = cursor.fetchone()

        if not ride_summary_row: return None
        summary_dict = dict(ride_summary_row)
        stored_time_str = summary_dict.get('start_time')
        stored_tz_str = summary_dict.get('timezone_str') or 'UTC' # Use fetched or default to UTC

        if stored_time_str:
            try:
                 naive_utc_dt = pd.to_datetime(stored_time_str)
                 utc_dt = pytz.utc.localize(naive_utc_dt)
                 local_tz = pytz.timezone(stored_tz_str)
                 local_dt = utc_dt.astimezone(local_tz)
                 summary_dict['start_time'] = local_dt
                 logger.debug(f"Converted DB start time ride {ride_id} to local: {local_dt}")
            except Exception as e:
                 logger.error(f"Error converting start_time '{stored_time_str}' to TZ '{stored_tz_str}': {e}")
                 try: summary_dict['start_time'] = pd.Timestamp(stored_time_str)
                 except Exception: summary_dict['start_time'] = None
        # Ensure timezone_str is in dict even if NULL from DB
        if 'timezone_str' not in summary_dict or summary_dict['timezone_str'] is None:
            summary_dict['timezone_str'] = 'UTC'

        return summary_dict
    except sqlite3.Error as e:
        logger.error(f"DB error fetching summary {ride_id}: {e}", exc_info=True)
        return None
    finally:
        if conn: conn.close()


# --- CORRECTED get_ride_data ---
def get_ride_data(ride_id):
    """Retrieves the full DataFrame for a specific ride ID by reading its Parquet file."""
    conn = get_db_connection()
    data_path_str = None
    if conn is None: return None

    # 1. Get the data path from the database
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT data_path FROM rides WHERE id = ?", (ride_id,))
        result = cursor.fetchone()
        if result and result['data_path']:
            data_path_str = result['data_path']
        else:
            logger.warning(f"No data path found in DB for ride ID {ride_id}.")
            # No need to close connection here, finally block handles it
            return None
    except sqlite3.Error as e:
        logger.error(f"DB error getting data path for ride ID {ride_id}: {e}", exc_info=True)
        # No need to close connection here, finally block handles it
        return None
    finally:
        if conn:
            conn.close()
            logger.debug(f"DB connection closed after getting data path ({ride_id}).")

    # 2. Check if the path exists and load the Parquet file
    if data_path_str:
        data_path = Path(data_path_str)
        if data_path.exists():
            try:
                ride_data = pd.read_parquet(data_path)
                logger.info(f"Loaded DataFrame from {data_path}")
                return ride_data
            except Exception as e:
                logger.error(f"Error reading Parquet file {data_path}: {e}", exc_info=True)
                return None
        else:
            logger.error(f"Data path '{data_path}' found in DB but file is missing.")
            return None
    else:
        logger.error(f"Data path string was None or empty for ride ID {ride_id}.")
        return None
# --- END OF CORRECTED get_ride_data ---


def delete_ride(ride_id):
    data_path_str = None # Initialize data_path_str
    conn = get_db_connection()
    if conn is None: logger.error("Delete: Failed DB connection."); return False
    # 1. Get the data path *before* deleting the DB record
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT data_path FROM rides WHERE id = ?", (ride_id,))
        result = cursor.fetchone()
        if result and result['data_path']:
            data_path_str = result['data_path']
        else:
            logger.warning(f"Could not find data_path for ride ID {ride_id} to delete file.")
    except sqlite3.Error as e:
        logger.error(f"Delete: Error getting path {ride_id}: {e}", exc_info=True)
        # Continue to attempt DB deletion even if path fetch fails
    finally:
        if conn: conn.close() # Close connection after fetching path

    # 2. Delete the record from the database
    conn = get_db_connection() # Re-establish connection
    deleted_db = False
    if conn is None: logger.error("Delete: Failed DB connection for deletion."); return False
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rides WHERE id = ?", (ride_id,))
        conn.commit()
        deleted_db = cursor.rowcount > 0
        if deleted_db: logger.info(f"DB deleted {ride_id}: OK")
        else: logger.warning(f"DB delete {ride_id}: Not Found")
    except sqlite3.Error as e:
        logger.error(f"DB error deleting {ride_id}: {e}", exc_info=True)
        deleted_db = False # Ensure deletion status is False on error
    finally:
        if conn: conn.close(); logger.debug(f"DB connection closed after delete attempt ({ride_id}).")

    # 3. Delete the Parquet file if the path was found and DB record was deleted
    deleted_file = False
    if data_path_str:
         data_path = Path(data_path_str)
         if data_path.exists():
             try: os.remove(data_path); logger.info(f"Deleted file: {data_path}"); deleted_file = True
             except OSError as e: logger.error(f"Error deleting file {data_path}: {e}", exc_info=True); deleted_file = False
         else: logger.warning(f"Delete: File not found: {data_path}"); deleted_file = True # Consider success if already gone
    elif deleted_db:
        logger.warning(f"DB deleted {ride_id}, but path unknown.")
        deleted_file = False # Cannot confirm file deletion

    # Determine overall success
    if data_path_str: return deleted_db and deleted_file
    else: return deleted_db # If path was unknown, success depends only on DB