# database.py
import sqlite3
import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np
import pytz # Import pytz for timezone handling if needed, though not strictly required here now

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "rides.db"

DB_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10,
                               detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) # Enable type detection
        conn.row_factory = sqlite3.Row
        logger.debug("DB connection established.")
        return conn
    except sqlite3.Error as e:
        logger.error(f"DB connection error: {e}", exc_info=True)
        return None

def init_db():
    """Initializes the database schema, adding timezone_str if it doesn't exist."""
    conn = get_db_connection()
    if conn is None:
        logger.error("DB init failed: No connection.")
        return
    try:
        cursor = conn.cursor()
        cursor.execute(""" PRAGMA journal_mode=WAL; """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL, -- Store as TEXT (ISO format) or INTEGER (Unix timestamp)
                total_time_seconds REAL,
                total_distance_km REAL,
                avg_speed_kmh REAL,
                avg_heart_rate REAL,
                avg_cadence REAL,
                avg_power REAL,
                total_elevation_gain_m REAL,
                max_hr REAL,
                max_cadence REAL,
                max_power REAL,
                moving_time_seconds REAL,
                min_temp_c REAL,
                max_temp_c REAL,
                avg_temp_c REAL,
                total_calories REAL,
                time_in_hr_zone_0 REAL, time_in_hr_zone_1 REAL, time_in_hr_zone_2 REAL,
                time_in_hr_zone_3 REAL, time_in_hr_zone_4 REAL, time_in_hr_zone_5 REAL,
                time_in_pwr_zone_0 REAL, time_in_pwr_zone_1 REAL, time_in_pwr_zone_2 REAL,
                time_in_pwr_zone_3 REAL, time_in_pwr_zone_4 REAL, time_in_pwr_zone_5 REAL,
                time_in_pwr_zone_6 REAL,
                data_path TEXT UNIQUE NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                -- timezone_str column added below if not exists
            )""")

        # --- Add timezone_str column if it doesn't exist ---
        cursor.execute("PRAGMA table_info(rides)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'timezone_str' not in columns:
             logger.info("Adding 'timezone_str' column to rides table.")
             cursor.execute("ALTER TABLE rides ADD COLUMN timezone_str TEXT")
        # ----------------------------------------------------

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rides_start_time ON rides (start_time);")
        conn.commit()
        logger.info("DB schema initialized/verified.")
    except sqlite3.Error as e:
        logger.error(f"DB init error: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.debug("DB connection closed after init.")

def add_ride(summary_data, dataframe):
    """Adds ride summary to DB (including timezone) and saves the full DataFrame."""
    conn = get_db_connection()
    if conn is None: return None

    filename = summary_data.get('filename')
    # start_time is now expected to be a timezone-aware datetime object (or naive UTC as fallback)
    start_time_obj = summary_data.get('start_time')
    timezone_str = summary_data.get('timezone_str', 'UTC') # Get timezone, default to UTC

    if not filename: logger.error("Add Ride: Filename missing."); conn.close(); return None
    if start_time_obj is None or pd.isna(start_time_obj):
        logger.error("Add Ride: Start time missing or invalid."); conn.close(); return None

    # Convert start_time_obj (potentially aware) to a naive UTC timestamp for filename consistency
    try:
        if isinstance(start_time_obj, pd.Timestamp):
             if start_time_obj.tzinfo is not None:
                  start_time_utc_naive = start_time_obj.tz_convert('UTC').tz_localize(None)
             else: # Already naive (assume UTC as per fit_parser fallback)
                  start_time_utc_naive = start_time_obj
        else: # Try converting from string/other
             start_time_utc_naive = pd.Timestamp(start_time_obj).tz_localize('UTC').tz_convert('UTC').tz_localize(None) # Robust conversion attempt

        if pd.isna(start_time_utc_naive): raise ValueError("Timestamp became NaT after UTC conversion.")
        timestamp_str = start_time_utc_naive.strftime('%Y%m%d_%H%M%S')
    except Exception as e:
        logger.error(f"Add Ride: Error converting start time '{start_time_obj}' for filename: {e}")
        conn.close()
        return None

    # --- Create Parquet filename and path ---
    safe_filename_base = "".join(c if c.isalnum() else "_" for c in Path(filename).stem)
    parquet_filename = f"ride_{timestamp_str}_{safe_filename_base}.parquet"
    data_path = DATA_DIR / parquet_filename
    data_path_str = str(data_path)

    # --- Save DataFrame to Parquet ---
    if dataframe is None or dataframe.empty:
        logger.warning("DataFrame empty, cannot save Parquet."); conn.close(); return None
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(data_path, index=False)
        logger.info(f"Full DataFrame saved to Parquet: {data_path}")
    except Exception as e:
        logger.error(f"Parquet save error for {data_path}: {e}", exc_info=True)
        if data_path.exists():
             try: os.remove(data_path); logger.info(f"Removed partial Parquet file {data_path} after save error.")
             except OSError as os_e: logger.error(f"Error removing partial Parquet {data_path}: {os_e}")
        conn.close(); return None

    # --- Prepare summary data for SQL insertion ---
    db_column_map = { # Map summary keys to DB columns
        'filename': 'filename', 'start_time': 'start_time', # start_time needs special handling
        'total_time': 'total_time_seconds', 'total_distance': 'total_distance_km',
        'avg_speed': 'avg_speed_kmh', 'avg_heart_rate': 'avg_heart_rate',
        'avg_cadence': 'avg_cadence', 'avg_power': 'avg_power',
        'total_elevation_gain_m': 'total_elevation_gain_m', 'max_hr': 'max_hr',
        'moving_time_seconds': 'moving_time_seconds', 'min_temp_c': 'min_temp_c',
        'max_temp_c': 'max_temp_c', 'avg_temp_c': 'avg_temp_c',
        'max_cadence': 'max_cadence', 'max_power': 'max_power',
        'total_calories': 'total_calories', 'timezone_str': 'timezone_str', # Add timezone
        'time_in_hr_zone_0': 'time_in_hr_zone_0', 'time_in_hr_zone_1': 'time_in_hr_zone_1',
        'time_in_hr_zone_2': 'time_in_hr_zone_2', 'time_in_hr_zone_3': 'time_in_hr_zone_3',
        'time_in_hr_zone_4': 'time_in_hr_zone_4', 'time_in_hr_zone_5': 'time_in_hr_zone_5',
        'time_in_pwr_zone_0': 'time_in_pwr_zone_0', 'time_in_pwr_zone_1': 'time_in_pwr_zone_1',
        'time_in_pwr_zone_2': 'time_in_pwr_zone_2', 'time_in_pwr_zone_3': 'time_in_pwr_zone_3',
        'time_in_pwr_zone_4': 'time_in_pwr_zone_4', 'time_in_pwr_zone_5': 'time_in_pwr_zone_5',
        'time_in_pwr_zone_6': 'time_in_pwr_zone_6'
    }

    params_for_sql = {}
    numeric_conversion_failures = []

    for summary_key, db_col in db_column_map.items():
        value = summary_data.get(summary_key)
        if pd.isna(value):
            params_for_sql[db_col] = None
        # --- Convert Timestamp to ISO string (UTC) for storage ---
        elif db_col == 'start_time':
            try:
                # Convert the potentially timezone-aware start_time back to naive UTC string
                start_time_for_db = pd.Timestamp(value) # Ensure it's a Timestamp
                if start_time_for_db.tzinfo is not None:
                     start_time_for_db = start_time_for_db.tz_convert('UTC').tz_localize(None)
                params_for_sql[db_col] = start_time_for_db.isoformat(sep=' ', timespec='seconds')
            except Exception as ts_e:
                 logger.warning(f"Could not convert start_time '{value}' to ISO string for DB: {ts_e}. Setting None.")
                 params_for_sql[db_col] = None
        # --- Store timezone string ---
        elif db_col == 'timezone_str':
             params_for_sql[db_col] = str(value) if value else 'UTC' # Store 'UTC' if None
        elif db_col == 'filename':
             params_for_sql[db_col] = str(value) # Ensure filename is string
        else: # Attempt numeric conversion
             try:
                 params_for_sql[db_col] = float(value)
             except (ValueError, TypeError):
                 logger.warning(f"Could not convert '{summary_key}' ('{value}') to float for DB column '{db_col}'. Setting None.")
                 params_for_sql[db_col] = None
                 numeric_conversion_failures.append(summary_key)

    params_for_sql['data_path'] = data_path_str

    # --- Build and execute SQL ---
    valid_db_columns = list(params_for_sql.keys())
    placeholders = ', '.join(':' + k for k in valid_db_columns)
    sql = f"INSERT INTO rides ({', '.join(valid_db_columns)}) VALUES ({placeholders})"

    logger.debug(f"Executing SQL: {sql}")
    # logger.debug(f"With Params for SQL: {params_for_sql}")

    ride_id = None; cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params_for_sql)
        conn.commit()
        ride_id = cursor.lastrowid
        if ride_id is not None:
            logger.info(f"Ride summary added to DB. ID: {ride_id}")
            if numeric_conversion_failures:
                 logger.warning(f"Ride {ride_id} added, but some fields failed numeric conversion: {numeric_conversion_failures}")
            return ride_id
        else:
            logger.error("DB Insert OK but no ride ID returned.")
            if data_path.exists(): os.remove(data_path); logger.info(f"Removed {data_path} (no DB ID).")
            return None
    except sqlite3.IntegrityError as e:
        logger.warning(f"IntegrityError adding ride for {data_path_str}: {e}. Ride likely exists.")
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Removed duplicate Parquet file {data_path_str} on IntegrityError.")
            except OSError as os_e: logger.error(f"Error removing duplicate Parquet {data_path_str}: {os_e}", exc_info=True)
        return None
    except sqlite3.Error as e:
        logger.error(f"DB error adding ride summary: {e} ({type(e).__name__})", exc_info=True)
        logger.error(f"Failed SQL: {sql}")
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Removed Parquet file {data_path_str} on DB Error.")
            except OSError as os_e: logger.error(f"Error removing Parquet {data_path_str}: {os_e}", exc_info=True)
        return None
    finally:
        if cursor: cursor.close()
        if conn: conn.close(); logger.debug("DB connection closed after add_ride.")


def get_rides():
    """Retrieves basic ride info (including timezone if available)."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        cursor = conn.cursor()
        # Select timezone_str as well
        cursor.execute("SELECT id, filename, start_time, timezone_str FROM rides ORDER BY start_time DESC")
        rides = cursor.fetchall()
        return [dict(row) for row in rides] if rides else []
    except sqlite3.Error as e:
        # Handle case where timezone_str column might not exist yet in older DBs
        if "no such column: timezone_str" in str(e):
             logger.warning("timezone_str column not found, fetching without it.")
             cursor.execute("SELECT id, filename, start_time FROM rides ORDER BY start_time DESC")
             rides = cursor.fetchall()
             # Add None for timezone_str manually for consistency upstream
             return [dict(row) | {'timezone_str': None} for row in rides] if rides else []
        else:
             logger.error(f"DB error fetching rides list: {e}", exc_info=True)
             return []
    finally:
        if conn: conn.close(); logger.debug("DB connection closed after get_rides.")


def get_ride_summary(ride_id):
    """Retrieves the summary data (including timezone) for a specific ride ID."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rides WHERE id = ?", (ride_id,))
        ride_summary_row = cursor.fetchone()
        if not ride_summary_row:
             return None

        summary_dict = dict(ride_summary_row)

        # --- Convert stored start_time (should be naive UTC string) back to local ---
        stored_time_str = summary_dict.get('start_time')
        stored_tz_str = summary_dict.get('timezone_str', 'UTC') # Default to UTC if missing
        if stored_time_str:
            try:
                 # Parse the naive UTC string/datetime from DB
                 naive_utc_dt = pd.to_datetime(stored_time_str)
                 # Localize to UTC, then convert to the target timezone
                 utc_dt = pytz.utc.localize(naive_utc_dt)
                 local_tz = pytz.timezone(stored_tz_str)
                 local_dt = utc_dt.astimezone(local_tz)
                 summary_dict['start_time'] = local_dt # Replace naive with aware local time
                 logger.debug(f"Converted DB start time for ride {ride_id} to local: {local_dt}")
            except Exception as e:
                 logger.error(f"Error converting stored start_time '{stored_time_str}' to local time zone '{stored_tz_str}' for ride {ride_id}: {e}")
                 # Fallback: try parsing as is, or leave as string? Let's try parsing directly.
                 try:
                      summary_dict['start_time'] = pd.Timestamp(stored_time_str) # Keep as Timestamp if possible
                 except:
                      summary_dict['start_time'] = None # Or keep original string? Set None for consistency.
        # Ensure timezone_str is in the dictionary, even if None/UTC
        if 'timezone_str' not in summary_dict:
             summary_dict['timezone_str'] = 'UTC'

        return summary_dict

    except sqlite3.Error as e:
        logger.error(f"DB error fetching summary for ride ID {ride_id}: {e}", exc_info=True)
        return None
    finally:
        if conn: conn.close(); logger.debug(f"DB connection closed after get_ride_summary ({ride_id}).")


# get_ride_data remains unchanged - it only fetches the path and reads Parquet
def get_ride_data(ride_id):
    """Retrieves the full DataFrame for a specific ride ID by reading its Parquet file."""
    conn = get_db_connection()
    data_path_str = None
    if conn is None: return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT data_path FROM rides WHERE id = ?", (ride_id,))
        result = cursor.fetchone()
        if result and result['data_path']: data_path_str = result['data_path']
        else: logger.warning(f"No data path found in DB for ride ID {ride_id}."); return None
    except sqlite3.Error as e: logger.error(f"DB error getting data path for ride ID {ride_id}: {e}", exc_info=True); return None
    finally:
        if conn: conn.close(); logger.debug(f"DB connection closed after getting data path ({ride_id}).")

    if data_path_str:
        data_path = Path(data_path_str)
        if data_path.exists():
            try:
                ride_data = pd.read_parquet(data_path)
                logger.info(f"Loaded DataFrame from {data_path}")
                return ride_data
            except Exception as e: logger.error(f"Error reading Parquet file {data_path}: {e}", exc_info=True); return None
        else: logger.error(f"Data path '{data_path}' found in DB but file is missing."); return None
    else: logger.error(f"Data path string was None/empty for ride ID {ride_id}."); return None


# delete_ride remains unchanged - it deletes based on ID and path
def delete_ride(ride_id):
    """Deletes a ride's summary from the DB and its associated Parquet data file."""
    conn = get_db_connection(); data_path_str = None
    if conn is None: logger.error("Delete Ride: Failed DB connection."); return False
    try:
        cursor = conn.cursor(); cursor.execute("SELECT data_path FROM rides WHERE id = ?", (ride_id,)); result = cursor.fetchone()
        if result and result['data_path']: data_path_str = result['data_path']
        else: logger.warning(f"Could not find data_path for ride ID {ride_id} to delete file.")
    except sqlite3.Error as e: logger.error(f"DB error getting data path for deletion (ride ID {ride_id}): {e}", exc_info=True)
    finally:
        if conn: conn.close()

    conn = get_db_connection(); deleted_db_record = False
    if conn is None: logger.error("Delete Ride: Failed DB connection for deletion."); return False
    try:
        cursor = conn.cursor(); cursor.execute("DELETE FROM rides WHERE id = ?", (ride_id,)); conn.commit()
        deleted_db_record = cursor.rowcount > 0
        if deleted_db_record: logger.info(f"Successfully deleted DB record for ride ID: {ride_id}")
        else: logger.warning(f"Ride ID {ride_id} not found in database for deletion.")
    except sqlite3.Error as e: logger.error(f"DB error deleting ride ID {ride_id}: {e}", exc_info=True); deleted_db_record = False
    finally:
        if conn: conn.close(); logger.debug(f"DB connection closed after delete attempt ({ride_id}).")

    deleted_file = False
    if data_path_str:
        data_path = Path(data_path_str)
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Successfully deleted data file: {data_path}"); deleted_file = True
            except OSError as e: logger.error(f"Error deleting data file {data_path}: {e}", exc_info=True); deleted_file = False
        else: logger.warning(f"File not found for deleted ride ID {ride_id}: {data_path}"); deleted_file = True
    elif deleted_db_record: logger.warning(f"DB record deleted for ride ID {ride_id}, but data_path unknown."); deleted_file = False

    if data_path_str: return deleted_db_record and deleted_file
    else: return deleted_db_record