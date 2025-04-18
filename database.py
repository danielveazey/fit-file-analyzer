# database.py
import sqlite3
import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10) # Increased timeout
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"DB connection error: {e}", exc_info=True)
        return None

def init_db():
    """Initializes the database schema if it doesn't exist."""
    conn = get_db_connection()
    if conn is None: return
    try:
        cursor = conn.cursor()
        cursor.execute(""" PRAGMA journal_mode=WAL; """) # Use WAL mode for better concurrency
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rides (
                id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL, total_time_seconds REAL, total_distance_km REAL,
                avg_speed_kmh REAL, avg_heart_rate REAL, avg_cadence REAL, avg_power REAL,
                total_elevation_gain_m REAL, max_hr REAL, moving_time_seconds REAL,
                min_temp_c REAL, max_temp_c REAL, avg_temp_c REAL,
                max_cadence REAL, max_power REAL, data_path TEXT UNIQUE NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        conn.commit()
        logger.info("DB schema initialized/verified.")
    except sqlite3.Error as e:
        logger.error(f"DB init error: {e}", exc_info=True)
    finally:
        if conn: conn.close()

def add_ride(summary_data, dataframe):
    """Adds ride summary and saves DataFrame to Parquet."""
    conn = get_db_connection()
    if conn is None: logger.error("Add Ride: Failed DB connection."); return None

    filename = summary_data.get('filename'); start_time_val = summary_data.get('start_time')
    if not filename: logger.error("Add Ride: Filename missing."); return None
    start_time_ts = pd.Timestamp(start_time_val)
    if start_time_val is None or pd.isna(start_time_ts): logger.error("Add Ride: Invalid start_time."); return None

    safe_filename = "".join(c if c.isalnum() else "_" for c in filename)
    timestamp_str = start_time_ts.strftime('%Y%m%d_%H%M%S')
    parquet_filename = f"ride_{timestamp_str}_{safe_filename}.parquet"
    data_path = DATA_DIR / parquet_filename

    # Save Parquet
    try:
        if dataframe is not None and not dataframe.empty: dataframe.to_parquet(data_path, index=False); logger.info(f"DF saved: {data_path}")
        else: logger.warning("DF empty, skip save."); return None
    except Exception as e: logger.error(f"Parquet save error for {data_path}: {e}", exc_info=True); return None

    # Prepare SQL Params using explicit map
    param_map = { 'filename': 'filename', 'start_time': 'start_time', 'total_time': 'total_time_seconds', 'total_distance': 'total_distance_km', 'avg_speed': 'avg_speed_kmh', 'avg_heart_rate': 'avg_heart_rate', 'avg_cadence': 'avg_cadence', 'avg_power': 'avg_power', 'total_elevation_gain_m': 'total_elevation_gain_m', 'max_hr': 'max_hr', 'moving_time_seconds': 'moving_time_seconds', 'min_temp_c': 'min_temp_c', 'max_temp_c': 'max_temp_c', 'avg_temp_c': 'avg_temp_c', 'max_cadence': 'max_cadence', 'max_power': 'max_power' }
    params_for_sql = {db_key: summary_data.get(parser_key) for parser_key, db_key in param_map.items()}
    params_for_sql['data_path'] = str(data_path)

    # Sanitize values
    sanitized_params = {}
    numeric_keys = list(param_map.values())[2:] # Exclude filename, start_time
    for key, value in params_for_sql.items():
        if pd.isna(value): sanitized_params[key] = None
        elif key in numeric_keys:
             try: sanitized_params[key] = float(value) if value is not None else None
             except (ValueError, TypeError): logger.warning(f"Could not convert '{key}' val '{value}' to float. Setting None."); sanitized_params[key] = None
        else: sanitized_params[key] = value # Assume correct type (str, datetime)

    # Build and execute SQL
    valid_keys = list(sanitized_params.keys()); columns_str = ', '.join(valid_keys); placeholders_str = ', '.join(':' + k for k in valid_keys)
    sql = f"INSERT INTO rides ({columns_str}) VALUES ({placeholders_str})"
    # logger.debug(f"SQL: {sql}"); logger.debug(f"Params: {sanitized_params}") # Removed for cleanup

    ride_id = None; cursor = None
    try:
        cursor = conn.cursor(); cursor.execute(sql, sanitized_params); conn.commit(); ride_id = cursor.lastrowid;
        if ride_id is not None: logger.info(f"Ride summary added. ID: {ride_id}"); return ride_id
        else: logger.error("DB insert yielded no ride ID."); return None # Should not happen
    except sqlite3.IntegrityError:
        logger.warning(f"IntegrityError saving {data_path}. Removing parquet.")
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Removed {data_path} on IntegrityError.")
            except OSError as e: logger.error(f"Error removing {data_path}: {e}", exc_info=True)
        return None
    except sqlite3.Error as e:
        logger.error(f"DB error adding ride ({type(e).__name__}): {e}", exc_info=True)
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Removed {data_path} on DB Error.")
            except OSError as os_e: logger.error(f"Error removing {data_path}: {os_e}", exc_info=True)
        return None
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


def get_rides():
    """Retrieves a list of all rides from the database."""
    conn = get_db_connection()
    if conn is None: return []
    try: cursor = conn.cursor(); cursor.execute("SELECT id, filename, start_time FROM rides ORDER BY start_time DESC"); rides = cursor.fetchall(); return [dict(row) for row in rides]
    except sqlite3.Error as e: logger.error(f"DB error fetching rides: {e}", exc_info=True); return []
    finally:
        if conn: conn.close()


def get_ride_summary(ride_id):
    """Retrieves summary details for a specific ride ID."""
    conn = get_db_connection()
    if conn is None: return None
    try: cursor = conn.cursor(); cursor.execute("SELECT * FROM rides WHERE id = ?", (ride_id,)); ride_summary = cursor.fetchone(); return dict(ride_summary) if ride_summary else None
    except sqlite3.Error as e: logger.error(f"DB error fetching summary ID {ride_id}: {e}", exc_info=True); return None
    finally:
        if conn: conn.close()


def get_ride_data(ride_id):
    """Loads the ride DataFrame from its Parquet file."""
    conn = get_db_connection(); ride_data = None; data_path_str = None;
    if conn is None: return None
    try: cursor = conn.cursor(); cursor.execute("SELECT data_path FROM rides WHERE id = ?", (ride_id,)); result = cursor.fetchone(); data_path_str = result['data_path'] if result else None;
    except sqlite3.Error as e: logger.error(f"DB error getting path {ride_id}: {e}", exc_info=True); return None
    finally:
        if conn: conn.close()

    if not data_path_str: logger.warning(f"No data path found in DB for ride ID {ride_id}"); return None
    data_path = Path(data_path_str)

    if data_path.exists():
        try: ride_data = pd.read_parquet(data_path); logger.info(f"Loaded DataFrame from {data_path}"); return ride_data
        except Exception as e: logger.error(f"Error reading Parquet {data_path}: {e}", exc_info=True); return None
    else: logger.error(f"Parquet file not found at path from DB: {data_path}"); return None


def delete_ride(ride_id):
    """Deletes a ride from the database and its associated data file."""
    conn = get_db_connection(); data_path_str = None;
    if conn is None: return False
    try: cursor = conn.cursor(); cursor.execute("SELECT data_path FROM rides WHERE id = ?", (ride_id,)); result = cursor.fetchone(); data_path_str = result['data_path'] if result else None;
    except sqlite3.Error as e: logger.error(f"Error getting path delete {ride_id}: {e}", exc_info=True)
    finally:
        if conn: conn.close()

    conn = get_db_connection(); deleted_db = False;
    if conn is None: return False
    try: cursor = conn.cursor(); cursor.execute("DELETE FROM rides WHERE id = ?", (ride_id,)); conn.commit(); deleted_db = cursor.rowcount > 0; logger.info(f"DB delete {ride_id}: {'OK' if deleted_db else 'Not Found'}")
    except sqlite3.Error as e: logger.error(f"Error deleting ride {ride_id}: {e}", exc_info=True)
    finally:
        if conn: conn.close()

    deleted_file = True # Assume success unless we fail to delete an existing file
    if data_path_str:
         data_path = Path(data_path_str)
         if data_path.exists():
             try: os.remove(data_path); logger.info(f"Deleted data file: {data_path}")
             except OSError as e: logger.error(f"Error deleting data file {data_path}: {e}", exc_info=True); deleted_file = False # Failed!
         else: logger.warning(f"Delete: Data file already gone: {data_path}")
    elif deleted_db: logger.warning(f"DB deleted {ride_id}, but data path was unknown.")

    return deleted_db and deleted_file