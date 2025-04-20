# database.py
import sqlite3
import os
import pandas as pd
from pathlib import Path
import logging
import numpy as np

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Already configured in app.py
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
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        logger.debug("DB connection established.")
        return conn
    except sqlite3.Error as e:
        logger.error(f"DB connection error: {e}", exc_info=True)
        return None

def init_db():
    """Initializes the database schema if it doesn't exist."""
    conn = get_db_connection()
    if conn is None:
        logger.error("DB init failed: No connection.")
        return
    try:
        cursor = conn.cursor()
        # Enable Write-Ahead Logging for better concurrency
        cursor.execute(""" PRAGMA journal_mode=WAL; """)
        # Create rides table - schema remains the same, storing summaries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rides (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
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
                data_path TEXT UNIQUE NOT NULL, -- Path to the detailed Parquet file
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        # Add an index on start_time for potentially faster sorting/lookup
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
    """Adds ride summary to DB and saves the full DataFrame to Parquet."""
    conn = get_db_connection()
    if conn is None:
        logger.error("Add Ride: Failed DB connection.")
        return None

    filename = summary_data.get('filename')
    start_time_val = summary_data.get('start_time') # This should be timezone-naive from fit_parser

    if not filename:
        logger.error("Add Ride: Filename missing in summary_data.")
        if conn: conn.close()
        return None
    if start_time_val is None:
        logger.error("Add Ride: Start time missing or invalid in summary_data.")
        if conn: conn.close()
        return None

    # Ensure start_time is a pandas Timestamp for formatting
    try:
        start_time_ts = pd.Timestamp(start_time_val)
        if pd.isna(start_time_ts):
            raise ValueError("Start time is NaT")
    except Exception as e:
        logger.error(f"Add Ride: Invalid start_time format '{start_time_val}': {e}")
        if conn: conn.close()
        return None

    # --- Create a safe filename for the Parquet file ---
    safe_filename_base = "".join(c if c.isalnum() else "_" for c in Path(filename).stem)
    timestamp_str = start_time_ts.strftime('%Y%m%d_%H%M%S')
    parquet_filename = f"ride_{timestamp_str}_{safe_filename_base}.parquet"
    data_path = DATA_DIR / parquet_filename
    data_path_str = str(data_path) # Use string representation for DB and checks

    # --- Save the FULL DataFrame to Parquet ---
    if dataframe is None or dataframe.empty:
        logger.warning("DataFrame is empty or None, cannot save Parquet or add ride.")
        if conn: conn.close()
        return None

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(data_path, index=False)
        logger.info(f"Full DataFrame saved to Parquet: {data_path}")
    except Exception as e:
        logger.error(f"Parquet save error for {data_path}: {e}", exc_info=True)
        if data_path.exists():
             try: os.remove(data_path); logger.info(f"Removed partial Parquet file {data_path} after save error.")
             except OSError as os_e: logger.error(f"Error removing partial Parquet {data_path}: {os_e}")
        if conn: conn.close()
        return None

    # --- Prepare summary data for SQL insertion ---
    db_column_map = {
        'filename': 'filename', 'start_time': 'start_time',
        'total_time': 'total_time_seconds', 'total_distance': 'total_distance_km',
        'avg_speed': 'avg_speed_kmh', 'avg_heart_rate': 'avg_heart_rate',
        'avg_cadence': 'avg_cadence', 'avg_power': 'avg_power',
        'total_elevation_gain_m': 'total_elevation_gain_m', 'max_hr': 'max_hr',
        'moving_time_seconds': 'moving_time_seconds', 'min_temp_c': 'min_temp_c',
        'max_temp_c': 'max_temp_c', 'avg_temp_c': 'avg_temp_c',
        'max_cadence': 'max_cadence', 'max_power': 'max_power',
        'total_calories': 'total_calories',
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
        # *** FIX: Convert Timestamp to ISO string for SQLite ***
        elif db_col == 'start_time':
            try:
                # Ensure it's a pandas Timestamp before formatting
                ts_value = pd.Timestamp(value)
                if pd.notna(ts_value):
                     params_for_sql[db_col] = ts_value.isoformat(sep=' ', timespec='seconds') # Use space separator, seconds precision
                else:
                     params_for_sql[db_col] = None
            except Exception as ts_e:
                 logger.warning(f"Could not convert start_time '{value}' to ISO string: {ts_e}. Setting None.")
                 params_for_sql[db_col] = None
        # *** END FIX ***
        elif db_col == 'filename': # Keep filename as string
             params_for_sql[db_col] = str(value) # Ensure it's a string
        else: # Attempt numeric conversion for all other expected summary fields
             try:
                 params_for_sql[db_col] = float(value)
             except (ValueError, TypeError):
                 logger.warning(f"Could not convert summary key '{summary_key}' with value '{value}' to float for DB column '{db_col}'. Setting None.")
                 params_for_sql[db_col] = None
                 numeric_conversion_failures.append(summary_key)

    params_for_sql['data_path'] = data_path_str

    # --- Build and execute SQL INSERT statement ---
    valid_db_columns = list(params_for_sql.keys())
    placeholders = ', '.join(':' + k for k in valid_db_columns)
    sql = f"INSERT INTO rides ({', '.join(valid_db_columns)}) VALUES ({placeholders})"

    logger.debug(f"Executing SQL: {sql}")
    # logger.debug(f"With Params for SQL: {params_for_sql}") # Log params if needed, be mindful of size/content

    ride_id = None
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params_for_sql)
        conn.commit()
        ride_id = cursor.lastrowid
        if ride_id is not None:
            logger.info(f"Ride summary added to DB. ID: {ride_id}")
            if numeric_conversion_failures:
                 logger.warning(f"Ride {ride_id} added, but some numeric summary fields failed conversion: {numeric_conversion_failures}")
            return ride_id
        else:
            logger.error("DB Insert successful but no valid ride ID returned.")
            if data_path.exists(): os.remove(data_path); logger.info(f"Removed {data_path} because DB insert yielded no ID.")
            return None
    except sqlite3.IntegrityError as e:
        logger.warning(f"IntegrityError adding ride for {data_path_str}: {e}. Ride likely already exists.")
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Removed duplicate Parquet file {data_path_str} on IntegrityError.")
            except OSError as os_e: logger.error(f"Error removing duplicate Parquet {data_path_str}: {os_e}", exc_info=True)
        return None
    except sqlite3.Error as e:
        logger.error(f"DB error adding ride summary: {e} ({type(e).__name__})", exc_info=True)
        logger.error(f"Failed SQL: {sql}")
        # logger.error(f"Failed Params: {params_for_sql}") # Be cautious logging potentially sensitive data
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Removed Parquet file {data_path_str} on DB Error.")
            except OSError as os_e: logger.error(f"Error removing Parquet {data_path_str}: {os_e}", exc_info=True)
        return None
    finally:
        if cursor: cursor.close()
        if conn:
            conn.close()
            logger.debug("DB connection closed after add_ride.")


# --- Other functions (get_rides, get_ride_summary, get_ride_data, delete_ride) remain unchanged ---
def get_rides():
    """Retrieves a list of all rides (ID, filename, start_time) from the DB."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, start_time FROM rides ORDER BY start_time DESC")
        rides = cursor.fetchall()
        # Convert sqlite3.Row objects to dictionaries
        return [dict(row) for row in rides] if rides else []
    except sqlite3.Error as e:
        logger.error(f"DB error fetching rides list: {e}", exc_info=True)
        return []
    finally:
        if conn:
            conn.close()
            logger.debug("DB connection closed after get_rides.")

def get_ride_summary(ride_id):
    """Retrieves the summary data for a specific ride ID from the DB."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rides WHERE id = ?", (ride_id,))
        ride_summary_row = cursor.fetchone()
        return dict(ride_summary_row) if ride_summary_row else None
    except sqlite3.Error as e:
        logger.error(f"DB error fetching summary for ride ID {ride_id}: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()
            logger.debug(f"DB connection closed after get_ride_summary ({ride_id}).")

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
            return None
    except sqlite3.Error as e:
        logger.error(f"DB error getting data path for ride ID {ride_id}: {e}", exc_info=True)
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
                # Optional: Perform sanity checks on loaded data if needed
                return ride_data
            except Exception as e:
                logger.error(f"Error reading Parquet file {data_path}: {e}", exc_info=True)
                return None
        else:
            logger.error(f"Data path '{data_path}' found in DB but file is missing.")
            return None
    else:
         # Should have been caught earlier, but as a fallback
        logger.error(f"Data path string was None or empty for ride ID {ride_id} after DB query.")
        return None


def delete_ride(ride_id):
    """Deletes a ride's summary from the DB and its associated Parquet data file."""
    conn = get_db_connection()
    data_path_str = None
    if conn is None:
        logger.error("Delete Ride: Failed DB connection.")
        return False

    # 1. Get the data path *before* deleting the DB record
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT data_path FROM rides WHERE id = ?", (ride_id,))
        result = cursor.fetchone()
        if result and result['data_path']:
            data_path_str = result['data_path']
        else:
            logger.warning(f"Could not find data_path for ride ID {ride_id} to delete file.")
            # Proceed with DB deletion attempt anyway
    except sqlite3.Error as e:
        logger.error(f"DB error getting data path for deletion (ride ID {ride_id}): {e}", exc_info=True)
        # Don't return yet, still try to delete DB record
    finally:
        if conn: conn.close() # Close connection after fetching path

    # 2. Delete the record from the database
    conn = get_db_connection() # Re-establish connection
    deleted_db_record = False
    if conn is None:
         logger.error("Delete Ride: Failed DB connection for deletion.")
         return False # Cannot proceed without DB connection
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rides WHERE id = ?", (ride_id,))
        conn.commit()
        deleted_db_record = cursor.rowcount > 0 # Check if any row was affected
        if deleted_db_record:
            logger.info(f"Successfully deleted DB record for ride ID: {ride_id}")
        else:
            logger.warning(f"Ride ID {ride_id} not found in database for deletion.")
    except sqlite3.Error as e:
        logger.error(f"DB error deleting ride ID {ride_id}: {e}", exc_info=True)
        # If DB delete fails, we probably shouldn't delete the file either
        if conn: conn.close()
        return False
    finally:
        if conn:
            conn.close()
            logger.debug(f"DB connection closed after delete attempt ({ride_id}).")

    # 3. Delete the Parquet file if the path was found and DB record was deleted
    deleted_file = False
    if data_path_str:
        data_path = Path(data_path_str)
        if data_path.exists():
            try:
                os.remove(data_path)
                logger.info(f"Successfully deleted data file: {data_path}")
                deleted_file = True
            except OSError as e:
                logger.error(f"Error deleting data file {data_path}: {e}", exc_info=True)
                deleted_file = False # File deletion failed
        else:
            logger.warning(f"File not found for deleted ride ID {ride_id}: {data_path}")
            deleted_file = True # Consider it success if file is already gone
    elif deleted_db_record:
         logger.warning(f"DB record deleted for ride ID {ride_id}, but data_path was unknown. File not deleted.")
         # Technically the DB record is gone, but file status is uncertain. Return based on DB status.
         deleted_file = False # Indicate uncertainty / potential issue

    # Return True only if both DB record and file (if path known) are successfully handled
    if data_path_str:
         return deleted_db_record and deleted_file
    else:
         # If path was unknown, success depends only on DB deletion
         return deleted_db_record