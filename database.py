# database.py
import sqlite3
import os
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
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
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        logger.info("Database connection successful.")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        return None

def init_db():
    """Initializes the database schema if it doesn't exist."""
    conn = get_db_connection()
    if conn is None: return
    try:
        cursor = conn.cursor()
        # --- Added total_elevation_gain_m column ---
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
                total_elevation_gain_m REAL, -- New column for elevation gain in meters
                data_path TEXT UNIQUE NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        logger.info("Database schema initialized or verified.")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
    finally:
        if conn: conn.close()

def add_ride(summary_data, dataframe):
    """Adds ride summary (including elevation gain) and data to the database."""
    conn = get_db_connection()
    if conn is None: return None

    safe_filename = "".join(c if c.isalnum() else "_" for c in summary_data['filename'])
    timestamp_str = pd.Timestamp(summary_data['start_time']).strftime('%Y%m%d_%H%M%S')
    parquet_filename = f"ride_{timestamp_str}_{safe_filename}.parquet"
    data_path = DATA_DIR / parquet_filename

    try:
        if dataframe is not None and not dataframe.empty:
            dataframe.to_parquet(data_path, index=False)
            logger.info(f"DataFrame saved to {data_path}")
        else:
             logger.warning("DataFrame is empty or None, skipping parquet save.")
             return None
    except Exception as e:
        logger.error(f"Error saving DataFrame to Parquet: {e}")
        return None

    # --- Include total_elevation_gain_m in SQL query and params ---
    sql = """
        INSERT INTO rides (filename, start_time, total_time_seconds, total_distance_km,
                           avg_speed_kmh, avg_heart_rate, avg_cadence, avg_power,
                           total_elevation_gain_m, data_path)
        VALUES (:filename, :start_time, :total_time_seconds, :total_distance_km,
                :avg_speed_kmh, :avg_heart_rate, :avg_cadence, :avg_power,
                :total_elevation_gain_m, :data_path)
    """
    params = {
        'filename': summary_data.get('filename'),
        'start_time': summary_data.get('start_time'),
        'total_time_seconds': summary_data.get('total_time'),
        'total_distance_km': summary_data.get('total_distance'),
        'avg_speed_kmh': summary_data.get('avg_speed'),
        'avg_heart_rate': summary_data.get('avg_heart_rate'),
        'avg_cadence': summary_data.get('avg_cadence'),
        'avg_power': summary_data.get('avg_power'),
        'total_elevation_gain_m': summary_data.get('total_elevation_gain_m'), # Get the new value
        'data_path': str(data_path)
    }
    # Replace potential pandas NA or numpy nan with None for SQLite compatibility
    for key, value in params.items():
         if pd.isna(value):
             params[key] = None

    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
        ride_id = cursor.lastrowid
        logger.info(f"Ride summary added to database with ID: {ride_id}")
        return ride_id
    except sqlite3.IntegrityError:
        logger.warning(f"Ride with data path {data_path} likely already exists.")
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Removed orphaned Parquet file: {data_path}")
            except OSError as e: logger.error(f"Error removing orphaned Parquet file {data_path}: {e}")
        return None
    except sqlite3.Error as e:
        logger.error(f"Database error adding ride: {e}")
        if data_path.exists():
             try: os.remove(data_path); logger.info(f"Removed orphaned Parquet file: {data_path}")
             except OSError as e: logger.error(f"Error removing orphaned Parquet file {data_path}: {e}")
        return None
    finally:
        if conn: conn.close()

def get_rides():
    """Retrieves a list of all rides from the database."""
    conn = get_db_connection()
    if conn is None: return []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, start_time FROM rides ORDER BY start_time DESC")
        rides = cursor.fetchall()
        return [dict(row) for row in rides]
    except sqlite3.Error as e:
        logger.error(f"Database error fetching rides: {e}")
        return []
    finally:
        if conn: conn.close()

def get_ride_summary(ride_id):
    """Retrieves summary details (including elevation gain) for a specific ride ID."""
    conn = get_db_connection()
    if conn is None: return None
    try:
        cursor = conn.cursor()
        # SELECT * should now include the new column
        cursor.execute("SELECT * FROM rides WHERE id = ?", (ride_id,))
        ride_summary = cursor.fetchone()
        return dict(ride_summary) if ride_summary else None
    except sqlite3.Error as e:
        logger.error(f"Database error fetching ride summary for ID {ride_id}: {e}")
        return None
    finally:
        if conn: conn.close()


def get_ride_data(ride_id):
    """Loads the ride DataFrame from its Parquet file."""
    conn = get_db_connection()
    if conn is None: return None
    ride_data = None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT data_path FROM rides WHERE id = ?", (ride_id,))
        result = cursor.fetchone()
        if result and result['data_path']:
            data_path = Path(result['data_path'])
            if data_path.exists():
                try:
                    ride_data = pd.read_parquet(data_path)
                    if 'timestamp' in ride_data.columns and not pd.api.types.is_datetime64_any_dtype(ride_data['timestamp']):
                       try: ride_data['timestamp'] = pd.to_datetime(ride_data['timestamp'])
                       except Exception as time_e: logger.warning(f"Could not auto-convert timestamp column to datetime: {time_e}")
                    logger.info(f"Successfully loaded DataFrame from {data_path}")
                except Exception as e: logger.error(f"Error reading Parquet file {data_path}: {e}"); return None
            else: logger.error(f"Parquet file not found: {data_path}"); return None
        else: logger.warning(f"No data path found for ride ID {ride_id}"); return None
    except sqlite3.Error as e: logger.error(f"Database error retrieving data path for ride ID {ride_id}: {e}"); return None
    finally:
        if conn: conn.close()
    return ride_data

def delete_ride(ride_id):
    """Deletes a ride from the database and its associated data file."""
    conn = get_db_connection()
    if conn is None: return False
    data_path_str = None
    try: # Get path first
        cursor = conn.cursor()
        cursor.execute("SELECT data_path FROM rides WHERE id = ?", (ride_id,))
        result = cursor.fetchone()
        if result: data_path_str = result['data_path']
    except sqlite3.Error as e: logger.error(f"Error retrieving data path for deletion (ID: {ride_id}): {e}")
    finally:
         if conn: conn.close() # Close connection after read attempt

    conn = get_db_connection() # Reopen for delete
    if conn is None: return False
    deleted_db = False
    try: # Delete DB entry
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rides WHERE id = ?", (ride_id,))
        conn.commit()
        if cursor.rowcount > 0: logger.info(f"Deleted ride ID {ride_id} from database."); deleted_db = True
        else: logger.warning(f"Ride ID {ride_id} not found in DB for deletion.")
    except sqlite3.Error as e: logger.error(f"Error deleting ride ID {ride_id} from DB: {e}")
    finally:
        if conn: conn.close() # Close after delete attempt

    # Delete data file (only if DB deletion was attempted/successful and path known)
    deleted_file = False
    if data_path_str:
        data_path = Path(data_path_str)
        if data_path.exists():
            try: os.remove(data_path); logger.info(f"Deleted data file: {data_path}"); deleted_file = True
            except OSError as e: logger.error(f"Error deleting data file {data_path}: {e}")
        else: logger.warning(f"Data file not found for deleted ride: {data_path}"); deleted_file = True # Treat as success if file is already gone
    elif deleted_db: # If DB deleted but path unknown, maybe warn?
         logger.warning(f"DB entry deleted for ride ID {ride_id}, but data path was unknown. File system check might be needed.")
         deleted_file = True # Consider file part 'done' as there was no path

    # Overall success requires DB deletion succeeded (or wasn't found) AND file deletion succeeded (or wasn't needed/possible)
    return (deleted_db or not data_path_str) and deleted_file