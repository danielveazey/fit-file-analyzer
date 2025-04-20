# app.py
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
from pathlib import Path
import fit_parser
import database
import plotting
import time
import logging
import os
import io

# --- Basic Logging Setup ---
# Set level to INFO, change to DEBUG for more verbose output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger for this module

# --- Constants and Configuration ---
KM_TO_MILES = 0.621371
METERS_TO_FEET = 3.28084
def C_to_F(temp_c):
    """Convert Celsius to Fahrenheit, handling None."""
    return (temp_c * 9/5) + 32 if temp_c is not None and pd.notna(temp_c) else None

# Display names and units for plotting (Imperial focused)
# Add more mappings here if you want other fields plottable by default
PLOT_DISPLAY_MAPPING = {
    'speed_kmh': 'Speed (mph)',
    'altitude': 'Elevation (ft)',
    'heart_rate': 'Heart Rate (bpm)',
    'cadence': 'Cadence (rpm)',
    'power': 'Power (W)',
    'temperature': 'Temperature (°F)',
    'grade': 'Grade (%)' # Example: Adding grade if commonly present
}

# Unit conversions for plotting
PLOT_CONVERSIONS = {
    'speed_kmh': KM_TO_MILES,
    'altitude': METERS_TO_FEET,
    'temperature': C_to_F
    # Add other conversions if needed, e.g., {'distance': KM_TO_MILES}
}

# Columns to potentially add regression lines for in plots
ADD_REGRESSION_FOR = ['power', 'heart_rate']

# Labels for Zone Charts
HR_ZONE_LABELS = ["Zone 1 (Warmup)", "Zone 2 (Easy)", "Zone 3 (Aerobic)", "Zone 4 (Threshold)", "Zone 5 (Maximal)", "Zone 6 (Anaerobic+)"]
POWER_ZONE_LABELS = ["Zone 1 (Active Rec.)", "Zone 2 (Endurance)", "Zone 3 (Tempo)", "Zone 4 (Threshold)", "Zone 5 (VO2 Max)", "Zone 6 (Anaerobic)", "Zone 7 (Neuromuscular)"]

# --- Page Config ---
st.set_page_config(page_title="FIT File Analyzer", page_icon="🚴", layout="wide")
st.title("🚴 GPS Ride Analyzer (.FIT Files)")
st.markdown("Upload a `.fit` file or select a previously uploaded ride from the sidebar.")

# --- Initialization ---
# Initialize database schema if needed
try:
    database.init_db()
except Exception as db_init_e:
    st.error(f"Fatal Error: Database initialization failed: {db_init_e}")
    st.stop() # Stop execution if DB cannot be initialized

# --- State Management ---
# Initialize session state keys if they don't exist
default_state = {
    'selected_ride_id': None,
    'ride_data_df': pd.DataFrame(), # Initialize with empty DataFrame
    'ride_summary': None,
    'upload_status_message': None,
    'upload_status_type': None, # 'success', 'error', 'info', 'warning'
    'confirm_delete': False
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Utility Functions ---
def format_duration(seconds):
    """Formats seconds into HH:MM:SS, handling None, NaN, and potential type issues robustly."""
    if seconds is None or pd.isna(seconds):
        return "N/A"
    try:
        secs_float = float(seconds)
        if secs_float < 0:
            logger.warning(f"format_duration received negative value {secs_float}, returning N/A")
            return "N/A"
        # Use int() only *after* confirming it's a valid non-negative float
        secs_int = int(secs_float)
        hours, remainder = divmod(secs_int, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}"
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not format duration from value: '{seconds}' (Type: {type(seconds)}). Error: {e}")
        return "N/A"

# --- Data Loading and Processing Functions ---
def load_ride_data(ride_id):
    """Loads summary and full data for a given ride ID into session state."""
    st.session_state['upload_status_message'] = None # Clear previous upload messages
    st.session_state['confirm_delete'] = False # Reset delete confirmation

    if ride_id is None:
        # Clear data if ride_id is None (e.g., "Clear Selection" clicked)
        st.session_state.update({
            'selected_ride_id': None,
            'ride_data_df': pd.DataFrame(),
            'ride_summary': None
        })
        logger.info("Cleared ride selection.")
        return True # Indicate success in clearing

    logger.info(f"Attempting to load data for ride ID: {ride_id}")
    st.session_state['selected_ride_id'] = ride_id

    # Retrieve summary first
    summary = database.get_ride_summary(ride_id)
    if not summary:
        logger.error(f"Failed to retrieve summary for ride ID {ride_id}.")
        st.session_state['upload_status_message'] = f"Error: Could not load summary for Ride ID {ride_id}."
        st.session_state['upload_status_type'] = "error"
        st.session_state.update({'ride_data_df': pd.DataFrame(), 'ride_summary': None}) # Clear potentially stale data
        return False
    st.session_state['ride_summary'] = summary
    logger.info(f"Summary loaded for ride ID {ride_id}.")

    # Retrieve full DataFrame from Parquet
    df = None
    try:
        with st.spinner(f"Loading data file for Ride ID {ride_id}..."):
            df = database.get_ride_data(ride_id)
    except Exception as e:
        logger.error(f"Exception during get_ride_data for {ride_id}: {e}", exc_info=True)
        st.session_state['upload_status_message'] = f"Error loading data file for Ride ID {ride_id}."
        st.session_state['upload_status_type'] = "error"
        st.session_state.update({'ride_data_df': pd.DataFrame()}) # Clear DF
        return False # Indicate failure


    if df is None or df.empty:
        logger.error(f"Failed to load DataFrame or DataFrame is empty for ride ID {ride_id}.")
        st.session_state['upload_status_message'] = f"Error: Data file missing or empty for Ride ID {ride_id}."
        st.session_state['upload_status_type'] = "error"
        st.session_state['ride_data_df'] = pd.DataFrame() # Ensure it's an empty DF
        # Keep summary loaded if it was retrieved? Or clear both? Let's keep summary for now.
        return False # Indicate failure (missing data file)

    # --- Post-load Data Type Verification/Conversion (Optional but Recommended) ---
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        logger.warning(f"Timestamp column in loaded DataFrame for ride {ride_id} is not datetime. Attempting conversion.")
        try:
            # Attempt conversion, coerce errors to NaT, then drop rows with NaT timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            if df.empty:
                 logger.error(f"All timestamps invalid after conversion for ride {ride_id}.")
                 st.session_state['upload_status_message'] = f"Error: Invalid timestamp data in file for Ride ID {ride_id}."
                 st.session_state['upload_status_type'] = "error"
                 st.session_state['ride_data_df'] = pd.DataFrame()
                 return False
        except Exception as e:
            logger.error(f"Timestamp conversion failed after loading DataFrame for ride {ride_id}: {e}")
            st.session_state['upload_status_message'] = f"Error processing timestamps for Ride ID {ride_id}."
            st.session_state['upload_status_type'] = "error"
            st.session_state['ride_data_df'] = pd.DataFrame()
            return False

    # Store the loaded and potentially cleaned DataFrame
    st.session_state['ride_data_df'] = df
    logger.info(f"Successfully loaded data for ride ID {ride_id}. DataFrame shape: {df.shape}")
    return True # Indicate successful load

def process_uploaded_file():
    """Callback function to handle FIT file uploads."""
    uploaded_file_object = st.session_state.get('fit_uploader') # Get file from widget state
    if uploaded_file_object is None:
        logger.info("Upload callback triggered but no file found in state.")
        return # No file to process

    filename = uploaded_file_object.name
    logger.info(f"Processing uploaded file: {filename}")
    st.session_state['upload_status_message'] = f"Processing {filename}..."
    st.session_state['upload_status_type'] = "info"

    file_buffer = None
    try:
        # Read file content into a BytesIO buffer for fit_parser
        file_content = uploaded_file_object.getvalue()
        file_buffer = io.BytesIO(file_content)

        # Use a spinner during parsing and saving
        with st.spinner(f"Parsing {filename}..."):
            df, summary = fit_parser.parse_fit_file(file_buffer)

        if df is None or df.empty or summary is None:
            logger.error(f"Parsing failed or returned no data/summary for {filename}.")
            st.session_state['upload_status_message'] = f"Error: Could not parse '{filename}' or file contains no record data."
            st.session_state['upload_status_type'] = "error"
            return

        # Add filename to summary before saving
        summary['filename'] = filename
        logger.info(f"Parsing successful for {filename}. Summary generated, DataFrame shape: {df.shape}")

        with st.spinner(f"Saving {filename} data..."):
            new_ride_id = database.add_ride(summary, df)

        if new_ride_id:
            logger.info(f"Ride '{filename}' successfully saved with ID: {new_ride_id}")
            st.session_state['upload_status_message'] = f"Ride '{filename}' saved successfully (ID: {new_ride_id})."
            st.session_state['upload_status_type'] = "success"
            # Automatically select and load the newly added ride
            load_ride_data(new_ride_id)
            # No need to rerun here, state change will trigger it if load_ride_data succeeds
        else:
            # add_ride returns None on failure (e.g., DB error, duplicate)
            logger.error(f"Failed to save ride '{filename}' to database.")
            # Check if it might be a duplicate based on path (IntegrityError logged by add_ride)
            st.session_state['upload_status_message'] = f"Error: Failed to save '{filename}'. It might already exist or a database error occurred."
            st.session_state['upload_status_type'] = "error"

    except Exception as e:
        logger.error(f"Unexpected error processing uploaded file '{filename}': {e}", exc_info=True)
        st.session_state['upload_status_message'] = f"Critical error processing '{filename}'. Check logs."
        st.session_state['upload_status_type'] = "error"
    finally:
        # Ensure the buffer is closed
        if file_buffer:
            file_buffer.close()
        # Clear the uploader state after processing to allow re-uploading the same file name
        # Note: This might require using st.empty() or careful state management if
        # the default file_uploader behavior is problematic upon clearing.
        # For now, let's assume the standard behavior is acceptable.
        # st.session_state['fit_uploader'] = None # Be cautious with this line

# --- Sidebar ---
st.sidebar.header("Upload & Select Ride")

# File Uploader
# Use 'on_change' for immediate processing after file selection
st.sidebar.file_uploader(
    "Upload a new .fit file",
    type=["fit"],
    key="fit_uploader", # Unique key for the widget
    on_change=process_uploaded_file, # Function to call when file changes
    accept_multiple_files=False,
    help="Upload a FIT file from your GPS device."
)

# Display upload status messages (will be updated by callbacks)
# We check these at the start of the main area render now.

st.sidebar.markdown("---")

# Ride Selection Dropdown
st.sidebar.header("View Past Ride")
try:
    past_rides = database.get_rides()
except Exception as e:
    st.sidebar.error(f"Error loading past rides: {e}")
    past_rides = []

# Create options for the selectbox: "Date Time - Filename (ID: id)"
ride_options = {}
if past_rides:
    for ride in past_rides:
        try:
            # Format timestamp safely
            start_ts = pd.to_datetime(ride.get('start_time'))
            if pd.notna(start_ts):
                 # Use a consistent, sortable format
                display_name = f"{start_ts.strftime('%Y-%m-%d %H:%M')} - {ride.get('filename', 'N/A')} (ID: {ride.get('id', 'N/A')})"
                ride_options[display_name] = ride.get('id')
            else:
                 display_name = f"Unknown Date - {ride.get('filename', 'N/A')} (ID: {ride.get('id', 'N/A')})"
                 ride_options[display_name] = ride.get('id')
        except Exception as format_e:
             logger.warning(f"Error formatting ride display name for ID {ride.get('id')}: {format_e}")
             display_name = f"Error - (ID: {ride.get('id', 'N/A')})"
             # Still add it with a placeholder name if ID exists
             if ride.get('id') is not None:
                  ride_options[display_name] = ride.get('id')

if not ride_options:
    st.sidebar.info("No past rides found in the database.")
else:
    # Sort options naturally (usually by date due to formatting)
    options_list = sorted(list(ride_options.keys()), reverse=True) # Show newest first

    # Find the display name corresponding to the currently selected ID for default selection
    current_id = st.session_state.get('selected_ride_id')
    id_to_option_name = {v: k for k, v in ride_options.items()}
    current_display_name = id_to_option_name.get(current_id)
    try:
         current_index = options_list.index(current_display_name) if current_display_name in options_list else 0
    except ValueError:
         current_index = 0 # Default to first item if current selection not found

    selected_ride_display_name = st.sidebar.selectbox(
        "Choose a ride:",
        options=options_list,
        index=current_index,
        key="ride_selector", # Unique key
        placeholder="Select a ride..."
    )

    # Check if selection changed
    newly_selected_ride_id = ride_options.get(selected_ride_display_name)
    if newly_selected_ride_id is not None and newly_selected_ride_id != st.session_state.get('selected_ride_id'):
        logger.info(f"Ride selected via dropdown: ID {newly_selected_ride_id}")
        # Load data and rerun only if loading is successful
        if load_ride_data(newly_selected_ride_id):
            st.rerun()
        else:
            # Error message should have been set by load_ride_data
            st.sidebar.error("Failed to load selected ride data.")

# --- Ride Management (Delete Button) ---
if st.session_state.get('selected_ride_id') is not None:
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Selection / View Welcome", key="clear_selection"):
        load_ride_data(None) # Clear data by passing None
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Manage Ride")
    ride_id_to_delete = st.session_state.selected_ride_id
    summary_for_delete = st.session_state.get('ride_summary')
    ride_name_for_delete = f"ID {ride_id_to_delete}"
    if summary_for_delete and summary_for_delete.get('filename'):
        ride_name_for_delete = f"{summary_for_delete['filename']} (ID: {ride_id_to_delete})"

    # Delete button triggers confirmation state
    if st.sidebar.button(f"🗑️ Delete Ride: {ride_name_for_delete}", type="primary", key="delete_button"):
        st.session_state.confirm_delete = True
        st.rerun() # Rerun to show confirmation

    # Confirmation section appears if confirm_delete is True
    if st.session_state.get('confirm_delete', False):
        st.sidebar.warning(f"**Are you sure you want to delete '{ride_name_for_delete}'?** This cannot be undone.")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Yes, Delete Permanently", key="confirm_delete_yes"):
                logger.info(f"Attempting to delete ride ID: {ride_id_to_delete}")
                with st.spinner(f"Deleting {ride_name_for_delete}..."):
                    delete_success = database.delete_ride(ride_id_to_delete)

                if delete_success:
                    logger.info(f"Successfully deleted ride ID: {ride_id_to_delete}")
                    st.session_state['upload_status_message'] = f"Ride '{ride_name_for_delete}' deleted."
                    st.session_state['upload_status_type'] = 'success'
                else:
                    logger.error(f"Failed to delete ride ID: {ride_id_to_delete}")
                    st.session_state['upload_status_message'] = f"Error: Failed to delete '{ride_name_for_delete}'."
                    st.session_state['upload_status_type'] = 'error'

                # Reset state after deletion attempt
                st.session_state.confirm_delete = False
                load_ride_data(None) # Clear selection
                time.sleep(0.5) # Brief pause to allow user to see message
                st.rerun()
        with col2:
            if st.button("Cancel", key="confirm_delete_cancel"):
                st.session_state.confirm_delete = False
                st.rerun() # Rerun to hide confirmation

# --- Main Area ---
# Display status messages first
if st.session_state.get('upload_status_message'):
    status_type = st.session_state.get('upload_status_type', 'info')
    msg = st.session_state['upload_status_message']
    if status_type == "success": st.success(msg)
    elif status_type == "error": st.error(msg)
    elif status_type == "warning": st.warning(msg)
    else: st.info(msg)
    # Clear the message after displaying it once
    st.session_state['upload_status_message'] = None
    st.session_state['upload_status_type'] = None


# Check if a ride is selected
current_ride_id = st.session_state.get('selected_ride_id')

if current_ride_id is None:
    st.markdown("## Welcome!")
    st.markdown("Use the sidebar to upload a `.fit` file or select a past ride to view its analysis.")
    # Optionally show some introductory text or images here
else:
    # A ride is selected, display its data
    df = st.session_state.get('ride_data_df')
    summary = st.session_state.get('ride_summary')

    if summary is None:
        st.error(f"Error: Summary data could not be loaded for Ride ID {current_ride_id}. Please try reloading or selecting another ride.")
        st.stop()
    # No need to check df for None here, load_ride_data handles that and sets error message if failed.
    # If df is empty after load_ride_data failed, the tabs below will handle it gracefully.

    # --- Header Display ---
    display_filename = summary.get('filename', f"Ride ID {current_ride_id}")
    start_time_val = summary.get('start_time')
    start_time_str = "N/A"
    try:
        # Summary start_time should be timezone-naive from fit_parser
        if start_time_val is not None:
            start_time_dt = pd.to_datetime(start_time_val) # Already naive UTC
            if pd.notna(start_time_dt):
                start_time_str = start_time_dt.strftime('%Y-%m-%d %H:%M:%S') + " UTC"
    except Exception as e:
        logger.warning(f"Could not format start_time from summary '{start_time_val}': {e}")

    st.header(f"Ride Details: {display_filename}")
    st.subheader(f"Started: {start_time_str}")

    # --- Display Metrics ---
    st.subheader("Overall Stats")
    col1, col2, col3, col4 = st.columns(4)
    summary_data = summary # Use the loaded summary dict

    with col1:
        st.metric("Total Duration", format_duration(summary_data.get('total_time_seconds')))
        st.metric("Moving Time", format_duration(summary_data.get('moving_time_seconds')))
    with col2:
        dist_km = summary_data.get('total_distance_km')
        dist_mi = dist_km * KM_TO_MILES if dist_km is not None else None
        st.metric("Total Distance", f"{dist_mi:.2f} mi" if dist_mi is not None else 'N/A')

        gain_m = summary_data.get('total_elevation_gain_m')
        gain_ft = gain_m * METERS_TO_FEET if gain_m is not None else None
        st.metric("Total Ascent", f"{gain_ft:.0f} ft" if gain_ft is not None else 'N/A')
    with col3:
        spd_kmh = summary_data.get('avg_speed_kmh')
        spd_mph = spd_kmh * KM_TO_MILES if spd_kmh is not None else None
        st.metric("Avg Speed", f"{spd_mph:.1f} mph" if spd_mph is not None else 'N/A')

        # Calculate Max Speed from DataFrame if possible
        max_spd_mph_str = "N/A"
        if df is not None and 'speed_kmh' in df.columns and df['speed_kmh'].notna().any():
            try:
                 # Ensure the column is numeric before calculating max
                 if pd.api.types.is_numeric_dtype(df['speed_kmh']):
                      max_spd_kmh = df['speed_kmh'].max(skipna=True)
                      if pd.notna(max_spd_kmh):
                           max_spd_mph = max_spd_kmh * KM_TO_MILES
                           max_spd_mph_str = f"{max_spd_mph:.1f} mph"
                 else:
                      logger.warning("Max Speed calculation skipped: 'speed_kmh' column not numeric.")
                      max_spd_mph_str = "Data Type Error"
            except Exception as e:
                 logger.error(f"Error calculating Max Speed from DataFrame: {e}", exc_info=True)
                 max_spd_mph_str="Calc Error"
        st.metric("Max Speed", max_spd_mph_str)
    with col4:
        total_cals = summary_data.get('total_calories')
        st.metric("Total Calories", f"{int(total_cals):,}" if total_cals is not None else 'N/A')

    st.subheader("Performance")
    col1a, col2a, col3a, col4a = st.columns(4)
    with col1a:
        avg_hr = summary_data.get('avg_heart_rate')
        st.metric("Avg HR", f"{avg_hr:.0f} bpm" if avg_hr is not None else 'N/A')
        max_hr = summary_data.get('max_hr')
        st.metric("Max HR", f"{max_hr:.0f} bpm" if max_hr is not None else 'N/A')
    with col2a:
        avg_cad = summary_data.get('avg_cadence')
        st.metric("Avg Cad", f"{avg_cad:.0f} rpm" if avg_cad is not None else 'N/A')
        max_cad = summary_data.get('max_cadence')
        st.metric("Max Cad", f"{max_cad:.0f} rpm" if max_cad is not None else 'N/A')
    with col3a:
        avg_pwr = summary_data.get('avg_power')
        st.metric("Avg Power", f"{avg_pwr:.0f} W" if avg_pwr is not None else 'N/A')
        max_pwr = summary_data.get('max_power')
        st.metric("Max Power", f"{max_pwr:.0f} W" if max_pwr is not None else 'N/A')
    with col4a:
        avg_tc = summary_data.get('avg_temp_c')
        avg_tf = C_to_F(avg_tc)
        st.metric("Avg Temp", f"{avg_tf:.0f} °F" if avg_tf is not None else 'N/A')

        min_tc = summary_data.get('min_temp_c')
        max_tc = summary_data.get('max_temp_c')
        min_tf = C_to_F(min_tc)
        max_tf = C_to_F(max_tc)
        temp_range_str = "N/A"
        if min_tf is not None and max_tf is not None:
             temp_range_str = f"{min_tf:.0f}°F - {max_tf:.0f}°F"
        st.metric("Temp Range", temp_range_str)

    st.markdown("---")

    # --- Tabs for Visualization ---
    if df is None or df.empty:
        st.warning(f"Ride data file could not be loaded or is empty for Ride ID {current_ride_id}. Cannot display map, plots, or raw data.")
    else:
        tab_map, tab_plots, tab_zones, tab_data = st.tabs(["🗺️ Route Map", "📊 Data Plots", "⏱️ Zones", "🗂️ Raw Data"])

        # --- Map Tab ---
        with tab_map:
            st.subheader("Route Map")
            # Pass the full DataFrame; plotting function handles extraction/dropna
            if 'latitude' in df.columns and 'longitude' in df.columns:
                 if df[['latitude', 'longitude']].dropna().shape[0] >= 2:
                     with st.spinner("Generating map..."):
                         try:
                             route_map = plotting.plot_route_map(df) # Pass full df
                             if route_map:
                                 # Use a unique key for the map based on ride ID to prevent state issues
                                 st_folium(route_map, height=500, use_container_width=True, key=f"folium_map_{current_ride_id}")
                             else:
                                 st.warning("Map could not be generated (plotting function returned None).")
                         except Exception as map_e:
                              logger.error(f"Error generating map: {map_e}", exc_info=True)
                              st.error("An error occurred while generating the map.")
                 else:
                     st.warning("Not enough valid GPS data points (latitude/longitude) to draw a map.")
            else:
                st.warning("No GPS data (latitude/longitude columns) found in the file.")

        # --- Plots Tab ---
        with tab_plots:
            st.subheader("Data Plots")
            # Find columns available for plotting based on mapping AND if they exist in DF and are numeric
            available_cols_internal = [
                col for col in PLOT_DISPLAY_MAPPING.keys()
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any()
            ]
            plot_options_display = [PLOT_DISPLAY_MAPPING[col] for col in available_cols_internal]

            if not plot_options_display:
                st.warning("No plottable numeric data found in the selected ride file.")
            else:
                # Default selection: Limit to first 3-4 common metrics if available
                default_selection_keys = ['speed_kmh', 'altitude', 'heart_rate', 'power', 'cadence']
                default_display = [PLOT_DISPLAY_MAPPING[k] for k in default_selection_keys if k in available_cols_internal][:4] # Max 4 default plots

                selected_display = st.multiselect(
                    "Select data to plot:",
                    options=plot_options_display,
                    default=default_display,
                    key=f"plot_selector_{current_ride_id}" # Unique key per ride
                )

                if selected_display:
                    # Map display names back to internal column names
                    inv_map = {v: k for k, v in PLOT_DISPLAY_MAPPING.items()}
                    selected_internal = [inv_map[name] for name in selected_display if name in inv_map]

                    if selected_internal:
                        with st.spinner("Generating plots..."):
                            try:
                                # Use the defined conversions and regression list
                                fig = plotting.plot_data(
                                    df,
                                    y_vars=selected_internal,
                                    x_var='timestamp', # Assuming timestamp column exists and is valid
                                    display_mapping=PLOT_DISPLAY_MAPPING,
                                    conversions=PLOT_CONVERSIONS,
                                    add_regression_for=ADD_REGRESSION_FOR
                                )

                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error("Plot generation failed. Check if 'timestamp' data is valid.")
                            except Exception as plot_e:
                                 logger.error(f"Error during plot generation: {plot_e}", exc_info=True)
                                 st.error("An error occurred while generating the plots.")
                    else:
                        st.error("Could not map selected plot options to data columns.") # Should not happen if options generated correctly
                else:
                    st.info("Select one or more data types from the list above to plot.")

        # --- Zones Tab ---
        with tab_zones:
            st.subheader("Time in Zones")
            if summary_data: # Ensure summary is loaded
                # --- Heart Rate Zones ---
                hr_zone_times = []
                hr_keys_found = False
                for i in range(len(HR_ZONE_LABELS)): # Iterate up to max potential zones
                    key = f'time_in_hr_zone_{i}'
                    time_val = summary_data.get(key) # Get value, defaults to None if key missing
                    # Append valid time or 0.0 if missing/invalid
                    if time_val is not None and pd.notna(time_val) and time_val >= 0:
                         hr_zone_times.append(float(time_val))
                         hr_keys_found = True # Mark that we found at least one valid zone time
                    else:
                         hr_zone_times.append(0.0) # Use 0 for missing/invalid zones in the list

                # Only plot if we found *any* HR zone data and the total time is > 0
                if hr_keys_found and sum(hr_zone_times) > 0:
                    # Adjust labels to match the length of data we actually have (up to MAX_HR_ZONES)
                    current_hr_labels = HR_ZONE_LABELS[:len(hr_zone_times)]
                    st.markdown("##### Heart Rate Zones")
                    try:
                         hr_fig = plotting.plot_zone_chart(hr_zone_times, current_hr_labels, "Time in HR Zones", color_scale='Plasma')
                         if hr_fig:
                              st.plotly_chart(hr_fig, use_container_width=True)
                         else:
                              st.warning("Could not generate HR Zone chart (plotting function returned None).")
                    except Exception as hr_chart_e:
                         logger.error(f"Error plotting HR zone chart: {hr_chart_e}", exc_info=True)
                         st.error("An error occurred generating the HR Zone chart.")
                else:
                    st.info("Heart Rate Zone data is unavailable or zero for this ride.")

                st.markdown("---") # Separator

                # --- Power Zones ---
                pwr_zone_times = []
                pwr_keys_found = False
                for i in range(len(POWER_ZONE_LABELS)): # Iterate up to max potential zones
                    key = f'time_in_pwr_zone_{i}'
                    time_val = summary_data.get(key)
                    if time_val is not None and pd.notna(time_val) and time_val >= 0:
                        pwr_zone_times.append(float(time_val))
                        pwr_keys_found = True
                    else:
                        pwr_zone_times.append(0.0)

                if pwr_keys_found and sum(pwr_zone_times) > 0:
                    current_pwr_labels = POWER_ZONE_LABELS[:len(pwr_zone_times)]
                    st.markdown("##### Power Zones")
                    try:
                        pwr_fig = plotting.plot_zone_chart(pwr_zone_times, current_pwr_labels, "Time in Power Zones", color_scale='Cividis')
                        if pwr_fig:
                            st.plotly_chart(pwr_fig, use_container_width=True)
                        else:
                             st.warning("Could not generate Power Zone chart (plotting function returned None).")
                    except Exception as pwr_chart_e:
                         logger.error(f"Error plotting Power zone chart: {pwr_chart_e}", exc_info=True)
                         st.error("An error occurred generating the Power Zone chart.")
                else:
                    st.info("Power Zone data is unavailable or zero for this ride.")
            else:
                # This case should be prevented by the check at the start of the 'else' block
                st.error("Ride summary data is missing, cannot display zone information.")

        # --- Raw Data Tab ---
        with tab_data:
            st.subheader("Raw Data Record Messages")
            st.markdown("Displaying all columns extracted from the FIT file's record messages.")
            # Display the potentially wide DataFrame
            st.dataframe(df, use_container_width=True)

            # Add download button for the full DataFrame
            try:
                # Create a safe filename stem
                safe_filename_stem = "ride_data"
                if summary_data and summary_data.get('filename'):
                    safe_filename_stem = "".join(c if c.isalnum() else "_" for c in Path(summary_data['filename']).stem)
                elif current_ride_id:
                     safe_filename_stem = f"ride_{current_ride_id}_data"

                csv_filename = f"{safe_filename_stem}.csv"
                # Convert DataFrame to CSV bytes
                csv_data = df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Download Full Data as CSV",
                    data=csv_data,
                    file_name=csv_filename,
                    mime='text/csv',
                    key=f"download_csv_{current_ride_id}" # Unique key
                )
            except Exception as e:
                logger.error(f"Error preparing CSV for download: {e}", exc_info=True)
                st.warning("Could not prepare data for CSV download.")

