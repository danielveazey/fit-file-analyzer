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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
KM_TO_MILES = 0.621371
METERS_TO_FEET = 3.28084
def C_to_F(temp_c): return (temp_c * 9/5) + 32 if temp_c is not None else None
PLOT_DISPLAY_MAPPING = { 'speed_kmh': 'Speed (mph)', 'altitude': 'Elevation (ft)', 'heart_rate': 'Heart Rate (bpm)', 'cadence': 'Cadence (rpm)', 'power': 'Power (W)', 'temperature': 'Temperature (°F)' }

# --- Page Config ---
st.set_page_config(page_title="FIT File Analyzer", page_icon="🚴", layout="wide")
st.title("🚴 GPS Ride Analyzer (.FIT Files)")

# --- Initialization ---
database.init_db()

# --- State Management ---
default_state = { 'selected_ride_id': None, 'ride_data_df': None, 'ride_summary': None, 'upload_status_message': None, 'upload_status_type': None, 'confirm_delete': False }
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Functions ---
def format_duration(seconds):
    """Formats seconds into HH:MM:SS, handling None or NaN."""
    if seconds is None or pd.isna(seconds): return "N/A"
    try:
        secs = int(float(seconds))
        if secs < 0: return "N/A"
        hours, rem = divmod(secs, 3600)
        mins, s = divmod(rem, 60)
        return f"{int(hours):02}:{int(mins):02}:{int(s):02}"
    except (ValueError, TypeError):
        logger.warning(f"Could not format duration from value: {seconds}")
        return "N/A"

def load_ride_data(ride_id):
    """Loads summary and DataFrame for a given ride ID into session state."""
    success = False
    st.session_state['ride_summary'] = None # Ensure reset before load
    st.session_state['ride_data_df'] = None

    if ride_id is not None:
        st.session_state['upload_status_message'], st.session_state['upload_status_type'] = None, None
        st.session_state['selected_ride_id'] = ride_id
        logger.info(f"Attempting load for ride {ride_id}...")

        # Load Summary
        retrieved_summary = database.get_ride_summary(ride_id)
        if retrieved_summary:
            st.session_state['ride_summary'] = dict(retrieved_summary)
            logger.info(f"Summary loaded for {ride_id}.")
        else:
            logger.error(f"Failed summary retrieval for ride {ride_id}.")

        # Load DataFrame
        with st.spinner(f"Loading data file for Ride ID {ride_id}..."):
            df = database.get_ride_data(ride_id)

        if df is not None and not df.empty:
            logger.info(f"DataFrame loaded for ride {ride_id}. Shape: {df.shape}")
            # Ensure timestamp column is datetime type
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    # Use errors='coerce' to handle unparseable dates gracefully
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df = df.dropna(subset=['timestamp']) # Drop rows where conversion failed
                    logger.info("Converted loaded timestamp column to datetime.")
                except Exception as e:
                    logger.warning(f"Failed ts conversion: {e}")
            st.session_state['ride_data_df'] = df
            success = st.session_state['ride_summary'] is not None # Success requires summary
            if not success:
                 logger.warning(f"Data loaded for {ride_id}, but summary missing.")
        else:
            logger.error(f"Failed DF load for ride {ride_id} or DF empty.")
            st.session_state['upload_status_message'] = f"Could not load data file for ID {ride_id}."
            st.session_state['upload_status_type'] = "error"
            st.session_state['ride_data_df'] = None
            success = False
    else: # Clearing selection
        st.session_state.update({ 'selected_ride_id': None, 'ride_data_df': None, 'ride_summary': None})
        success = True
    return success

def process_uploaded_file():
    """Callback function for file uploader. Parses, saves, and loads new ride."""
    uploaded_file_object = st.session_state.get('fit_uploader')
    if uploaded_file_object is not None:
        filename = uploaded_file_object.name
        logger.info(f"Processing uploaded file: {filename}")
        st.session_state['upload_status_message'], st.session_state['upload_status_type'] = None, None
        file_buffer = None
        try:
            file_content = uploaded_file_object.getvalue()
            file_buffer = io.BytesIO(file_content)

            # Parse the file
            with st.spinner(f"Parsing {filename}..."):
                 df, summary = fit_parser.parse_fit_file(file_buffer)

            # Check if parsing was successful
            if df is not None and not df.empty and summary is not None:
                summary['filename'] = filename
                # Save to database
                # ** FIXED: Indented block for with statement **
                with st.spinner(f"Saving {filename} to database..."):
                    new_ride_id = database.add_ride(summary, df)
                    if new_ride_id:
                        st.session_state['upload_status_message'] = f"Ride '{filename}' saved (ID: {new_ride_id}). Loading..."
                        st.session_state['upload_status_type'] = "success"
                        # Load data immediately after saving
                        load_ride_data(new_ride_id)
                    else:
                        # add_ride handles duplicate/error logging
                        st.session_state['upload_status_message'] = f"Failed to save ride '{filename}' (maybe exists or DB error)."
                        st.session_state['upload_status_type'] = "error"
                # ** End Fix **
            else:
                logger.error(f"Parsing failed for {filename}. DF is None/empty or Summary is None.")
                st.session_state['upload_status_message'] = f"Could not parse data from {filename}. File might be invalid or lack required records."
                st.session_state['upload_status_type'] = "error"
        except Exception as e:
            logger.error(f"Error during file processing callback for {filename}: {e}", exc_info=True)
            st.session_state['upload_status_message'] = f"An unexpected error occurred while processing {filename}. Check logs."
            st.session_state['upload_status_type'] = "error"
        finally:
            if file_buffer: file_buffer.close(); logger.debug("Closed file buffer.")

# --- Sidebar ---
st.sidebar.header("Upload & Select Ride")
st.sidebar.file_uploader(
    "Upload a new .fit file", type=["fit"], key="fit_uploader",
    on_change=process_uploaded_file, accept_multiple_files=False
)
# Display status messages
if st.session_state.get('upload_status_message'):
    status_type = st.session_state.get('upload_status_type', 'info')
    msg = st.session_state['upload_status_message']
    # Use multiline if/elif/else for clarity
    if status_type == "success":
        st.sidebar.success(msg)
    elif status_type == "error":
        st.sidebar.error(msg)
    else:
        st.sidebar.info(msg)
    st.session_state['upload_status_message'] = None

st.sidebar.markdown("---")

# Ride Selection Section
past_rides = database.get_rides()
ride_options = {
    f"{pd.to_datetime(ride['start_time']).strftime('%Y-%m-%d %H:%M')} - {ride['filename']} (ID: {ride['id']})": ride['id']
    for ride in past_rides
}
st.sidebar.header("View Past Ride")

if not ride_options:
    st.sidebar.info("No past rides found.")
else:
    options_list = list(ride_options.keys())
    current_id = st.session_state.get('selected_ride_id')
    id_to_option_name = {v: k for k, v in ride_options.items()}
    current_display_name = id_to_option_name.get(current_id)
    try: current_index = options_list.index(current_display_name) if current_display_name else None
    except ValueError: current_index = None

    selected_ride_display_name = st.sidebar.selectbox(
        "Choose a ride:", options=options_list, index=current_index, key="ride_selector", placeholder="Select a ride..."
    )
    newly_selected_ride_id = ride_options.get(selected_ride_display_name)

    # Handle selection change
    if newly_selected_ride_id is not None and newly_selected_ride_id != current_id:
        logger.info(f"Ride selected: ID {newly_selected_ride_id}")
        if not load_ride_data(newly_selected_ride_id):
            st.sidebar.error("Failed to fully load selected ride.") # Feedback if needed
        st.rerun()

# Buttons only if ride selected
if st.session_state.get('selected_ride_id') is not None:
    # Clear Selection Button
    if st.sidebar.button("Clear Selection / View Welcome"):
        st.session_state.update({'selected_ride_id': None, 'ride_data_df': None, 'ride_summary': None, 'confirm_delete': False})
        st.rerun()

    # Delete Ride Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Manage Ride")
    ride_id_to_delete = st.session_state.selected_ride_id
    name = f"ID {ride_id_to_delete}"
    summary_del = st.session_state.get('ride_summary')
    name = summary_del.get('filename', name) if summary_del else name

    if st.sidebar.button(f"🗑️ Delete Ride: {name}", type="primary"):
        st.session_state.confirm_delete = True

    # Confirmation Dialog
    if st.session_state.get('confirm_delete', False):
        st.sidebar.warning(f"**Delete '{name}'?**")
        col1, col2 = st.sidebar.columns(2)
        # Corrected: Logic inside "Yes, Delete" button's if block
        if col1.button("Yes, Delete", key="confirm_del_yes"):
            # ** FIXED: Indented block for with statement **
            with st.spinner(f"Deleting {name}..."):
                success = database.delete_ride(ride_id_to_delete)
                st.session_state.upload_status_message = f"'{name}' deleted." if success else f"Failed to delete '{name}'."
                st.session_state.upload_status_type = 'success' if success else 'error'
                # Reset state after action
                st.session_state.update({'selected_ride_id': None, 'ride_data_df': None, 'ride_summary': None, 'confirm_delete': False})
            # ** End Fix **
            time.sleep(0.5)
            st.rerun()
        # Corrected: Logic inside "Cancel" button's if block
        if col2.button("Cancel", key="confirm_del_cancel"):
             st.session_state.confirm_delete = False
             st.rerun()

# --- Main Area ---
current_ride_id = st.session_state.get('selected_ride_id')

if current_ride_id is None:
    # Welcome screen
    st.markdown("## Welcome to the FIT File Analyzer!")
    st.markdown("Use the sidebar to upload or select a ride.")
    # Show residual status message
    if st.session_state.get('upload_status_message'):
        status_type = st.session_state.get('upload_status_type', 'info'); msg = st.session_state['upload_status_message']
        if status_type == "success": st.success(msg)
        elif status_type == "error": st.error(msg)
        else: st.info(msg)
        st.session_state['upload_status_message'] = None # Clear message
else:
    # Displaying selected ride
    df = st.session_state.get('ride_data_df')
    summary = st.session_state.get('ride_summary')
    ride_id = current_ride_id

    # Debug logging removed for cleanup

    # Status message from load/etc.
    if st.session_state.get('upload_status_message'):
        status_type = st.session_state.get('upload_status_type', 'info'); msg = st.session_state['upload_status_message'];
        if status_type == "success": st.success(msg)
        elif status_type == "error": st.error(msg)
        else: st.info(msg)
        st.session_state['upload_status_message'] = None

    # Display Header
    display_filename = f"Ride ID {ride_id}"; start_time_str = "N/A"
    if summary:
        display_filename = summary.get('filename', display_filename)
        start_time_val = summary.get('start_time')
        # Corrected: Safely format start time
        try:
            if start_time_val is not None:
                start_time_dt = pd.to_datetime(start_time_val)
                if pd.notna(start_time_dt):
                    start_time_str = start_time_dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.warning(f"Could not format start_time '{start_time_val}': {e}")
            start_time_str = "N/A" # Ensure stays N/A if formatting fails
    st.header(f"Ride Details: {display_filename}")
    st.subheader(f"Started: {start_time_str}")

    # Display Metrics
    summary_data = summary if summary else {}
    st.subheader("Overall Stats")
    col1, col2, col3, col4 = st.columns(4)
    # Col 1
    with col1:
        st.metric("Total Duration", format_duration(summary_data.get('total_time_seconds')))
        st.metric("Moving Time", format_duration(summary_data.get('moving_time_seconds')))
    # Col 2
    with col2:
        dist_km = summary_data.get('total_distance_km'); dist_mi = dist_km * KM_TO_MILES if dist_km is not None else None
        st.metric("Total Distance", f"{dist_mi:.2f} mi" if dist_mi is not None else 'N/A')
        gain_m = summary_data.get('total_elevation_gain_m'); gain_ft = gain_m * METERS_TO_FEET if gain_m is not None else None
        st.metric("Total Ascent", f"{gain_ft:.0f} ft" if gain_ft is not None else 'N/A')
    # Col 3 (Fixed Max Speed calculation structure)
    with col3:
        spd_kmh = summary_data.get('avg_speed_kmh')
        spd_mph = spd_kmh * KM_TO_MILES if spd_kmh is not None else None
        st.metric("Avg Speed", f"{spd_mph:.1f} mph" if spd_mph is not None else 'N/A')
        # Calculate Max Speed separately
        max_spd_mph_str = "N/A"
        if df is not None and 'speed_kmh' in df and df['speed_kmh'].notna().any():
            try:
                max_spd_kmh = df['speed_kmh'].max()
                max_spd_mph = max_spd_kmh * KM_TO_MILES
                max_spd_mph_str = f"{max_spd_mph:.1f} mph"
            except Exception as e:
                logger.error(f"Error calculating Max Speed: {e}", exc_info=True)
                max_spd_mph_str="Error"
        st.metric("Max Speed", max_spd_mph_str)
    # Col 4
    with col4:
         st.metric("Est. Calories", "N/A") # Placeholder

    st.subheader("Performance")
    col1a, col2a, col3a, col4a = st.columns(4)
    # Separated metrics for readability
    with col1a:
        avg_hr = summary_data.get('avg_heart_rate'); st.metric("Avg HR", f"{avg_hr:.0f} bpm" if avg_hr is not None else 'N/A')
        max_hr = summary_data.get('max_hr'); st.metric("Max HR", f"{max_hr:.0f} bpm" if max_hr is not None else 'N/A')
    with col2a:
        avg_cad = summary_data.get('avg_cadence'); st.metric("Avg Cad", f"{avg_cad:.0f} rpm" if avg_cad is not None else 'N/A')
        max_cad = summary_data.get('max_cadence'); st.metric("Max Cad", f"{max_cad:.0f} rpm" if max_cad is not None else 'N/A')
    with col3a:
        avg_pwr = summary_data.get('avg_power'); st.metric("Avg Power", f"{avg_pwr:.0f} W" if avg_pwr is not None else 'N/A')
        max_pwr = summary_data.get('max_power'); st.metric("Max Power", f"{max_pwr:.0f} W" if max_pwr is not None else 'N/A')
    with col4a:
        avg_tc = summary_data.get('avg_temp_c'); avg_tf = C_to_F(avg_tc); st.metric("Avg Temp", f"{avg_tf:.0f} °F" if avg_tf is not None else 'N/A')
        min_tc = summary_data.get('min_temp_c'); max_tc = summary_data.get('max_temp_c'); min_tf = C_to_F(min_tc); max_tf = C_to_F(max_tc); temp_range_str = f"{min_tf:.0f}-{max_tf:.0f}°F" if min_tf is not None and max_tf is not None else "N/A"; st.metric("Temp Range", temp_range_str)

    st.markdown("---")

    # Error/Warning display if DF is missing but summary exists
    if df is None and summary is not None: st.warning(f"Data missing for {ride_id}.")
    # Error if both missing
    elif df is None and summary is None: st.error(f"Load failed for {ride_id}.")
    # --- Tabs Display --- Corrected Syntax/Structure ---
    elif df is not None and not df.empty: # Check df is not None AND not empty
        tab_map, tab_plots, tab_data = st.tabs(["🗺️ Route Map", "📊 Data Plots", "🗂️ Raw Data"])
        # Map Tab
        with tab_map:
            st.subheader("Route Map")
            map_df = df[['latitude', 'longitude']].dropna()
            if len(map_df) >= 2:
                 with st.spinner("Generating map..."):
                     route_map = plotting.plot_route_map(map_df)
                 # Display after spinner context ends
                 if route_map: st_folium(route_map, height=500, use_container_width=True, key=f"folium_map_{ride_id}")
                 else: st.warning("Map generation failed.")
            else:
                 st.warning("No GPS data.")
        # Plots Tab
        with tab_plots:
            st.subheader("Data Plots")
            available_cols = [col for col in PLOT_DISPLAY_MAPPING.keys() if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any()]
            plot_options = [PLOT_DISPLAY_MAPPING[col] for col in available_cols if col in PLOT_DISPLAY_MAPPING]

            if not plot_options:
                 st.warning("No plottable data.")
            else:
                 selected_display = st.multiselect(
                    "Select data:", options=plot_options,
                    default=plot_options[:min(len(plot_options), 3)], key=f"plot_selector_{ride_id}"
                 )
                 if selected_display:
                     inv_map = {v: k for k, v in PLOT_DISPLAY_MAPPING.items()}
                     selected_internal = [inv_map[name] for name in selected_display if name in inv_map]

                     if selected_internal:
                         # Plot generation needs to be indented here
                         with st.spinner("Generating plots..."):
                             x_col = 'timestamp'
                             if x_col not in df.columns:
                                 st.error("No timestamp data available for plots.")
                             else:
                                 conversions = { 'speed_kmh': KM_TO_MILES, 'altitude': METERS_TO_FEET, 'temperature': C_to_F }
                                 active_conv = {k: v for k, v in conversions.items() if k in selected_internal }
                                 reg_cols = ['power', 'heart_rate']
                                 fig = plotting.plot_data(
                                     df, y_vars=selected_internal, x_var=x_col,
                                     display_mapping=PLOT_DISPLAY_MAPPING,
                                     conversions=active_conv, add_regression_for=reg_cols
                                 )
                         # Display after spinner
                         if fig: st.plotly_chart(fig, use_container_width=True)
                         else: st.error("Plot generation failed.")
                     else:
                         # This case handles if mapping failed (unlikely but safe)
                         st.error("Could not map selected plot options.")
                 else:
                      # Case where multiselect is empty
                      st.info("Select one or more data types above to plot.")
        # Data Tab
        with tab_data:
            st.subheader("Raw Data (Metric)")
            st.dataframe(df, use_container_width=True)
            # Corrected try-except for download button
            try:
                # Ensure summary_data is used safely if summary is None
                filename_stem = Path(summary_data.get('filename', f'ride_{ride_id}')).stem
                csv_filename = f"{filename_stem}_data.csv"
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV", csv_data, csv_filename, 'text/csv', key=f"download_csv_{ride_id}"
                )
            except Exception as e:
                 st.warning(f"CSV download error: {e}", exc_info=True)