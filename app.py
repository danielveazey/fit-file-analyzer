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

# --- Constants for Unit Conversion ---
KM_TO_MILES = 0.621371
METERS_TO_FEET = 3.28084

# --- Mappings for Plottable Variables ---
PLOT_DISPLAY_MAPPING = {
    'speed_kmh':        'Speed (mph)',
    'altitude':         'Elevation (ft)', # Display label for plotting
    'heart_rate':       'Heart Rate (bpm)',
    'cadence':          'Cadence (rpm)',
    'power':            'Power (W)',
    'temperature':      'Temperature (°C)'
}

# --- Page Config ---
st.set_page_config(page_title="FIT File Analyzer", page_icon="🚴", layout="wide")
st.title("🚴 GPS Ride Analyzer (.FIT Files)")

# --- Initialization ---
database.init_db() # Ensure DB exists and has the latest schema

# --- State Management ---
# Initialize session state variables if they don't exist
default_state = {
    'selected_ride_id': None,
    'ride_data_df': None,
    'ride_summary': None,
    'upload_status_message': None,
    'upload_status_type': None,
    'confirm_delete': False
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Functions ---
def format_duration(seconds):
    if seconds is None or pd.isna(seconds): return "N/A"
    try:
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}"
    except Exception: return "N/A"

def load_ride_data(ride_id):
    # (Function as defined in previous response - no change needed here)
    success = False
    if ride_id is not None:
        st.session_state['upload_status_message'] = None
        st.session_state['upload_status_type'] = None
        st.session_state['selected_ride_id'] = ride_id
        st.session_state['ride_summary'] = database.get_ride_summary(ride_id)
        with st.spinner(f"Loading data for Ride ID {ride_id}..."):
            df = database.get_ride_data(ride_id)
            if df is not None:
                logger.info(f"Loaded DataFrame for ride {ride_id}. Shape: {df.shape}")
                if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                     try: df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                     except Exception as e: logger.warning(f"Failed timestamp conversion: {e}")
                st.session_state['ride_data_df'] = df
                success = True
            else:
                 logger.error(f"Failed to load DataFrame for ride {ride_id}.")
                 st.session_state['upload_status_message'] = f"Could not load data file for ride {ride_id}."
                 st.session_state['upload_status_type'] = "error"
                 st.session_state['ride_data_df'] = None
                 st.session_state['ride_summary'] = None # Clear summary too if data load fails
                 success = False
    else: # Clearing selection
        st.session_state['selected_ride_id'] = None
        st.session_state['ride_data_df'] = None
        st.session_state['ride_summary'] = None
        success = True
    return success


def process_uploaded_file():
    # (Function as defined in previous response - no change needed here)
    uploaded_file_object = st.session_state.get('fit_uploader')
    if uploaded_file_object is not None:
        filename = uploaded_file_object.name
        logger.info(f"Processing uploaded file via callback: {filename}")
        st.session_state['upload_status_message'] = None
        st.session_state['upload_status_type'] = None
        file_buffer = None
        try:
            file_content = uploaded_file_object.getvalue()
            file_buffer = io.BytesIO(file_content)
            with st.spinner(f"Parsing {filename}..."):
                df, summary = fit_parser.parse_fit_file(file_buffer)

            if df is not None and not df.empty and summary is not None:
                summary['filename'] = filename
                with st.spinner(f"Saving {filename} to database..."):
                    new_ride_id = database.add_ride(summary, df)
                if new_ride_id:
                    st.session_state['upload_status_message'] = f"Ride '{filename}' saved successfully with ID: {new_ride_id}. Loading..."
                    st.session_state['upload_status_type'] = "success"
                    load_ride_data(new_ride_id) # Trigger load
                else:
                    st.session_state['upload_status_message'] = f"Failed to save ride '{filename}' (may already exist)."
                    st.session_state['upload_status_type'] = "error"
            else:
                st.session_state['upload_status_message'] = f"Could not parse data from {filename}."
                st.session_state['upload_status_type'] = "error"
        except Exception as e:
            logger.error(f"Error during file processing callback: {e}", exc_info=True)
            st.session_state['upload_status_message'] = f"Error processing {filename}."
            st.session_state['upload_status_type'] = "error"
        finally:
            if file_buffer: file_buffer.close()

# --- Sidebar ---
st.sidebar.header("Upload & Select Ride")
st.sidebar.file_uploader("Upload a new .fit file", type=["fit"], key="fit_uploader", on_change=process_uploaded_file)
if st.session_state.get('upload_status_message'):
    status_type = st.session_state.get('upload_status_type', 'info')
    if status_type == "success": st.sidebar.success(st.session_state['upload_status_message'])
    elif status_type == "error": st.sidebar.error(st.session_state['upload_status_message'])
    else: st.sidebar.info(st.session_state['upload_status_message'])
    st.session_state['upload_status_message'] = None # Clear after displaying once

st.sidebar.markdown("---")

past_rides = database.get_rides()
ride_options = { f"{pd.to_datetime(ride['start_time']).strftime('%Y-%m-%d %H:%M')} - {ride['filename']} (ID: {ride['id']})": ride['id'] for ride in past_rides }

st.sidebar.header("View Past Ride")
if not ride_options:
    st.sidebar.info("No past rides found.")
else:
    current_id = st.session_state.get('selected_ride_id')
    options_list = list(ride_options.keys())
    id_to_option_name = {v: k for k, v in ride_options.items()}
    current_display_name = id_to_option_name.get(current_id)
    current_index = options_list.index(current_display_name) if current_display_name in options_list else None

    selected_ride_display_name = st.sidebar.selectbox("Choose a ride:", options=options_list, index=current_index, key="ride_selector", placeholder="Select a ride...")
    newly_selected_ride_id = ride_options.get(selected_ride_display_name)

    if newly_selected_ride_id is not None and newly_selected_ride_id != st.session_state.get('selected_ride_id'):
        logger.info(f"Ride selected via dropdown: ID {newly_selected_ride_id}")
        load_ride_data(newly_selected_ride_id)
        st.rerun() # Ensure main pane updates after selection

    if st.session_state.get('selected_ride_id') is not None:
        if st.sidebar.button("Clear Selection / View Welcome"):
            st.session_state.selected_ride_id = None; st.session_state.ride_data_df = None; st.session_state.ride_summary = None; st.session_state.upload_status_message = None; st.session_state.confirm_delete = False; st.session_state.ride_selector = None;
            st.rerun()

    if st.session_state.get('selected_ride_id') is not None:
        st.sidebar.markdown("---"); st.sidebar.subheader("Manage Ride")
        ride_id = st.session_state.selected_ride_id
        name = f"ID {ride_id}"; summary = st.session_state.get('ride_summary')
        if summary: name = summary.get('filename', name)
        if st.sidebar.button(f"🗑️ Delete Ride: {name}", type="primary"): st.session_state.confirm_delete = True

        if st.session_state.get('confirm_delete', False):
            st.sidebar.warning(f"**Permanently delete '{name}'?**")
            col1, col2 = st.sidebar.columns(2)
            if col1.button("Yes, Delete"):
                with st.spinner(f"Deleting ride {name}..."):
                    success = database.delete_ride(ride_id)
                    st.session_state.upload_status_message = f"Ride '{name}' deleted." if success else f"Failed to delete ride '{name}'."
                    st.session_state.upload_status_type = 'success' if success else 'error'
                st.session_state.selected_ride_id = None; st.session_state.ride_data_df = None; st.session_state.ride_summary = None; st.session_state.confirm_delete = False; st.session_state.ride_selector = None;
                time.sleep(0.5); st.rerun()
            if col2.button("Cancel"): st.session_state.confirm_delete = False; st.rerun()

# --- Main Area ---
current_ride_id = st.session_state.get('selected_ride_id')

if current_ride_id is None:
    st.markdown("## Welcome to the FIT File Analyzer!"); st.markdown("Use the sidebar to upload or select a ride.")
    if st.session_state.get('upload_status_message'): # Show leftover messages on welcome screen
         status_type = st.session_state.get('upload_status_type', 'info')
         if status_type == "success": st.success(st.session_state['upload_status_message'])
         elif status_type == "error": st.error(st.session_state['upload_status_message'])
         else: st.info(st.session_state['upload_status_message'])
         st.session_state['upload_status_message'] = None
else:
    df = st.session_state.get('ride_data_df')
    summary = st.session_state.get('ride_summary')
    ride_id = current_ride_id

    if st.session_state.get('upload_status_message'): # Show load-related messages
        status_type = st.session_state.get('upload_status_type', 'info')
        if status_type == "success": st.success(st.session_state['upload_status_message'])
        elif status_type == "error": st.error(st.session_state['upload_status_message'])
        else: st.info(st.session_state['upload_status_message'])
        st.session_state['upload_status_message'] = None

    if summary:
        st.header(f"Ride Details: {summary.get('filename', f'Ride ID {ride_id}')}")
        start_time_str = pd.to_datetime(summary.get('start_time')).strftime('%Y-%m-%d %H:%M:%S') if summary.get('start_time') else 'N/A'
        st.subheader(f"Started: {start_time_str}")

        col1, col2, col3, col4 = st.columns(4)
        with col1: # Duration, HR
            st.metric("Total Duration", format_duration(summary.get('total_time_seconds')))
            hr = summary.get('avg_heart_rate')
            st.metric("Avg Heart Rate", f"{hr:.1f} bpm" if hr is not None else 'N/A')
        with col2: # Distance, Cadence
            dist_km = summary.get('total_distance_km')
            dist_mi = dist_km * KM_TO_MILES if dist_km is not None else None
            st.metric("Total Distance", f"{dist_mi:.2f} mi" if dist_mi is not None else 'N/A')
            cad = summary.get('avg_cadence')
            st.metric("Avg Cadence", f"{cad:.1f} rpm" if cad is not None else 'N/A')
        with col3: # Avg Speed, Avg Power
            speed_kmh = summary.get('avg_speed_kmh')
            speed_mph = speed_kmh * KM_TO_MILES if speed_kmh is not None else None
            st.metric("Avg Speed", f"{speed_mph:.1f} mph" if speed_mph is not None else 'N/A')
            pwr = summary.get('avg_power')
            st.metric("Avg Power", f"{pwr:.1f} W" if pwr is not None else 'N/A')
        with col4: # Max Speed, Total Ascent
            max_speed_mph_str = "N/A"
            if df is not None and 'speed_kmh' in df.columns and df['speed_kmh'].notna().any():
                max_speed_mph = df['speed_kmh'].max() * KM_TO_MILES
                max_speed_mph_str = f"{max_speed_mph:.1f} mph"
            st.metric("Max Speed", max_speed_mph_str)

            # --- Display Total Ascent ---
            total_gain_ft_str = "N/A"
            gain_m = summary.get('total_elevation_gain_m') # Get value from summary
            if gain_m is not None:
                gain_ft = gain_m * METERS_TO_FEET # Convert to feet
                total_gain_ft_str = f"{gain_ft:.0f} ft" # Display as integer feet
            st.metric("Total Ascent", total_gain_ft_str) # Use new label and value
            # --- End Total Ascent Display ---

        st.markdown("---")

    # Handle cases where loading failed
    if df is None and summary is not None:
        st.warning(f"Detailed data for Ride ID {ride_id} could not be loaded. Only summary shown.")
    elif df is None and summary is None and ride_id is not None: # Check ride_id to ensure it's not the welcome screen
        st.error(f"Could not load any information for Ride ID {ride_id}.")

    # Display tabs only if df is loaded
    if df is not None and not df.empty:
        tab_map, tab_plots, tab_data = st.tabs(["🗺️ Route Map", "📊 Data Plots", "🗂️ Raw Data"])

        with tab_map:
            # (Tab content as before)
            st.subheader("Route Map")
            map_df = df[['latitude', 'longitude']].dropna()
            if len(map_df) >= 2:
                with st.spinner("Generating map..."):
                    route_map = plotting.plot_route_map(map_df)
                    if route_map: st_folium(route_map, height=500, use_container_width=True, key=f"folium_map_{ride_id}")
                    else: st.warning("Could not generate map.")
            else: st.warning("Not enough GPS data for map.")

        with tab_plots:
            # (Tab content as before, using display mapping)
            st.subheader("Data Plots")
            available_internal_cols = [col for col in PLOT_DISPLAY_MAPPING.keys() if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any()]
            plottable_display_options = [PLOT_DISPLAY_MAPPING[col] for col in available_internal_cols if col in PLOT_DISPLAY_MAPPING]

            if not plottable_display_options:
                 st.warning("No plottable data found.")
            else:
                 selected_display_names = st.multiselect("Select data to plot:", options=plottable_display_options, default=plottable_display_options[:min(len(plottable_display_options), 3)], key=f"plot_selector_{ride_id}")
                 if selected_display_names:
                     internal_to_display = {v: k for k, v in PLOT_DISPLAY_MAPPING.items()}
                     selected_internal_cols = [internal_to_display[name] for name in selected_display_names if name in internal_to_display]
                     if selected_internal_cols:
                         with st.spinner("Generating plots..."):
                             x_axis = 'elapsed_time_s' if 'elapsed_time_s' in df.columns else 'timestamp'
                             conversions_for_plot = { 'speed_kmh': KM_TO_MILES, 'altitude': METERS_TO_FEET }
                             active_conversions = { k: v for k, v in conversions_for_plot.items() if k in selected_internal_cols }
                             fig = plotting.plot_data(df, y_vars=selected_internal_cols, x_var=x_axis, display_mapping=PLOT_DISPLAY_MAPPING, conversions=active_conversions)
                             if fig: st.plotly_chart(fig, use_container_width=True)
                             else: st.error("Could not generate plot.")
                     else: st.error("Failed to map selected options.")
                 else: st.info("Select data types to plot.")

        with tab_data:
             # (Tab content as before)
            st.subheader("Raw Data Table (Metric Units)")
            st.dataframe(df, use_container_width=True)
            try:
                csv = df.to_csv(index=False).encode('utf-8')
                fname = f"ride_data_{ride_id}_{Path(summary.get('filename', '')).stem if summary else ''}"
                st.download_button("Download Data as CSV", csv, f"{fname}.csv", 'text/csv', key=f"download_csv_{ride_id}")
            except Exception as e: st.warning(f"Could not generate CSV: {e}")