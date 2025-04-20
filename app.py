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
import pytz # Import pytz

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and Configuration (unchanged)
KM_TO_MILES = 0.621371
METERS_TO_FEET = 3.28084
def C_to_F(temp_c):
    return (temp_c * 9/5) + 32 if temp_c is not None and pd.notna(temp_c) else None

PLOT_DISPLAY_MAPPING = { 'speed_kmh': 'Speed (mph)', 'altitude': 'Elevation (ft)', 'heart_rate': 'Heart Rate (bpm)', 'cadence': 'Cadence (rpm)', 'power': 'Power (W)', 'temperature': 'Temperature (°F)', 'grade': 'Grade (%)' }
PLOT_CONVERSIONS = { 'speed_kmh': KM_TO_MILES, 'altitude': METERS_TO_FEET, 'temperature': C_to_F }
ADD_REGRESSION_FOR = ['power', 'heart_rate']
HR_ZONE_LABELS = ["Zone 1 (Warmup)", "Zone 2 (Easy)", "Zone 3 (Aerobic)", "Zone 4 (Threshold)", "Zone 5 (Maximal)", "Zone 6 (Anaerobic+)"]
POWER_ZONE_LABELS = ["Zone 1 (Active Rec.)", "Zone 2 (Endurance)", "Zone 3 (Tempo)", "Zone 4 (Threshold)", "Zone 5 (VO2 Max)", "Zone 6 (Anaerobic)", "Zone 7 (Neuromuscular)"]

# Page Config
st.set_page_config(page_title="FIT File Analyzer", page_icon="🚴", layout="wide")
st.title("🚴 GPS Ride Analyzer (.FIT Files)")
st.markdown("Upload a `.fit` file or select a previously uploaded ride from the sidebar.")

# Initialization
try:
    database.init_db()
except Exception as db_init_e:
    st.error(f"Fatal Error: Database initialization failed: {db_init_e}")
    st.stop()

# State Management
default_state = { 'selected_ride_id': None, 'ride_data_df': pd.DataFrame(), 'ride_summary': None, 'upload_status_message': None, 'upload_status_type': None, 'confirm_delete': False }
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# Utility Functions (format_duration unchanged)
def format_duration(seconds):
    if seconds is None or pd.isna(seconds): return "N/A"
    try:
        secs_float = float(seconds);
        if secs_float < 0: return "N/A"
        secs_int = int(secs_float); hours, remainder = divmod(secs_int, 3600); minutes, secs = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}"
    except (ValueError, TypeError) as e: return "N/A"

# Data Loading and Processing Functions
def load_ride_data(ride_id):
    st.session_state['upload_status_message'] = None
    st.session_state['confirm_delete'] = False
    if ride_id is None:
        st.session_state.update({'selected_ride_id': None, 'ride_data_df': pd.DataFrame(), 'ride_summary': None}); logger.info("Cleared ride selection."); return True

    logger.info(f"Attempting to load data for ride ID: {ride_id}")
    st.session_state['selected_ride_id'] = ride_id

    # Retrieve summary (now includes timezone_str and aware local start_time)
    summary = database.get_ride_summary(ride_id)
    if not summary:
        logger.error(f"Failed summary retrieval for {ride_id}.")
        st.session_state['upload_status_message'] = f"Error: Could not load summary for Ride ID {ride_id}."
        st.session_state['upload_status_type'] = "error"
        st.session_state.update({'ride_data_df': pd.DataFrame(), 'ride_summary': None}); return False
    st.session_state['ride_summary'] = summary
    logger.info(f"Summary loaded for ride ID {ride_id}. Timezone: {summary.get('timezone_str', 'N/A')}")

    # Retrieve full DataFrame
    df = None
    try:
        with st.spinner(f"Loading data file for Ride ID {ride_id}..."):
            df = database.get_ride_data(ride_id)
    except Exception as e:
        logger.error(f"Exception during get_ride_data for {ride_id}: {e}", exc_info=True)
        st.session_state['upload_status_message'] = f"Error loading data file for Ride ID {ride_id}."
        st.session_state['upload_status_type'] = "error"
        st.session_state.update({'ride_data_df': pd.DataFrame()}); return False

    if df is None or df.empty:
        logger.error(f"DF load failed/empty for {ride_id}.")
        st.session_state['upload_status_message'] = f"Error: Data file missing or empty for Ride ID {ride_id}."
        st.session_state['upload_status_type'] = "error"
        st.session_state['ride_data_df'] = pd.DataFrame(); return False

    # Post-load checks (Timestamp format)
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        logger.warning(f"Timestamp column in loaded DF for ride {ride_id} not datetime. Attempting conversion.")
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            if df.empty:
                 logger.error(f"All timestamps invalid after conversion for ride {ride_id}.")
                 st.session_state['upload_status_message'] = f"Error: Invalid timestamp data for Ride ID {ride_id}."
                 st.session_state['upload_status_type'] = "error"
                 st.session_state['ride_data_df'] = pd.DataFrame(); return False
        except Exception as e:
            logger.error(f"Timestamp conversion failed after loading DF for ride {ride_id}: {e}")
            st.session_state['upload_status_message'] = f"Error processing timestamps for Ride ID {ride_id}."
            st.session_state['upload_status_type'] = "error"
            st.session_state['ride_data_df'] = pd.DataFrame(); return False

    st.session_state['ride_data_df'] = df
    logger.info(f"Successfully loaded data for ride ID {ride_id}. DF shape: {df.shape}")
    return True

def process_uploaded_file():
    uploaded_file_object = st.session_state.get('fit_uploader')
    if uploaded_file_object is None: return

    filename = uploaded_file_object.name
    logger.info(f"Processing uploaded file: {filename}")
    st.session_state['upload_status_message'] = f"Processing {filename}..."
    st.session_state['upload_status_type'] = "info"
    file_buffer = None
    try:
        file_content = uploaded_file_object.getvalue(); file_buffer = io.BytesIO(file_content)
        with st.spinner(f"Parsing {filename}..."):
            # parse_fit_file now returns df, summary (which includes timezone_str and local start_time)
            df, summary = fit_parser.parse_fit_file(file_buffer)

        if df is None or df.empty or summary is None:
            logger.error(f"Parsing failed or returned no data/summary for {filename}.")
            st.session_state['upload_status_message'] = f"Error: Could not parse '{filename}' or file has no record data."
            st.session_state['upload_status_type'] = "error"; return

        summary['filename'] = filename # Add filename to summary
        logger.info(f"Parsing successful for {filename}. TZ={summary.get('timezone_str')}, DF shape: {df.shape}")

        with st.spinner(f"Saving {filename} data..."):
            # add_ride expects timezone_str and start_time in summary
            new_ride_id = database.add_ride(summary, df)

        if new_ride_id:
            logger.info(f"Ride '{filename}' saved with ID: {new_ride_id}")
            st.session_state['upload_status_message'] = f"Ride '{filename}' saved (ID: {new_ride_id})."
            st.session_state['upload_status_type'] = "success"
            load_ride_data(new_ride_id) # Auto-select
        else:
            logger.error(f"Failed to save ride '{filename}' to database.")
            st.session_state['upload_status_message'] = f"Error saving '{filename}'. Might be duplicate or DB error."
            st.session_state['upload_status_type'] = "error"

    except Exception as e:
        logger.error(f"Error processing upload '{filename}': {e}", exc_info=True)
        st.session_state['upload_status_message'] = f"Critical error processing '{filename}'."; st.session_state['upload_status_type'] = "error"
    finally:
        if file_buffer: file_buffer.close()


# --- Sidebar (Dropdown Generation Adjusted) ---
st.sidebar.header("Upload & Select Ride")
st.sidebar.file_uploader("Upload a new .fit file", type=["fit"], key="fit_uploader", on_change=process_uploaded_file, accept_multiple_files=False, help="Upload a FIT file.")
st.sidebar.markdown("---")
st.sidebar.header("View Past Ride")

try:
    past_rides = database.get_rides()
except Exception as e: st.sidebar.error(f"Error loading past rides: {e}"); past_rides = []

ride_options = {}
if past_rides:
    for ride in past_rides:
        ride_id = ride.get('id')
        if ride_id is None: continue # Skip if ID is missing

        filename = ride.get('filename', f'Ride_{ride_id}')
        start_time_val = ride.get('start_time') # This is the UTC string from DB
        timezone_str = ride.get('timezone_str', 'UTC') # Get timezone

        display_name_prefix = f"ID {ride_id}" # Fallback prefix
        try:
            if start_time_val:
                 # Convert stored UTC string to local time for display in dropdown
                 naive_utc = pd.to_datetime(start_time_val)
                 local_dt = pytz.utc.localize(naive_utc).astimezone(pytz.timezone(timezone_str))
                 tz_abbr = local_dt.tzname() # Get abbreviation
                 display_name_prefix = local_dt.strftime(f'%Y-%m-%d %H:%M {tz_abbr}')
            display_name = f"{display_name_prefix} - {filename}"
            ride_options[display_name] = ride_id
        except Exception as format_e:
             logger.warning(f"Error formatting dropdown name for ride {ride_id}: {format_e}")
             display_name = f"Error - {filename} (ID: {ride_id})"
             ride_options[display_name] = ride_id

if not ride_options:
    st.sidebar.info("No past rides found.")
else:
    options_list = sorted(list(ride_options.keys()), reverse=True)
    current_id = st.session_state.get('selected_ride_id')
    id_to_option_name = {v: k for k, v in ride_options.items()}
    current_display_name = id_to_option_name.get(current_id)
    try: current_index = options_list.index(current_display_name) if current_display_name in options_list else 0
    except ValueError: current_index = 0

    selected_ride_display_name = st.sidebar.selectbox("Choose a ride:", options=options_list, index=current_index, key="ride_selector", placeholder="Select a ride...")
    newly_selected_ride_id = ride_options.get(selected_ride_display_name)
    if newly_selected_ride_id is not None and newly_selected_ride_id != st.session_state.get('selected_ride_id'):
        logger.info(f"Ride selected via dropdown: ID {newly_selected_ride_id}")
        if load_ride_data(newly_selected_ride_id): st.rerun()
        else: st.sidebar.error("Failed to load selected ride data.")

# --- Ride Management (Delete Button - unchanged) ---
if st.session_state.get('selected_ride_id') is not None:
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Selection / View Welcome", key="clear_selection"): load_ride_data(None); st.rerun()
    st.sidebar.markdown("---"); st.sidebar.subheader("Manage Ride")
    ride_id_to_delete = st.session_state.selected_ride_id; summary_for_delete = st.session_state.get('ride_summary')
    ride_name_for_delete = f"ID {ride_id_to_delete}";
    if summary_for_delete and summary_for_delete.get('filename'): ride_name_for_delete = f"{summary_for_delete['filename']} (ID: {ride_id_to_delete})"
    if st.sidebar.button(f"🗑️ Delete Ride: {ride_name_for_delete}", type="primary", key="delete_button"): st.session_state.confirm_delete = True; st.rerun()
    if st.session_state.get('confirm_delete', False):
        st.sidebar.warning(f"**Delete '{ride_name_for_delete}'?**"); col1, col2 = st.sidebar.columns(2)
        if col1.button("Yes, Delete", key="confirm_del_yes"):
            logger.info(f"Attempting delete: {ride_id_to_delete}")
            with st.spinner(f"Deleting {ride_name_for_delete}..."): delete_success = database.delete_ride(ride_id_to_delete)
            st.session_state['upload_status_message'] = f"Ride '{ride_name_for_delete}' deleted." if delete_success else f"Failed delete '{ride_name_for_delete}'."; st.session_state['upload_status_type'] = 'success' if delete_success else 'error'
            st.session_state.confirm_delete = False; load_ride_data(None); time.sleep(0.5); st.rerun()
        if col2.button("Cancel", key="confirm_del_cancel"): st.session_state.confirm_delete = False; st.rerun()


# --- Main Area ---
# Display Status Messages
if st.session_state.get('upload_status_message'):
    status_type = st.session_state.get('upload_status_type', 'info'); msg = st.session_state['upload_status_message']
    if status_type == "success": st.success(msg)
    elif status_type == "error": st.error(msg)
    elif status_type == "warning": st.warning(msg)
    else: st.info(msg)
    st.session_state['upload_status_message'] = None; st.session_state['upload_status_type'] = None

# Main Content Area Logic
current_ride_id = st.session_state.get('selected_ride_id')

if current_ride_id is None:
    st.markdown("## Welcome!")
    st.markdown("Use the sidebar to upload a `.fit` file or select a past ride.")
else:
    df = st.session_state.get('ride_data_df')
    summary = st.session_state.get('ride_summary')
    if summary is None: st.error(f"Error loading summary for Ride ID {current_ride_id}."); st.stop()

    # --- Header Display ---
    display_filename = summary.get('filename', f"Ride ID {current_ride_id}")
    start_time_local = summary.get('start_time') # Should be timezone-aware local time from get_ride_summary
    timezone_str = summary.get('timezone_str', 'UTC')
    start_time_display_str = "N/A"
    if start_time_local and pd.notna(start_time_local):
        try:
            tz_abbr = start_time_local.tzname() # Get abbreviation from the aware object
            start_time_display_str = start_time_local.strftime(f'%Y-%m-%d %H:%M:%S ({tz_abbr})')
        except Exception as e:
             logger.warning(f"Could not format local start_time {start_time_local}: {e}")
             start_time_display_str = str(start_time_local) # Fallback

    st.header(f"Ride Details: {display_filename}")
    st.subheader(f"Started: {start_time_display_str}") # Display local time with TZ

    # --- Metrics Display (unchanged logic, uses summary keys) ---
    st.subheader("Overall Stats"); col1, col2, col3, col4 = st.columns(4)
    summary_data = summary
    with col1: st.metric("Total Duration", format_duration(summary_data.get('total_time_seconds'))); st.metric("Moving Time", format_duration(summary_data.get('moving_time_seconds')))
    with col2: dist_km = summary_data.get('total_distance_km'); dist_mi = dist_km * KM_TO_MILES if dist_km is not None else None; st.metric("Total Distance", f"{dist_mi:.2f} mi" if dist_mi is not None else 'N/A'); gain_m = summary_data.get('total_elevation_gain_m'); gain_ft = gain_m * METERS_TO_FEET if gain_m is not None else None; st.metric("Total Ascent", f"{gain_ft:.0f} ft" if gain_ft is not None else 'N/A')
    with col3:
        spd_kmh = summary_data.get('avg_speed_kmh'); spd_mph = spd_kmh * KM_TO_MILES if spd_kmh is not None else None; st.metric("Avg Speed", f"{spd_mph:.1f} mph" if spd_mph is not None else 'N/A')
        max_spd_mph_str = "N/A"
        if df is not None and 'speed_kmh' in df.columns and df['speed_kmh'].notna().any():
            try:
                 if pd.api.types.is_numeric_dtype(df['speed_kmh']): max_spd_kmh = df['speed_kmh'].max(skipna=True); max_spd_mph = max_spd_kmh * KM_TO_MILES if pd.notna(max_spd_kmh) else None; max_spd_mph_str = f"{max_spd_mph:.1f} mph" if max_spd_mph is not None else "N/A"
                 else: max_spd_mph_str = "Type Error"
            except Exception as e: max_spd_mph_str="Calc Error"
        st.metric("Max Speed", max_spd_mph_str)
    with col4: total_cals = summary_data.get('total_calories'); st.metric("Total Calories", f"{int(total_cals):,}" if total_cals is not None else 'N/A')

    st.subheader("Performance"); col1a, col2a, col3a, col4a = st.columns(4)
    with col1a: avg_hr = summary_data.get('avg_heart_rate'); st.metric("Avg HR", f"{avg_hr:.0f} bpm" if avg_hr is not None else 'N/A'); max_hr = summary_data.get('max_hr'); st.metric("Max HR", f"{max_hr:.0f} bpm" if max_hr is not None else 'N/A')
    with col2a: avg_cad = summary_data.get('avg_cadence'); st.metric("Avg Cad", f"{avg_cad:.0f} rpm" if avg_cad is not None else 'N/A'); max_cad = summary_data.get('max_cadence'); st.metric("Max Cad", f"{max_cad:.0f} rpm" if max_cad is not None else 'N/A')
    with col3a: avg_pwr = summary_data.get('avg_power'); st.metric("Avg Power", f"{avg_pwr:.0f} W" if avg_pwr is not None else 'N/A'); max_pwr = summary_data.get('max_power'); st.metric("Max Power", f"{max_pwr:.0f} W" if max_pwr is not None else 'N/A')
    with col4a: avg_tc = summary_data.get('avg_temp_c'); avg_tf = C_to_F(avg_tc); st.metric("Avg Temp", f"{avg_tf:.0f} °F" if avg_tf is not None else 'N/A'); min_tc = summary_data.get('min_temp_c'); max_tc = summary_data.get('max_temp_c'); min_tf = C_to_F(min_tc); max_tf = C_to_F(max_tc); temp_range_str = f"{min_tf:.0f}°F - {max_tf:.0f}°F" if min_tf is not None and max_tf is not None else "N/A"; st.metric("Temp Range", temp_range_str)
    st.markdown("---")

    # --- Tabs ---
    if df is None or df.empty:
        st.warning(f"Ride data file missing/empty for Ride ID {current_ride_id}.")
    else:
        tab_map, tab_plots, tab_zones, tab_data = st.tabs(["🗺️ Route Map", "📊 Data Plots", "⏱️ Zones", "🗂️ Raw Data"])

        # Map Tab (unchanged call)
        with tab_map:
            st.subheader("Route Map")
            if 'latitude' in df.columns and 'longitude' in df.columns and df[['latitude', 'longitude']].dropna().shape[0] >= 2:
                 with st.spinner("Generating map..."):
                     try: route_map = plotting.plot_route_map(df);
                     except Exception as map_e: st.error(f"Map generation error: {map_e}"); route_map = None
                 if route_map: st_folium(route_map, height=500, use_container_width=True, key=f"folium_map_{current_ride_id}")
                 else: st.warning("Map could not be generated.")
            else: st.warning("No valid GPS data for map.")

        # Plots Tab (pass timezone_str)
        with tab_plots:
            st.subheader("Data Plots")
            available_cols_internal = [col for col in PLOT_DISPLAY_MAPPING.keys() if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any()]
            plot_options_display = [PLOT_DISPLAY_MAPPING[col] for col in available_cols_internal]
            if not plot_options_display: st.warning("No plottable numeric data found.")
            else:
                default_selection_keys = ['speed_kmh', 'altitude', 'heart_rate', 'power']; default_display = [PLOT_DISPLAY_MAPPING[k] for k in default_selection_keys if k in available_cols_internal][:4]
                selected_display = st.multiselect("Select data:", options=plot_options_display, default=default_display, key=f"plot_selector_{current_ride_id}")
                if selected_display:
                    inv_map = {v: k for k, v in PLOT_DISPLAY_MAPPING.items()}; selected_internal = [inv_map[name] for name in selected_display if name in inv_map]
                    if selected_internal:
                        with st.spinner("Generating plots..."):
                            try:
                                # *** Pass the timezone string to plot_data ***
                                current_timezone = summary.get('timezone_str', 'UTC')
                                fig = plotting.plot_data(df, y_vars=selected_internal, x_var='timestamp', timezone_str=current_timezone, display_mapping=PLOT_DISPLAY_MAPPING, conversions=PLOT_CONVERSIONS, add_regression_for=ADD_REGRESSION_FOR)
                                if fig: st.plotly_chart(fig, use_container_width=True)
                                else: st.error("Plot generation failed.")
                            except Exception as plot_e: logger.error(f"Plot gen error: {plot_e}", exc_info=True); st.error("Error generating plots.")
                    else: st.error("Could not map plot options.")
                else: st.info("Select data types to plot.")

        # Zones Tab (unchanged logic)
        with tab_zones:
            st.subheader("Time in Zones")
            if summary_data:
                hr_zone_times = []; hr_keys_found = False
                for i in range(len(HR_ZONE_LABELS)):
                    key = f'time_in_hr_zone_{i}'; time_val = summary_data.get(key)
                    if time_val is not None and pd.notna(time_val) and time_val >= 0: hr_zone_times.append(float(time_val)); hr_keys_found = True
                    else: hr_zone_times.append(0.0)
                if hr_keys_found and sum(hr_zone_times) > 0:
                    current_hr_labels = HR_ZONE_LABELS[:len(hr_zone_times)]
                    st.markdown("##### Heart Rate Zones");
                    try: hr_fig = plotting.plot_zone_chart(hr_zone_times, current_hr_labels, "Time in HR Zones", color_scale='Plasma')
                    except Exception as e: st.error(f"HR Zone plot error: {e}"); hr_fig = None
                    if hr_fig: st.plotly_chart(hr_fig, use_container_width=True)
                    else: st.warning("Could not generate HR Zone chart.")
                else: st.info("Heart Rate Zone data unavailable/zero.")
                st.markdown("---")
                pwr_zone_times = []; pwr_keys_found = False
                for i in range(len(POWER_ZONE_LABELS)):
                     key = f'time_in_pwr_zone_{i}'; time_val = summary_data.get(key)
                     if time_val is not None and pd.notna(time_val) and time_val >= 0: pwr_zone_times.append(float(time_val)); pwr_keys_found = True
                     else: pwr_zone_times.append(0.0)
                if pwr_keys_found and sum(pwr_zone_times) > 0:
                     current_pwr_labels = POWER_ZONE_LABELS[:len(pwr_zone_times)]
                     st.markdown("##### Power Zones");
                     try: pwr_fig = plotting.plot_zone_chart(pwr_zone_times, current_pwr_labels, "Time in Power Zones", color_scale='Cividis')
                     except Exception as e: st.error(f"Power Zone plot error: {e}"); pwr_fig = None
                     if pwr_fig: st.plotly_chart(pwr_fig, use_container_width=True)
                     else: st.warning("Could not generate Power Zone chart.")
                else: st.info("Power Zone data unavailable/zero.")
            else: st.error("Summary data missing.")

        # Data Tab (unchanged logic)
        with tab_data:
            st.subheader("Raw Data Record Messages")
            st.markdown("Displaying all columns extracted from record messages.")
            st.dataframe(df, use_container_width=True)
            try:
                safe_filename_stem = "ride_data";
                if summary_data and summary_data.get('filename'): safe_filename_stem = "".join(c if c.isalnum() else "_" for c in Path(summary_data['filename']).stem)
                elif current_ride_id: safe_filename_stem = f"ride_{current_ride_id}_data"
                csv_filename = f"{safe_filename_stem}.csv"; csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full Data as CSV", csv_data, csv_filename, 'text/csv', key=f"download_csv_{current_ride_id}")
            except Exception as e: logger.error(f"CSV download prep error: {e}", exc_info=True); st.warning("CSV download error.")