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
import pytz

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Configuration ---
KM_TO_MILES = 0.621371; METERS_TO_FEET = 3.28084
def C_to_F(temp_c): return (temp_c * 9/5) + 32 if temp_c is not None and pd.notna(temp_c) else None
PLOT_DISPLAY_MAPPING = { 'speed_kmh': 'Speed (mph)', 'altitude': 'Elevation (ft)', 'heart_rate': 'Heart Rate (bpm)', 'cadence': 'Cadence (rpm)', 'power': 'Power (W)', 'temperature': 'Temperature (°F)', 'grade': 'Grade (%)' }
PLOT_CONVERSIONS = { 'speed_kmh': KM_TO_MILES, 'altitude': METERS_TO_FEET, 'temperature': C_to_F }
ADD_REGRESSION_FOR = ['power', 'heart_rate']
HR_ZONE_LABELS = ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6+"]
POWER_ZONE_LABELS = ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6", "Zone 7+"]
NUM_HR_ZONES_TO_SET = database.MAX_HR_ZONES_USER - 1
NUM_PWR_ZONES_TO_SET = database.MAX_PWR_ZONES_USER - 1


# --- Page Config & Initialization ---
st.set_page_config(page_title="FIT File Analyzer", page_icon="🚴", layout="wide")
st.title("🚴 GPS Ride Analyzer (.FIT Files)")
st.markdown("Upload a `.fit` file or select a previously uploaded ride.")
try: database.init_db()
except Exception as db_init_e: st.error(f"DB init failed: {db_init_e}"); st.stop()

# --- State Management ---
default_state = { 'selected_ride_id': None, 'ride_data_df': pd.DataFrame(), 'ride_summary': None, 'upload_status_message': None, 'upload_status_type': None, 'confirm_delete': False, 'user_hr_zones': {}, 'user_pwr_zones': {}}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Utility Functions ---
def format_duration(seconds):
    if seconds is None or pd.isna(seconds): return "N/A"
    try: secs_int = int(float(seconds)); hours, rem = divmod(secs_int, 3600); mins, s = divmod(rem, 60); return f"{int(hours):02}:{int(mins):02}:{int(s):02}"
    except (ValueError, TypeError): return "N/A"

# --- Data Loading & Processing ---
def load_ride_data(ride_id):
    st.session_state['upload_status_message'] = None; st.session_state['confirm_delete'] = False
    if ride_id is None: st.session_state.update({'selected_ride_id': None, 'ride_data_df': pd.DataFrame(), 'ride_summary': None}); logger.info("Cleared selection."); return True

    logger.info(f"Loading ride ID: {ride_id}"); st.session_state['selected_ride_id'] = ride_id
    summary = database.get_ride_summary(ride_id)
    if not summary: logger.error(f"Failed summary load {ride_id}."); st.session_state['upload_status_message'] = f"Error loading summary ID {ride_id}."; st.session_state['upload_status_type'] = "error"; return False
    st.session_state['ride_summary'] = summary; logger.info(f"Summary loaded {ride_id}. TZ: {summary.get('timezone_str')}")

    df = None
    try:
        with st.spinner(f"Loading data file ID {ride_id}..."): df = database.get_ride_data(ride_id)
    except Exception as e: logger.error(f"get_ride_data failed {ride_id}: {e}"); st.session_state['upload_status_message'] = f"Error loading data file ID {ride_id}."; st.session_state['upload_status_type'] = "error"; return False
    if df is None or df.empty: logger.error(f"DF load fail/empty {ride_id}."); st.session_state['upload_status_message'] = f"Error: Data file missing/empty ID {ride_id}."; st.session_state['upload_status_type'] = "error"; st.session_state['ride_data_df'] = pd.DataFrame(); return False

    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        logger.warning(f"Timestamp column in loaded DF for ride {ride_id} not datetime. Attempting conversion.")
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            if df.empty:
                 logger.error(f"All timestamps invalid after conversion for ride {ride_id}.")
                 st.session_state['upload_status_message'] = f"Error: Invalid timestamp data for Ride ID {ride_id}."
                 st.session_state['upload_status_type'] = "error"
                 st.session_state['ride_data_df'] = pd.DataFrame()
                 return False
        except Exception as e:
            logger.error(f"Timestamp conversion failed after loading DF for ride {ride_id}: {e}", exc_info=True)
            st.session_state['upload_status_message'] = f"Error processing timestamps for Ride ID {ride_id}."
            st.session_state['upload_status_type'] = "error"
            st.session_state['ride_data_df'] = pd.DataFrame()
            return False

    st.session_state['ride_data_df'] = df; logger.info(f"Load successful {ride_id}. DF shape: {df.shape}"); return True

def process_uploaded_file():
    uploaded_file_object = st.session_state.get('fit_uploader')
    if uploaded_file_object is None: return
    filename = uploaded_file_object.name; logger.info(f"Processing upload: {filename}")
    st.session_state['upload_status_message'] = f"Processing {filename}..."; st.session_state['upload_status_type'] = "info"
    file_buffer = None
    try:
        file_content = uploaded_file_object.getvalue(); file_buffer = io.BytesIO(file_content)
        user_zones = database.get_user_zones()
        with st.spinner(f"Parsing {filename}..."):
            df, summary = fit_parser.parse_fit_file(file_buffer, user_zones=user_zones)

        if df is None or df.empty or summary is None: logger.error(f"Parse failed: {filename}."); st.session_state['upload_status_message'] = f"Error parsing '{filename}'."; st.session_state['upload_status_type'] = "error"; return

        summary['filename'] = filename; logger.info(f"Parse OK: {filename}. TZ={summary.get('timezone_str')}, DF shape: {df.shape}")
        with st.spinner(f"Saving {filename}..."): new_ride_id = database.add_ride(summary, df)

        if new_ride_id: logger.info(f"Ride '{filename}' saved ID: {new_ride_id}"); st.session_state['upload_status_message'] = f"Ride '{filename}' saved (ID: {new_ride_id})."; st.session_state['upload_status_type'] = "success"; load_ride_data(new_ride_id)
        else: logger.error(f"Failed save '{filename}'."); st.session_state['upload_status_message'] = f"Error saving '{filename}'. Duplicate/DB error?"; st.session_state['upload_status_type'] = "error"
    except Exception as e: logger.error(f"Upload error '{filename}': {e}", exc_info=True); st.session_state['upload_status_message'] = f"Critical error processing '{filename}'."; st.session_state['upload_status_type'] = "error"
    finally:
        if file_buffer: file_buffer.close()

def load_user_zones_into_state():
    """Fetches zones from DB and updates session state."""
    zones = database.get_user_zones()
    st.session_state['user_hr_zones'] = {k + 1: v for k, v in zones.get('hr', {}).items()}
    st.session_state['user_pwr_zones'] = {k + 1: v for k, v in zones.get('power', {}).items()}
    logger.info("User zones loaded into session state.")

# --- Sidebar ---
with st.sidebar:
    st.header("Upload & Select Ride")
    st.file_uploader("Upload new .fit file", type=["fit"], key="fit_uploader", on_change=process_uploaded_file, accept_multiple_files=False, help="Upload FIT file.")
    st.markdown("---")
    st.header("View Past Ride")
    try: past_rides = database.get_rides()
    except Exception as e: st.error(f"Error loading rides: {e}"); past_rides = []

    ride_options = {}
    if past_rides:
        for ride in past_rides:
            ride_id = ride.get('id'); filename = ride.get('filename', f'Ride_{ride_id}'); start_time_val = ride.get('start_time'); timezone_str = ride.get('timezone_str', 'UTC')
            if ride_id is None: continue
            prefix = f"ID {ride_id}"
            try:
                 if start_time_val: naive_utc = pd.to_datetime(start_time_val); local_dt = pytz.utc.localize(naive_utc).astimezone(pytz.timezone(timezone_str)); tz_abbr = local_dt.tzname(); prefix = local_dt.strftime(f'%Y-%m-%d %H:%M {tz_abbr}')
                 display_name = f"{prefix} - {filename}"; ride_options[display_name] = ride_id
            except Exception: display_name = f"Error - {filename} (ID: {ride_id})"; ride_options[display_name] = ride_id

    if not ride_options: st.info("No past rides found.")
    else:
        options_list = sorted(list(ride_options.keys()), reverse=True); current_id = st.session_state.get('selected_ride_id'); id_to_option_name = {v: k for k, v in ride_options.items()}; current_display_name = id_to_option_name.get(current_id)
        try: current_index = options_list.index(current_display_name) if current_display_name in options_list else 0
        except ValueError: current_index = 0
        selected_ride_display_name = st.selectbox("Choose a ride:", options=options_list, index=current_index, key="ride_selector", placeholder="Select ride...")
        newly_selected_ride_id = ride_options.get(selected_ride_display_name)
        if newly_selected_ride_id is not None and newly_selected_ride_id != st.session_state.get('selected_ride_id'):
            logger.info(f"Dropdown selection: ID {newly_selected_ride_id}")
            if load_ride_data(newly_selected_ride_id): st.rerun()
            else: st.error("Failed load.")

    if st.session_state.get('selected_ride_id') is not None:
        st.markdown("---");
        if st.button("Clear Selection / View Welcome", key="clear_selection"): load_ride_data(None); st.rerun()
        st.markdown("---"); st.subheader("Manage Ride")
        ride_id_del = st.session_state.selected_ride_id; summary_del = st.session_state.get('ride_summary'); name_del = f"ID {ride_id_del}";
        if summary_del and summary_del.get('filename'): name_del = f"{summary_del['filename']} (ID: {ride_id_del})"
        if st.button(f"🗑️ Delete Ride: {name_del}", type="primary", key="delete_button"): st.session_state.confirm_delete = True; st.rerun()
        if st.session_state.get('confirm_delete', False):
            st.warning(f"**Delete '{name_del}'?**"); col1, col2 = st.columns(2)
            if col1.button("Yes, Delete", key="confirm_del_yes"):
                logger.info(f"Attempting delete: {ride_id_del}")
                with st.spinner(f"Deleting {name_del}..."): success = database.delete_ride(ride_id_del)
                st.session_state['upload_status_message'] = f"'{name_del}' deleted." if success else f"Failed delete '{name_del}'."; st.session_state['upload_status_type'] = 'success' if success else 'error'
                st.session_state.confirm_delete = False; load_ride_data(None); time.sleep(0.5); st.rerun()
            if col2.button("Cancel", key="confirm_del_cancel"): st.session_state.confirm_delete = False; st.rerun()

    st.markdown("---")
    st.header("Zone Settings")
    if not st.session_state['user_hr_zones'] and not st.session_state['user_pwr_zones']:
         load_user_zones_into_state()

    with st.expander("Edit Heart Rate & Power Zones", expanded=False):
        st.markdown("Enter **UPPER** limit for each zone. Leave blank to clear.")
        temp_hr_zones = {} ; temp_pwr_zones = {}; hr_valid = True; pwr_valid = True; last_hr_val = 0; last_pwr_val = 0

        st.subheader("Heart Rate Zones (bpm)")
        cols_hr = st.columns(NUM_HR_ZONES_TO_SET)
        for i in range(1, NUM_HR_ZONES_TO_SET + 1):
            with cols_hr[i-1]:
                current_val = st.session_state['user_hr_zones'].get(i)
                val = st.number_input(f"Zone {i} Upper", min_value=0, max_value=250, value=current_val, step=1, format="%d", key=f"hr_zone_{i}", help=f"Max BPM for Zone {i}")
                if val is not None:
                     if val <= last_hr_val: st.error(f"Z{i} <= Z{i-1}"); hr_valid = False
                     temp_hr_zones[i] = val; last_hr_val = val
                else: temp_hr_zones[i] = None

        st.subheader("Power Zones (watts)")
        cols_pwr = st.columns(NUM_PWR_ZONES_TO_SET)
        for i in range(1, NUM_PWR_ZONES_TO_SET + 1):
            with cols_pwr[i-1]:
                 current_val = st.session_state['user_pwr_zones'].get(i)
                 val = st.number_input(f"Zone {i} Upper", min_value=0, max_value=2000, value=current_val, step=5, format="%d", key=f"pwr_zone_{i}", help=f"Max Watts for Zone {i}")
                 if val is not None:
                     if val <= last_pwr_val: st.error(f"Z{i} <= Z{i-1}"); pwr_valid = False
                     temp_pwr_zones[i] = val; last_pwr_val = val
                 else: temp_pwr_zones[i] = None

        if st.button("Save Custom Zones", key="save_zones"):
            if not hr_valid or not pwr_valid: st.error("Invalid zones. Ensure zones increase.")
            else:
                with st.spinner("Saving..."): save_success = database.save_user_zones(temp_hr_zones, temp_pwr_zones)
                if save_success:
                    st.success("Zones saved!"); load_user_zones_into_state()
                    if st.session_state.selected_ride_id: load_ride_data(st.session_state.selected_ride_id)
                    time.sleep(0.5); st.rerun()
                else: st.error("Failed save.")

# --- Main Area ---
if st.session_state.get('upload_status_message'):
    msg_type = st.session_state.get('upload_status_type', 'info'); msg_text = st.session_state['upload_status_message']
    if msg_type == "success": st.success(msg_text)
    elif msg_type == "error": st.error(msg_text)
    elif msg_type == "warning": st.warning(msg_text)
    else: st.info(msg_text)
    st.session_state['upload_status_message'] = None; st.session_state['upload_status_type'] = None

current_ride_id = st.session_state.get('selected_ride_id')

if current_ride_id is None:
    st.markdown("## Welcome!")
    st.markdown("Upload a `.fit` file or select a past ride.")
else:
    df = st.session_state.get('ride_data_df'); summary = st.session_state.get('ride_summary')
    if summary is None: st.error(f"Error loading summary ID {current_ride_id}."); st.stop()

    display_filename = summary.get('filename', f"Ride ID {current_ride_id}")
    start_time_local = summary.get('start_time'); timezone_str = summary.get('timezone_str', 'UTC'); start_time_display_str = "N/A"
    if start_time_local and pd.notna(start_time_local):
        try: tz_abbr = start_time_local.tzname() if start_time_local.tzinfo else timezone_str; start_time_display_str = start_time_local.strftime(f'%Y-%m-%d %H:%M:%S ({tz_abbr})')
        except Exception: start_time_display_str = str(start_time_local)
    st.header(f"Ride Details: {display_filename}"); st.subheader(f"Started: {start_time_display_str}")

    st.subheader("Overall Stats"); col1, col2, col3, col4 = st.columns(4)
    summary_data = summary;
    with col1: st.metric("Total Duration", format_duration(summary_data.get('total_time_seconds'))); st.metric("Moving Time", format_duration(summary_data.get('moving_time_seconds')))
    with col2: dist_km = summary_data.get('total_distance_km'); dist_mi = dist_km * KM_TO_MILES if dist_km else None; st.metric("Total Distance", f"{dist_mi:.2f} mi" if dist_mi else 'N/A'); gain_m = summary_data.get('total_elevation_gain_m'); gain_ft = gain_m * METERS_TO_FEET if gain_m else None; st.metric("Total Ascent", f"{gain_ft:.0f} ft" if gain_ft else 'N/A')
    with col3:
        spd_kmh = summary_data.get('avg_speed_kmh'); spd_mph = spd_kmh * KM_TO_MILES if spd_kmh else None; st.metric("Avg Speed", f"{spd_mph:.1f} mph" if spd_mph else 'N/A')
        max_spd_mph_str = "N/A"
        if df is not None and 'speed_kmh' in df.columns and df['speed_kmh'].notna().any():
            try:
                if pd.api.types.is_numeric_dtype(df['speed_kmh']):
                     max_spd_kmh = df['speed_kmh'].max(skipna=True)
                     max_spd_mph = max_spd_kmh * KM_TO_MILES if pd.notna(max_spd_kmh) else None
                     max_spd_mph_str = f"{max_spd_mph:.1f} mph" if max_spd_mph is not None else "N/A"
                else:
                     logger.warning("Max Speed calc skipped: 'speed_kmh' not numeric.")
                     max_spd_mph_str = "Type Error"
            except Exception as e:
                logger.error(f"Error calculating Max Speed: {e}", exc_info=True)
                max_spd_mph_str="Calc Error"
        st.metric("Max Speed", max_spd_mph_str)
    with col4: total_cals = summary_data.get('total_calories'); st.metric("Total Calories", f"{int(total_cals):,}" if total_cals else 'N/A')

    st.subheader("Performance"); col1a, col2a, col3a, col4a = st.columns(4)
    with col1a: avg_hr = summary_data.get('avg_heart_rate'); st.metric("Avg HR", f"{avg_hr:.0f} bpm" if avg_hr else 'N/A'); max_hr = summary_data.get('max_hr'); st.metric("Max HR", f"{max_hr:.0f} bpm" if max_hr else 'N/A')
    with col2a: avg_cad = summary_data.get('avg_cadence'); st.metric("Avg Cad", f"{avg_cad:.0f} rpm" if avg_cad else 'N/A'); max_cad = summary_data.get('max_cadence'); st.metric("Max Cad", f"{max_cad:.0f} rpm" if max_cad else 'N/A')
    with col3a: avg_pwr = summary_data.get('avg_power'); st.metric("Avg Power", f"{avg_pwr:.0f} W" if avg_pwr else 'N/A'); max_pwr = summary_data.get('max_power'); st.metric("Max Power", f"{max_pwr:.0f} W" if max_pwr else 'N/A')
    with col4a: avg_tc = summary_data.get('avg_temp_c'); avg_tf = C_to_F(avg_tc); st.metric("Avg Temp", f"{avg_tf:.0f} °F" if avg_tf else 'N/A'); min_tc = summary_data.get('min_temp_c'); max_tc = summary_data.get('max_temp_c'); min_tf = C_to_F(min_tc); max_tf = C_to_F(max_tc); temp_range_str = f"{min_tf:.0f}°F - {max_tf:.0f}°F" if min_tf and max_tf else "N/A"; st.metric("Temp Range", temp_range_str)
    st.markdown("---")

    if df is None or df.empty: st.warning(f"Ride data file missing/empty for ID {current_ride_id}.")
    else:
        tab_map, tab_plots, tab_zones, tab_data = st.tabs(["🗺️ Route Map", "📊 Data Plots", "⏱️ Zones", "🗂️ Raw Data"])

        with tab_map:
            st.subheader("Route Map")
            if 'latitude' in df.columns and 'longitude' in df.columns and df[['latitude', 'longitude']].dropna().shape[0] >= 2:
                 with st.spinner("Generating map..."): route_map = plotting.plot_route_map(df)
                 if route_map: st_folium(route_map, height=500, use_container_width=True, key=f"folium_map_{current_ride_id}")
                 else: st.warning("Map generation failed.")
            else: st.warning("No valid GPS data for map.")

        with tab_plots:
            st.subheader("Data Plots")
            available_cols = [col for col in PLOT_DISPLAY_MAPPING.keys() if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any()]
            plot_options = [PLOT_DISPLAY_MAPPING[col] for col in available_cols]
            if not plot_options: st.warning("No plottable data.")
            else:
                 default_keys = ['speed_kmh', 'altitude', 'heart_rate', 'power']; default_opts = [PLOT_DISPLAY_MAPPING[k] for k in default_keys if k in available_cols][:4]
                 selected_display = st.multiselect("Select data:", options=plot_options, default=default_opts, key=f"plot_selector_{current_ride_id}")
                 if selected_display:
                     inv_map = {v: k for k, v in PLOT_DISPLAY_MAPPING.items()}; selected_internal = [inv_map[name] for name in selected_display if name in inv_map]
                     if selected_internal:
                         with st.spinner("Generating plots..."):
                             current_timezone = summary.get('timezone_str', 'UTC')
                             fig = plotting.plot_data(df, y_vars=selected_internal, x_var='timestamp', timezone_str=current_timezone, display_mapping=PLOT_DISPLAY_MAPPING, conversions=PLOT_CONVERSIONS, add_regression_for=ADD_REGRESSION_FOR)
                         if fig: st.plotly_chart(fig, use_container_width=True)
                         else: st.error("Plot generation failed.")
                 else: st.info("Select data types to plot.")

        # --- CORRECTED Zones Tab ---
        with tab_zones:
            st.subheader("Time in Zones")

            # Check if summary data exists before trying to access anything from it
            if summary_data:
                # Define sources *inside* the block where summary_data is guaranteed to exist
                hr_source = summary_data.get('hr_zone_source', 'N/A')
                pwr_source = summary_data.get('pwr_zone_source', 'N/A')
                st.caption(f"HR Zones based on: {hr_source} | Power Zones based on: {pwr_source}")

                # --- Heart Rate Zone Display ---
                hr_zone_times = []
                hr_keys_found = False
                for i in range(len(HR_ZONE_LABELS)):
                    key = f'time_in_hr_zone_{i}'
                    time_val = summary_data.get(key)
                    # Check for valid numeric value >= 0
                    if time_val is not None and pd.notna(time_val) and isinstance(time_val, (int, float)) and time_val >= 0:
                        hr_zone_times.append(float(time_val))
                        hr_keys_found = True # Mark if any valid data found
                    else:
                        hr_zone_times.append(0.0) # Use 0.0 for missing/invalid
                # Ensure list length matches labels for plotting
                hr_zone_times = hr_zone_times[:len(HR_ZONE_LABELS)]

                # HR Zone Plotting/Info Logic (Correct Indentation)
                if hr_keys_found and sum(hr_zone_times) > 0: # Check if any data > 0 exists
                    st.markdown("##### Heart Rate Zones")
                    try:
                        hr_fig = plotting.plot_zone_chart(hr_zone_times, HR_ZONE_LABELS, "Time in HR Zones", color_scale='Plasma')
                        if hr_fig:
                            st.plotly_chart(hr_fig, use_container_width=True)
                        else:
                            st.warning("Could not generate HR Zone chart.")
                    except Exception as hr_chart_e:
                        logger.error(f"Error plotting HR zone chart: {hr_chart_e}", exc_info=True)
                        st.error("An error occurred generating the HR Zone chart.")
                elif hr_source != "None": # Check if zones were defined (User/File) but time was zero
                    st.info(f"HR Zone data ({hr_source}) is zero for this ride.")
                else: # No zones defined from either source
                    st.info("Heart Rate Zone data is unavailable for this ride.")
                # End HR Zone Logic

                st.markdown("---") # Separator

                # --- Power Zone Display ---
                pwr_zone_times = []
                pwr_keys_found = False
                for i in range(len(POWER_ZONE_LABELS)):
                     key = f'time_in_pwr_zone_{i}'
                     time_val = summary_data.get(key)
                     if time_val is not None and pd.notna(time_val) and isinstance(time_val, (int, float)) and time_val >= 0:
                         pwr_zone_times.append(float(time_val))
                         pwr_keys_found = True
                     else:
                         pwr_zone_times.append(0.0)
                # Ensure list length matches labels
                pwr_zone_times = pwr_zone_times[:len(POWER_ZONE_LABELS)]

                # Power Zone Plotting/Info Logic (Correct Indentation)
                if pwr_keys_found and sum(pwr_zone_times) > 0: # Check if data > 0 exists
                     st.markdown("##### Power Zones")
                     try:
                         pwr_fig = plotting.plot_zone_chart(pwr_zone_times, POWER_ZONE_LABELS, "Time in Power Zones", color_scale='Cividis')
                         if pwr_fig:
                             st.plotly_chart(pwr_fig, use_container_width=True)
                         else:
                              st.warning("Could not generate Power Zone chart.")
                     except Exception as pwr_chart_e:
                          logger.error(f"Error plotting Power zone chart: {pwr_chart_e}", exc_info=True)
                          st.error("An error occurred generating the Power Zone chart.")
                elif pwr_source != "None": # This elif is now correctly aligned with the 'if' above
                     st.info(f"Power Zone data ({pwr_source}) is zero for this ride.")
                else: # This else is now correctly aligned with the 'if' above
                     st.info("Power Zone data is unavailable for this ride.")
            else:
                 # This case handles where summary_data itself might be falsy (e.g., empty dict)
                 st.error("Summary data missing, cannot display zone information.")
        # --- END CORRECTED Zones Tab ---

        with tab_data:
            st.subheader("Raw Data Record Messages")
            st.markdown("Displaying all columns extracted.")
            st.dataframe(df, use_container_width=True)
            try:
                safe_stem = "ride_data";
                if summary_data and summary_data.get('filename'): safe_stem = "".join(c if c.isalnum() else "_" for c in Path(summary_data['filename']).stem)
                elif current_ride_id: safe_stem = f"ride_{current_ride_id}_data"
                csv_fname = f"{safe_stem}.csv"; csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full Data as CSV", csv_bytes, csv_fname, 'text/csv', key=f"download_csv_{current_ride_id}")
            except Exception as e: logger.error(f"CSV download error: {e}", exc_info=True); st.warning("CSV download error.")