# plotting.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import pandas as pd
import logging
from scipy.stats import linregress
import numpy as np
import pytz # Added pytz

logger = logging.getLogger(__name__)

# Function to plot time series data
def plot_data(df, y_vars, x_var='timestamp', timezone_str=None, display_mapping=None, conversions=None, add_regression_for=None): # Added timezone_str
    """Creates interactive Plotly chart, converting timestamp x-axis to local timezone if provided."""
    if df is None or df.empty: logger.warning("Plotting: DF empty."); return None
    if not y_vars: logger.warning("Plotting: No y_vars."); return None
    if x_var not in df.columns: logger.warning(f"Plotting: x_var '{x_var}' not found."); return None

    # --- Timezone Handling ---
    x_data_original = df[x_var].copy() # Keep original naive UTC
    x_data = x_data_original # Start with the original
    x_axis_label = "Time" # Default label
    if timezone_str and timezone_str != 'UTC':
         try:
             # Create the aware local time series
             x_data_local = x_data_original.dt.tz_localize('UTC').dt.tz_convert(timezone_str)
             x_data = x_data_local # Use the converted data for plotting

             # Attempt to get timezone abbreviation for label
             try:
                  tz_obj = pytz.timezone(timezone_str)
                  # Use the timezone name/abbreviation at the *first* data point's time
                  tz_abbr = x_data.iloc[0].tzname() if not x_data.empty else timezone_str
                  x_axis_label = f"Time ({tz_abbr})"
             except Exception as tz_name_e:
                  logger.warning(f"Could not get abbreviation for timezone '{timezone_str}': {tz_name_e}")
                  x_axis_label = f"Time ({timezone_str})" # Fallback to full name
         except Exception as tz_conv_e:
             logger.error(f"Failed to convert plot X-axis to timezone '{timezone_str}': {tz_conv_e}. Plotting with UTC.", exc_info=True)
             x_axis_label = "Time (UTC)"
             x_data = x_data_original # *** Fallback to original naive UTC data if conversion fails ***
    else: # No timezone provided or it's UTC
        x_axis_label = "Time (UTC)"
        # x_data is already naive UTC

    # --- Regression Handling ---
    has_elapsed_time = 'elapsed_time_s' in df.columns
    if add_regression_for and not has_elapsed_time:
        logger.warning("Plotting: 'elapsed_time_s' needed for regression but not found. Regression disabled.")
        add_regression_for = []

    # --- Plot Setup ---
    display_mapping = display_mapping if display_mapping is not None else {}
    conversions = conversions if conversions is not None else {}
    add_regression_for = add_regression_for if add_regression_for is not None else []

    num_plots = len(y_vars)
    subplot_titles = [display_mapping.get(var, var.replace('_', ' ').title()) for var in y_vars]
    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, subplot_titles=subplot_titles, vertical_spacing=0.08)

    # --- Plotting Loop ---
    plot_row = 1
    successful_plots = 0
    for y_var_internal in y_vars:
        display_name = display_mapping.get(y_var_internal, y_var_internal.replace('_', ' ').title())

        if y_var_internal not in df.columns:
            logger.warning(f"Plotting: Skipping '{display_name}' (column '{y_var_internal}' not found).")
            fig.add_annotation(text=f"Data N/A for '{display_name}'", row=plot_row, col=1, showarrow=False, yshift=10)
            plot_row += 1
            continue
        if not pd.api.types.is_numeric_dtype(df[y_var_internal]):
             logger.warning(f"Plotting: Skipping '{display_name}' ('{y_var_internal}' not numeric: {df[y_var_internal].dtype}).")
             fig.add_annotation(text=f"Non-numeric data for '{display_name}'", row=plot_row, col=1, showarrow=False, yshift=10)
             plot_row += 1
             continue

        y_data_original = df[y_var_internal].copy()
        y_data_display = y_data_original
        if y_var_internal in conversions:
            conv_action = conversions[y_var_internal]
            try:
                if isinstance(conv_action, (int, float)): y_data_display = y_data_original.astype(float) * conv_action
                elif callable(conv_action): y_data_display = y_data_original.apply(conv_action)
            except Exception as conv_e:
                logger.error(f"Conversion failed for {y_var_internal}: {conv_e}", exc_info=True)
                y_data_display = y_data_original # Revert on error

        if not y_data_display.notna().any():
            logger.warning(f"Plotting: Skipping '{display_name}', no valid data points after conversion.")
            fig.add_annotation(text=f"No Data for '{display_name}'", row=plot_row, col=1, showarrow=False, yshift=10)
            plot_row += 1
            continue

        # Main Trace: Use the (potentially timezone-converted) x_data
        try:
            fig.add_trace(go.Scattergl(x=x_data, y=y_data_display, mode='lines', name=display_name, showlegend=False, connectgaps=False), row=plot_row, col=1)
            successful_plots += 1
        except Exception as trace_e:
             logger.error(f"Failed to add trace for {display_name}: {trace_e}", exc_info=True)
             fig.add_annotation(text=f"Plot error for '{display_name}'", row=plot_row, col=1, showarrow=False, yshift=10)
             plot_row += 1
             continue

        # Regression: Uses elapsed_time_s (numeric) for calc, but plots against converted x_data
        if y_var_internal in add_regression_for and has_elapsed_time:
            # Use the potentially timezone-converted x_data for the display axis ('x_disp')
            reg_df = pd.DataFrame({'x_num': df['elapsed_time_s'], 'y_reg': y_data_display, 'x_disp': x_data}).dropna()
            if len(reg_df) >= 2:
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(reg_df['x_num'], reg_df['y_reg'])
                    if not np.isnan(slope) and not np.isnan(intercept):
                        reg_line_y = intercept + slope * reg_df['x_num']
                        r_sq = r_value**2
                        fig.add_trace(go.Scattergl(x=reg_df['x_disp'], y=reg_line_y, mode='lines', line=dict(color='rgba(255,0,0,0.6)', width=1, dash='dash'), name=f"Trend (R²={r_sq:.2f})", showlegend=False, hoverinfo='skip'), row=plot_row, col=1)
                        logger.info(f"Added regression for {y_var_internal}, R²={r_sq:.3f}")
                except Exception as reg_e: logger.warning(f"Regression failed for {y_var_internal}: {reg_e}", exc_info=True)

        fig.update_yaxes(title_text=display_name, row=plot_row, col=1)
        plot_row += 1

    # --- Final Layout ---
    if successful_plots > 0:
        fig.update_xaxes(title_text=x_axis_label, row=num_plots, col=1) # Use the potentially updated label
        dynamic_height = max(400, num_plots * 200)
        fig.update_layout(height=dynamic_height, title_text="Ride Data Over Time", hovermode='x unified', margin=dict(l=70, r=30, t=60, b=70), uirevision='constant')
    else:
         fig.update_layout(title_text="No Plottable Data Found")

    return fig


# plot_zone_chart remains unchanged
def plot_zone_chart(zone_times_values, zone_labels, title, color_scale='Viridis'):
    if not zone_times_values or not zone_labels or len(zone_times_values) != len(zone_labels):
        logger.error(f"Invalid input for plot_zone_chart: {title}")
        return None
    valid_times = [t for t in zone_times_values if t is not None and pd.notna(t) and t >= 0]
    total_time = sum(valid_times)
    if total_time == 0:
        logger.warning(f"Total time in zones for {title} is zero, percentages not calculated.")
        percentages = [0.0] * len(zone_times_values)
    else:
        percentages = [(t / total_time * 100) if t is not None and pd.notna(t) and t >= 0 else 0.0 for t in zone_times_values]

    def format_time(seconds):
        if seconds is None or pd.isna(seconds) or seconds < 0: return "00:00:00"
        try:
             secs = int(float(seconds))
             hours, rem = divmod(secs, 3600); mins, s = divmod(rem, 60)
             return f"{int(hours):02}:{int(mins):02}:{int(s):02}"
        except (ValueError, TypeError): return "00:00:00"
    display_times = [format_time(t) for t in zone_times_values]
    hover_text = [f"{lbl}: {fmt_t} ({pct:.1f}%)" for lbl, fmt_t, pct in zip(zone_labels, display_times, percentages)]
    text_on_bars = display_times
    plot_values = [(t) if t is not None and pd.notna(t) and t >=0 else 0 for t in zone_times_values]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=zone_labels, x=plot_values, orientation='h', text=text_on_bars, textposition='auto', hovertext=hover_text, hoverinfo='text', marker_color=percentages, marker_colorscale=color_scale, marker_colorbar=dict(title='% of Time')))
    fig.update_layout(title=title, xaxis_title="Time (Seconds)", yaxis_title="Zone", yaxis=dict(autorange="reversed"), height=max(300, len(zone_labels) * 50), margin=dict(l=120, r=30, t=50, b=50), xaxis=dict(range=[0, max(plot_values) * 1.1]))
    return fig

# plot_route_map corrected
def plot_route_map(df):
    """Generates a Folium map showing the ride route."""
    if df is None or df.empty: logger.warning("Map: DataFrame is empty."); return None
    if 'latitude' not in df.columns or 'longitude' not in df.columns: logger.error("Map plotting requires 'latitude' and 'longitude'."); return None
    map_df = df[['latitude', 'longitude']].dropna().copy()
    map_df = map_df[map_df['latitude'].between(-90, 90) & map_df['longitude'].between(-180, 180)]
    if len(map_df) < 2: logger.warning(f"Map: Need >= 2 valid GPS points after range check, found {len(map_df)}."); return None
    try:
        locations = map_df.values.tolist(); avg_lat = map_df['latitude'].mean(); avg_lon = map_df['longitude'].mean()
    except Exception as e: logger.error(f"Map calculation error: {e}", exc_info=True); return None
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles='CartoDB positron')
    if locations:
        try: folium.PolyLine(locations=locations, color='blue', weight=3, opacity=0.7).add_to(m)
        except Exception as e: logger.error(f"Map PolyLine drawing error: {e}")

        # Add Start Marker
        try:
            folium.Marker(location=locations[0], popup='Start', icon=folium.Icon(color='green', icon='play')).add_to(m)
        except IndexError:
            pass # Corrected: pass on its own line
        except Exception as e:
            logger.error(f"Map start marker error: {e}")

        # Add End Marker
        try:
            if locations: # Check again just to be safe
                 folium.Marker(location=locations[-1], popup='End', icon=folium.Icon(color='red', icon='stop')).add_to(m)
        except IndexError:
            pass # Corrected: pass on its own line
        except Exception as e:
            logger.error(f"Map end marker error: {e}")

        # Fit map bounds
        try: m.fit_bounds(folium.PolyLine(locations=locations).get_bounds(), padding=(0.01, 0.01))
        except Exception as e: logger.warning(f"Map fit_bounds error: {e}")

    return m