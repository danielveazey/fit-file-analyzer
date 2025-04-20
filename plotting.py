# plotting.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import pandas as pd
import logging
from scipy.stats import linregress
import numpy as np

logger = logging.getLogger(__name__)

# Function to plot time series data (remains largely unchanged, handles potential non-numeric gracefully)
def plot_data(df, y_vars, x_var='timestamp', display_mapping=None, conversions=None, add_regression_for=None):
    """Creates interactive Plotly chart with unit conversions and optional regression lines."""
    if df is None or df.empty:
        logger.warning("Plotting: DataFrame is empty or None.")
        return None
    if not y_vars:
        logger.warning("Plotting: No y_vars specified.")
        return None
    if x_var not in df.columns:
        logger.warning(f"Plotting: x_var '{x_var}' not found in DataFrame columns.")
        # Optionally return an empty figure or raise error
        # Let's return None for simplicity in Streamlit context
        return None
    if not pd.api.types.is_datetime64_any_dtype(df[x_var]):
        logger.warning(f"Plotting: x_var '{x_var}' is not a datetime type. Plotting might fail or look incorrect.")
        # Attempt conversion or return None? Let's proceed cautiously.
        try:
            df[x_var] = pd.to_datetime(df[x_var])
        except Exception:
            logger.error(f"Failed to convert x_var '{x_var}' to datetime.")
            return None # Cannot plot without a valid time axis

    has_elapsed_time = 'elapsed_time_s' in df.columns
    if add_regression_for and not has_elapsed_time:
        logger.warning("Plotting: 'elapsed_time_s' needed for regression but not found. Regression disabled.")
        add_regression_for = []

    display_mapping = display_mapping if display_mapping is not None else {}
    conversions = conversions if conversions is not None else {}
    add_regression_for = add_regression_for if add_regression_for is not None else []

    num_plots = len(y_vars)
    subplot_titles = [display_mapping.get(var, var.replace('_', ' ').title()) for var in y_vars]
    # Increased vertical spacing slightly
    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, subplot_titles=subplot_titles, vertical_spacing=0.08)

    plot_row = 1
    successful_plots = 0
    for y_var_internal in y_vars:
        display_name = display_mapping.get(y_var_internal, y_var_internal.replace('_', ' ').title())

        if y_var_internal not in df.columns:
            logger.warning(f"Plotting: Skipping '{display_name}' (column '{y_var_internal}' not found).")
            # Add annotation to indicate missing data in the subplot area
            fig.add_annotation(text=f"Data N/A for '{display_name}'", row=plot_row, col=1, showarrow=False, yshift=10)
            plot_row += 1
            continue

        # Check if data is numeric *before* attempting plotting/conversion
        if not pd.api.types.is_numeric_dtype(df[y_var_internal]):
             logger.warning(f"Plotting: Skipping '{display_name}' (column '{y_var_internal}' is not numeric: {df[y_var_internal].dtype}).")
             fig.add_annotation(text=f"Non-numeric data for '{display_name}'", row=plot_row, col=1, showarrow=False, yshift=10)
             plot_row += 1
             continue

        y_data_original = df[y_var_internal].copy() # Work on a copy

        # Apply Conversion if applicable
        y_data_display = y_data_original
        if y_var_internal in conversions:
            conv_action = conversions[y_var_internal]
            try:
                # Ensure data is float before multiplication if conv_action is a number
                if isinstance(conv_action, (int, float)):
                     y_data_display = y_data_original.astype(float) * conv_action
                elif callable(conv_action):
                     y_data_display = y_data_original.apply(conv_action)
                else:
                    logger.warning(f"Invalid conversion action type for {y_var_internal}: {type(conv_action)}. Skipping conversion.")
            except Exception as conv_e:
                logger.error(f"Conversion failed for {y_var_internal}: {conv_e}", exc_info=True)
                # Keep original data or set to NaN? Let's keep original if conversion fails.
                y_data_display = y_data_original # Revert to original on error

        # Check for actual plottable data *after* potential conversion
        if not y_data_display.notna().any():
            logger.warning(f"Plotting: Skipping '{display_name}', no valid data points after conversion.")
            fig.add_annotation(text=f"No Data for '{display_name}'", row=plot_row, col=1, showarrow=False, yshift=10)
            plot_row += 1
            continue

        # Main Trace (using Scattergl for performance)
        try:
            fig.add_trace(go.Scattergl(
                x=df[x_var],
                y=y_data_display,
                mode='lines',
                name=display_name,
                showlegend=False,
                connectgaps=False # Do not connect over NaN gaps
                ), row=plot_row, col=1)
            successful_plots += 1
        except Exception as trace_e:
             logger.error(f"Failed to add trace for {display_name}: {trace_e}", exc_info=True)
             fig.add_annotation(text=f"Plot error for '{display_name}'", row=plot_row, col=1, showarrow=False, yshift=10)
             plot_row += 1
             continue


        # Add Regression Line if requested and possible
        if y_var_internal in add_regression_for and has_elapsed_time:
            # Create temporary numeric representations for regression
            reg_df = pd.DataFrame({
                'x_num': df['elapsed_time_s'],
                'y_reg': y_data_display,
                'x_disp': df[x_var] # Keep original datetime for plotting the line
                }).dropna() # Drop rows where either x or y is NaN for regression

            if len(reg_df) >= 2: # Need at least two points for linear regression
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(reg_df['x_num'], reg_df['y_reg'])

                    # Check if regression results are valid numbers
                    if not np.isnan(slope) and not np.isnan(intercept):
                        # Calculate regression line values using the numeric x-axis
                        reg_line_y = intercept + slope * reg_df['x_num']
                        r_sq = r_value**2

                        # Plot the regression line using the original datetime x-axis ('x_disp')
                        fig.add_trace(go.Scattergl(
                            x=reg_df['x_disp'],
                            y=reg_line_y,
                            mode='lines',
                            line=dict(color='rgba(255,0,0,0.6)', width=1, dash='dash'),
                            name=f"Trend (R²={r_sq:.2f})", # Include R-squared in name/hover
                            showlegend=False, # Keep legends clean
                            hoverinfo='skip' # Don't show hover for trend line itself
                            ), row=plot_row, col=1)
                        logger.info(f"Added regression for {y_var_internal}, R²={r_sq:.3f}")
                    else:
                         logger.warning(f"Regression calculation resulted in NaN for {y_var_internal}.")
                except Exception as reg_e:
                    logger.warning(f"Regression calculation failed for {y_var_internal}: {reg_e}", exc_info=True)

        # Update Y-axis title for the current subplot
        fig.update_yaxes(title_text=display_name, row=plot_row, col=1)
        plot_row += 1

    # Final Layout Adjustments
    if successful_plots == 0:
         logger.warning("Plotting: No data traces were successfully added to the figure.")
         # Return None or an empty figure? Let's return the figure with annotations.
         fig.update_layout(title_text="No Plottable Data Found")

    else:
        x_axis_label = "Time" if x_var == 'timestamp' else x_var.replace('_', ' ').title()
        # Apply x-axis title only to the bottom-most plot
        fig.update_xaxes(title_text=x_axis_label, row=num_plots, col=1)
        # Calculate dynamic height - ensure minimum, increase per plot
        dynamic_height = max(400, num_plots * 200) # Slightly reduced height per plot
        fig.update_layout(
            height=dynamic_height,
            title_text="Ride Data Over Time",
            hovermode='x unified', # Show hover for all subplots at a given x
            margin=dict(l=70, r=30, t=60, b=70), # Adjusted margins
            uirevision='constant' # Preserve zoom/pan on Streamlit rerun
        )

    return fig


# --- Zone Chart Function (remains unchanged) ---
def plot_zone_chart(zone_times_values, zone_labels, title, color_scale='Viridis'):
    """
    Creates a horizontal bar chart for time spent in zones.

    Args:
        zone_times_values (list): List of time values (in seconds) for each zone.
        zone_labels (list): List of labels for each zone.
        title (str): Title for the chart.
        color_scale (str): Plotly discrete color scale name.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object, or None if data invalid.
    """
    if not zone_times_values or not zone_labels or len(zone_times_values) != len(zone_labels):
        logger.error(f"Invalid input for plot_zone_chart: {title}")
        return None

    # Filter out None values before summing and calculating percentages
    valid_times = [t for t in zone_times_values if t is not None and pd.notna(t) and t >= 0]
    total_time = sum(valid_times)

    if total_time == 0:
        logger.warning(f"Total time in zones for {title} is zero, percentages not calculated.")
        percentages = [0.0] * len(zone_times_values)
    else:
        # Ensure we handle None/NaN during percentage calculation as well
        percentages = [(t / total_time * 100) if t is not None and pd.notna(t) and t >= 0 else 0.0 for t in zone_times_values]

    # Format time values for display (HH:MM:SS)
    def format_time(seconds):
        if seconds is None or pd.isna(seconds) or seconds < 0: return "00:00:00"
        try:
             secs = int(float(seconds))
             hours, rem = divmod(secs, 3600)
             mins, s = divmod(rem, 60)
             return f"{int(hours):02}:{int(mins):02}:{int(s):02}"
        except (ValueError, TypeError):
             logger.warning(f"Could not format time value: {seconds}")
             return "00:00:00"

    display_times = [format_time(t) for t in zone_times_values]

    # Prepare hover text and bar text
    hover_text = [f"{lbl}: {fmt_t} ({pct:.1f}%)" for lbl, fmt_t, pct in zip(zone_labels, display_times, percentages)]
    text_on_bars = display_times # Display HH:MM:SS on bars

    # Use the valid_times for plotting bar lengths to avoid errors with None
    plot_values = [(t) if t is not None and pd.notna(t) and t >=0 else 0 for t in zone_times_values]

    fig = go.Figure()

    # Add horizontal bar trace
    fig.add_trace(go.Bar(
        y=zone_labels,          # Zones on Y-axis
        x=plot_values,          # Time (numeric seconds) on X-axis for bar length
        orientation='h',        # Make it horizontal
        text=text_on_bars,      # Show HH:MM:SS inside/outside bars
        textposition='auto',    # Position text automatically
        hovertext=hover_text,   # Custom hover text
        hoverinfo='text',       # Show only custom hover text
        marker_color=percentages, # Color bars based on percentage
        marker_colorscale=color_scale, # Use a color scale
        marker_colorbar=dict(title='% of Time') # Add color bar legend
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (Seconds)",
        yaxis_title="Zone",
        yaxis=dict(autorange="reversed"), # Show Zone 1 at the top
        height=max(300, len(zone_labels) * 50), # Dynamic height
        margin=dict(l=120, r=30, t=50, b=50), # Increased left margin for labels
        xaxis=dict(range=[0, max(plot_values) * 1.1]) # Ensure axis starts at 0 and gives space
    )
    # Optional: Hide X-axis ticks/labels if desired
    # fig.update_xaxes(showticklabels=False)

    return fig

# --- Route Map Function (remains unchanged) ---
def plot_route_map(df):
    """Generates a Folium map showing the ride route."""
    if df is None or df.empty:
        logger.warning("Map: DataFrame is empty.")
        return None
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logger.error("Map plotting requires 'latitude' and 'longitude' columns.")
        return None

    # Drop rows with invalid GPS data specifically for map plotting
    map_df = df[['latitude', 'longitude']].dropna().copy()

    if len(map_df) < 2:
        logger.warning(f"Map: Need at least 2 valid GPS points, found {len(map_df)}.")
        return None

    try:
        # Ensure coordinates are within valid ranges (belt-and-braces check)
        map_df = map_df[map_df['latitude'].between(-90, 90) & map_df['longitude'].between(-180, 180)]
        if len(map_df) < 2:
            logger.warning(f"Map: Need at least 2 valid GPS points after range check, found {len(map_df)}.")
            return None

        locations = map_df.values.tolist()
        # Calculate center for initial map view
        avg_lat = map_df['latitude'].mean()
        avg_lon = map_df['longitude'].mean()

    except Exception as e:
        logger.error(f"Map calculation error: {e}", exc_info=True)
        return None

    # Create Folium map centered on the average location
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles='CartoDB positron')

    # Add the route as a PolyLine
    if locations:
        try:
            folium.PolyLine(locations=locations, color='blue', weight=3, opacity=0.7).add_to(m)
        except Exception as e:
            logger.error(f"Map PolyLine drawing error: {e}", exc_info=True)
            # Don't necessarily fail the whole map if polyline fails

        # Add Start Marker
        try:
            folium.Marker(
                location=locations[0],
                popup='Start',
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
        except IndexError: pass # No locations if list is empty (should be caught earlier)
        except Exception as e: logger.error(f"Map start marker error: {e}")

        # Add End Marker
        try:
             # Ensure there's at least one point before trying to access index -1
            if locations:
                 folium.Marker(
                     location=locations[-1],
                     popup='End',
                     icon=folium.Icon(color='red', icon='stop')
                 ).add_to(m)
        except IndexError: pass
        except Exception as e: logger.error(f"Map end marker error: {e}")

        # Fit map bounds to the route
        try:
            # Use the PolyLine object itself to get bounds
            route_line = folium.PolyLine(locations=locations)
            m.fit_bounds(route_line.get_bounds(), padding=(0.01, 0.01))
        except Exception as e:
            # fit_bounds can sometimes fail with odd data, log warning but don't crash
            logger.warning(f"Map fit_bounds error: {e}. Map may not be zoomed correctly.")

    return m
