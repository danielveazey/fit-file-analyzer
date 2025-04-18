# plotting.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import pandas as pd
import logging
from scipy.stats import linregress # For regression
import numpy as np

logger = logging.getLogger(__name__)

def plot_data(df, y_vars, x_var='timestamp', display_mapping=None, conversions=None, add_regression_for=None):
    """
    Creates interactive Plotly chart with unit conversions and optional regression lines.
    """
    if df is None or df.empty: logger.warning("Plotting: DF empty."); return None
    if not y_vars: logger.warning("Plotting: No y_vars."); return None
    if x_var not in df.columns: logger.warning(f"Plotting: x_var '{x_var}' not found."); return None
    if add_regression_for and 'elapsed_time_s' not in df.columns:
        logger.warning("Plotting: 'elapsed_time_s' needed for regression but not found. Regression disabled.")
        add_regression_for = []

    display_mapping = display_mapping if display_mapping is not None else {}
    conversions = conversions if conversions is not None else {}
    add_regression_for = add_regression_for if add_regression_for is not None else []

    num_plots = len(y_vars)
    subplot_titles = [display_mapping.get(var, var.replace('_', ' ').title()) for var in y_vars]

    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, subplot_titles=subplot_titles, vertical_spacing=0.05)

    plot_row = 1
    for y_var_internal in y_vars:
        display_name = display_mapping.get(y_var_internal, y_var_internal.replace('_', ' ').title())

        if y_var_internal not in df.columns:
            logger.warning(f"Plotting: Skipping '{display_name}', col '{y_var_internal}' not found.")
            # Optionally add annotation for missing data
            fig.add_annotation(text=f"Data not available", showarrow=False, row=plot_row, col=1)
            plot_row += 1; continue

        y_data_original = df[y_var_internal]

        # Apply Conversion
        y_data_display = y_data_original # Default to original
        if y_var_internal in conversions:
            conv_action = conversions[y_var_internal]
            if callable(conv_action): y_data_display = y_data_original.apply(conv_action)
            else: y_data_display = y_data_original.astype(float) * conv_action

        if not y_data_display.notna().any():
            logger.warning(f"Plotting: Skipping '{display_name}', no valid data points after conversion.")
            fig.add_annotation(text=f"No valid data", showarrow=False, row=plot_row, col=1)
            plot_row += 1; continue

        # Main Data Trace
        fig.add_trace(go.Scattergl(x=df[x_var], y=y_data_display, mode='lines', name=display_name, showlegend=False), row=plot_row, col=1)

        # Add Regression Line
        if y_var_internal in add_regression_for and 'elapsed_time_s' in df.columns:
            # Ensure data exists before attempting regression
            reg_df = pd.DataFrame({'x_numeric': df['elapsed_time_s'], 'y_for_reg': y_data_display, 'x_display': df[x_var]}).dropna()
            if len(reg_df) >= 2:
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(reg_df['x_numeric'], reg_df['y_for_reg'])
                    if not np.isnan(slope) and not np.isnan(intercept): # Check for valid result
                        reg_line_y = intercept + slope * reg_df['x_numeric']; r_squared = r_value**2
                        fig.add_trace(go.Scattergl(
                            x=reg_df['x_display'], y=reg_line_y, mode='lines',
                            line=dict(color='rgba(255,0,0,0.6)', width=1.5, dash='dash'),
                            name=f"Trend (R²={r_squared:.2f})", showlegend=False), row=plot_row, col=1)
                        logger.info(f"Added regression for {y_var_internal}: R²={r_squared:.3f}")
                    else: logger.warning(f"Linregress resulted in NaN slope/intercept for {y_var_internal}")
                except Exception as reg_e: logger.warning(f"Could not add regression for {y_var_internal}: {reg_e}", exc_info=True)
            else: logger.warning(f"Skipping regression for {y_var_internal}: not enough points ({len(reg_df)})")

        fig.update_yaxes(title_text=display_name, row=plot_row, col=1)
        plot_row += 1

    # Update Axes and Layout
    x_axis_label = "Time" if x_var == 'timestamp' else x_var.replace('_', ' ').title()
    fig.update_xaxes(title_text=x_axis_label, row=num_plots, col=1) # Ensure bottom axis is labeled
    plot_height = max(400, num_plots * 200)
    fig.update_layout(height=plot_height, title_text="Ride Data Analysis", hovermode='x unified', margin=dict(l=60, r=20, t=60, b=60), uirevision='constant')

    return fig


def plot_route_map(df):
    """Creates an interactive Folium map displaying the ride route."""
    if 'latitude' not in df.columns or 'longitude' not in df.columns: logger.error("Map requires lat/lon."); return None
    map_df = df[['latitude', 'longitude']].dropna()
    if len(map_df) < 2: logger.warning("Map: Need >= 2 valid GPS points."); return None
    try: locations = map_df.values.tolist(); avg_lat, avg_lon = map_df['latitude'].mean(), map_df['longitude'].mean();
    except Exception as e: logger.error(f"Map calc error: {e}"); return None

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles='CartoDB positron')
    if locations:
        try: folium.PolyLine(locations=locations, color='blue', weight=3, opacity=0.7).add_to(m)
        except Exception as e: logger.error(f"Map PolyLine error: {e}")
        try: folium.Marker(locations[0], popup='Start', icon=folium.Icon(color='green', icon='play')).add_to(m)
        except IndexError: pass # No first point
        except Exception as e: logger.error(f"Map start marker error: {e}")
        try: folium.Marker(locations[-1], popup='End', icon=folium.Icon(color='red', icon='stop')).add_to(m)
        except IndexError: pass # No last point
        except Exception as e: logger.error(f"Map end marker error: {e}")
        try: m.fit_bounds(folium.PolyLine(locations=locations).get_bounds(), padding=(0.01, 0.01))
        except Exception as e: logger.warning(f"Map fit_bounds error: {e}")
    return m