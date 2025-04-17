# plotting.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Conversion factors are now passed in, no need to define them here.

def plot_data(df, y_vars, x_var='elapsed_time_s', display_mapping=None, conversions=None):
    """
    Creates an interactive Plotly chart for selected ride data with unit conversions.

    Args:
        df (pd.DataFrame): DataFrame containing the ride data (in original metric units).
        y_vars (list): List of *internal* column names to plot on the y-axis
                       (e.g., ['speed_kmh', 'altitude']).
        x_var (str): Column name to plot on the x-axis (default: 'elapsed_time_s').
        display_mapping (dict, optional): Mapping from internal column names to
                                          user-facing display names including units
                                          (e.g., {'speed_kmh': 'Speed (mph)'}).
        conversions (dict, optional): Dictionary mapping internal column names requiring
                                      conversion to their multiplication factors
                                      (e.g., {'speed_kmh': 0.621371}).

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object, or None if error.
    """
    if df is None or df.empty:
        logger.warning("Cannot plot: DataFrame is empty or None.")
        return None
    if not y_vars:
        logger.warning("Cannot plot: No variables selected for y-axis.")
        return None
    if x_var not in df.columns:
        logger.warning(f"Cannot plot: x-axis variable '{x_var}' not found in DataFrame.")
        return None

    # Use empty dicts as defaults if None is passed
    display_mapping = display_mapping if display_mapping is not None else {}
    conversions = conversions if conversions is not None else {}

    num_plots = len(y_vars)
    # Prepare subplot titles using the display mapping (fall back to internal name)
    subplot_titles = [display_mapping.get(var, var.replace('_', ' ').title()) for var in y_vars]

    fig = make_subplots(
        rows=num_plots,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05
    )

    plot_row = 1
    for y_var_internal in y_vars: # Iterate through INTERNAL names
        display_name = display_mapping.get(y_var_internal, y_var_internal.replace('_', ' ').title())

        if y_var_internal not in df.columns:
            logger.warning(f"Skipping plot for '{y_var_internal}' ({display_name}): column not found.")
            fig.add_annotation(text=f"Data for '{display_name}' not available",
                               xref="paper", yref="paper",
                               x=0.5, y=(num_plots - plot_row + 0.5)/num_plots,
                               showarrow=False, row=plot_row, col=1)
            plot_row += 1
            continue

        # Get original data series (metric)
        y_data_original = df[y_var_internal]

        # Apply Conversion for display if specified
        if y_var_internal in conversions:
            conversion_factor = conversions[y_var_internal]
            # Important: operate on a copy or ensure dtype is float for multiplication
            y_data_display = y_data_original.astype(float) * conversion_factor
            logger.debug(f"Converting '{y_var_internal}' using factor {conversion_factor} for display as '{display_name}'")
        else:
            y_data_display = y_data_original # No conversion needed
            logger.debug(f"No conversion specified for '{y_var_internal}', displaying as '{display_name}'")

        # Check if data exists after potential conversion/selection
        # Use .notna().any() for better check than isnull().all()
        if not y_data_display.notna().any():
            logger.warning(f"Skipping plot for '{display_name}': No valid data points after conversion.")
            fig.add_annotation(text=f"No valid data for '{display_name}'",
                               xref="paper", yref="paper",
                               x=0.5, y=(num_plots - plot_row + 0.5)/num_plots,
                               showarrow=False, row=plot_row, col=1)
            plot_row +=1
            continue

        # Add trace with CONVERTED data
        fig.add_trace(go.Scattergl( # Use Scattergl for potentially large datasets
            x=df[x_var],
            y=y_data_display, # Use the potentially converted data series for plotting
            mode='lines',
            name=display_name, # Use the display name (incl. units) for hover info
            showlegend=False # Subplot titles are generally sufficient
        ), row=plot_row, col=1)

        # Set Y-axis title using the display name
        fig.update_yaxes(title_text=display_name, row=plot_row, col=1)
        plot_row += 1

    # Update X-axis Label
    x_axis_label = x_var.replace('_', ' ').title()
    if x_var == 'elapsed_time_s':
        x_axis_label = 'Elapsed Time (s)' # More specific label
    fig.update_xaxes(title_text=x_axis_label, row=plot_row - 1, col=1) # Apply to bottom-most axis

    # Dynamic height based on number of plots
    plot_height = max(400, num_plots * 200)
    fig.update_layout(
        height=plot_height,
        title_text="Ride Data Analysis",
        hovermode='x unified', # Shows hover data for all plots at a given x
        margin=dict(l=50, r=20, t=50, b=50), # Adjusted margins for labels
        uirevision='constant' # Persist zoom/pan across updates not changing structure
    )

    return fig


def plot_route_map(df):
    """
    Creates an interactive Folium map displaying the ride route.

    Args:
        df (pd.DataFrame): DataFrame with 'latitude' and 'longitude' columns.

    Returns:
        folium.Map: The Folium map object, or None if insufficient data.
    """
    # Ensure input df has the required columns before dropping NA
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
         logger.error("Map plotting requires 'latitude' and 'longitude' columns.")
         return None

    map_df = df[['latitude', 'longitude']].dropna()

    if len(map_df) < 2:
        logger.warning("Cannot create map: Need at least two valid coordinate points.")
        return None

    # Calculate map center and extract locations safely
    try:
        avg_lat = map_df['latitude'].mean()
        avg_lon = map_df['longitude'].mean()
        locations = map_df[['latitude', 'longitude']].values.tolist()
    except Exception as e:
        logger.error(f"Error calculating map center/bounds: {e}")
        return None

    # Create map centered around the average coordinates
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles='CartoDB positron')

    # Add the route polyline
    try:
         folium.PolyLine(
             locations=locations,
             color='blue',
             weight=3,
             opacity=0.7
         ).add_to(m)
    except Exception as e:
         logger.error(f"Error adding PolyLine to map: {e}")

    # Add start and end markers if locations exist
    if locations:
        try:
            folium.Marker(
                 location=locations[0],
                 popup='Start',
                 icon=folium.Icon(color='green', icon='play')
             ).add_to(m)
            folium.Marker(
                 location=locations[-1],
                 popup='End',
                 icon=folium.Icon(color='red', icon='stop')
             ).add_to(m)
        except Exception as e:
             logger.error(f"Error adding start/end markers: {e}")

    # Fit map bounds to the line if possible
    if locations:
        try:
            m.fit_bounds(folium.PolyLine(locations=locations).get_bounds(), padding=(0.01, 0.01))
        except Exception as e:
            logger.warning(f"Could not fit map bounds automatically: {e}. Map might not be optimally zoomed.")

    return m