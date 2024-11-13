import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from scipy.spatial import cKDTree
import os
import re
from pyproj import Proj, transform, Transformer
from pathlib import Path

def find_nc_file(base_path, variable, target_datetime):
    """
    Finds the .nc file that contains data for the target date and time.

    Parameters:
    - base_path: str, the base directory where the data is stored.
    - variable: str, the variable of interest ('SIS', 'SID', 'CAL').
    - target_datetime: str, the target date and time in 'YYYY-MM-DDTHH:MM:SS' format.

    Returns:
    - str, path to the .nc file containing the target date and time data.
    """
    # Convert target datetime to match file naming pattern
    date_str = target_datetime.replace('-', '').replace(':', '')[:8]
    
    # Navigate to the variable directory
    variable_path = os.path.join(base_path, variable)
    
    # Iterate over subdirectories
    for root, dirs, files in os.walk(variable_path):
        for file in files:
            # Check if the file name matches the target date
            if re.search(rf'{variable.lower()}in{date_str}', file.lower()):
                return os.path.join(root, file)
    return None


def find_msgcpp_nc_file(base_path, target_datetime):
    """
    Finds the .nc file that contains data for the target date and time in the SEVIR dataset.

    Parameters:
    - base_path: str, the base directory where the data is stored.
    - target_datetime: str, the target date and time in 'YYYY-MM-DDTHH:MM:SS' format.

    Returns:
    - str, path to the .nc file containing the target date and time data.
    """
    # Convert target datetime to match file naming pattern
    date_str = target_datetime.replace('-', '').replace(':', '')[:8]
    time_str = target_datetime.replace('-', '').replace(':', '')[9:15]
    search_pattern = rf'{date_str}T{time_str}'

    # Iterate over year directories
    for year in ['2021', '2022', '2023']:
        year_path = Path(base_path) / year
        
        # Use glob to find files matching the pattern in the year directory
        for file in year_path.glob(f'**/*{search_pattern}*.nc'):
            return str(file)

    return None


def read_msgcpp_nc_file(base_path, target_datetime):
    """
    Extracts specified variables from the .nc file for the target date and time.

    Parameters:
    - base_path: str, the base directory where the data is stored.
    - target_datetime: str, the target date and time in 'YYYY-MM-DDTHH:MM:SS' format.
    - variables: list of str, names of the variables to extract (default: ['sds', 'sds_cs']).

    Returns:
    - dict, containing the extracted variables.
    """
    # Find the corresponding .nc file
    nc_file = find_msgcpp_nc_file(base_path, target_datetime)
    
    if not nc_file:
        print(f"No file found for {target_datetime}")
        return None
    
    # Load the dataset
    dataset = xr.open_dataset(nc_file)

    # Extract variables
    #extracted_vars = {var: dataset[var] for var in variables if var in dataset}
    
    return dataset


def plot_msgcpp_sds_onetimestep(dataset, lon_slice, lat_slice):
    # Load the dataset
    # Parameters
    ds = dataset
    #variable_name='sds'
    # Create a projection object using the provided proj4_params
    proj_string = '+proj=geos +a=6378.169 +b=6356.584 +h=35785.832 +lat_0=0 +lon_0=0.000000'
    proj = Proj(proj_string)
    transformer = Transformer.from_proj(proj, proj.to_latlong(), always_xy=True)

    # Transform x and y coordinates to longitude and latitude
    x_coords = ds['x'].values
    y_coords = ds['y'].values
    X, Y = np.meshgrid(x_coords, y_coords)
    lon, lat = transformer.transform(X, Y)

    # Flatten the arrays for masking and plotting
    lon_flat = lon.flatten()
    lat_flat = lat.flatten()
    data_flat = ds.values.flatten()

    # Create masks for the specified longitude and latitude ranges
    lon_mask = (lon_flat >= lon_slice[0]) & (lon_flat <= lon_slice[1])
    lat_mask = (lat_flat >= lat_slice[0]) & (lat_flat <= lat_slice[1])
    combined_mask = lon_mask & lat_mask

    # Apply masks
    lon_masked = lon_flat[combined_mask]
    lat_masked = lat_flat[combined_mask]
    data_masked = data_flat[combined_mask]

    # Plot the data
    plt.figure(figsize=(6, 3))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_slice[0], lon_slice[1], lat_slice[0], lat_slice[1]], crs=ccrs.PlateCarree())

    # Add a background map of Denmark
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)

    # Scatter plot of the masked data
    scatter = ax.scatter(lon_masked, lat_masked, c=data_masked, cmap='gist_heat', s=10, transform=ccrs.PlateCarree())
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, orientation='vertical')

    # Add title and labels
    plt.title('MSGCPP SDS')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.show()





def plot_radiation_variable(base_path, variable_name, target_datetime, lon_slice, lat_slice):
    """
    Function to plot the specified radiation variable at a given date and time.

    Parameters:
    - base_path: str, the base directory where the data is stored.
    - variable_name: str, name of the variable to plot (e.g., 'SIS', 'SID', 'CAL').
    - target_datetime: str, the target date and time in 'YYYY-MM-DDTHH:MM:SS' format.
    - lon_slice: tuple, longitude range to slice (e.g., (0, 20)).
    - lat_slice: tuple, latitude range to slice (e.g., (50, 65)).
    """
    # Find the corresponding .nc file
    nc_file = find_nc_file(base_path, variable_name, target_datetime)
    
    if not nc_file:
        print(f"No file found for {target_datetime}")
        return
    
    # Load the dataset
    dataset = xr.open_dataset(nc_file)

    # Select the larger domain including south Sweden, Norway, and north Germany at the specified time
    var = dataset[variable_name].sel(lon=slice(*lon_slice), lat=slice(*lat_slice)).sel(time=target_datetime)

    # Plot the data
    fig, ax = plt.subplots(1, 1, figsize=(6,3), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
    
    var.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='gist_heat')

    # Add coastlines and borders
    ax.coastlines(resolution='10m', color='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, color='lightgray')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Add title
    ax.set_title(f'{variable_name} at {target_datetime}')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()



# Function to load and merge multiple .nc files within a specified date range
def load_and_merge_nc_files(start_date, end_date, base_path, variable_name):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    datasets = []
    
    for date in date_range:
        date_str = date.strftime('%Y%m%d')
        nc_file = find_nc_file(base_path, variable_name, date_str)
        
        if nc_file:
            try:
                ds = xr.open_dataset(nc_file)
                datasets.append(ds)
            except FileNotFoundError:
                print(f"File not found: {nc_file}")
                continue
        else:
            print(f"No file found for {date}")
            continue
    
    if datasets:
        merged_dataset = xr.concat(datasets, dim='time')
        return merged_dataset
    else:
        raise FileNotFoundError("No .nc files found for the specified date range.")


# Function to load and merge multiple .nc files within a specified date range
def load_and_merge_msgcpp_nc_files(base_path, start_datetime, end_datetime, variables=['sds', 'sds_cs']):
    """
    Loads and merges multiple .nc files within a specified date range from the SEVIR dataset.

    Parameters:
    - base_path: str, the base directory where the data is stored.
    - start_datetime: str, the start date and time in 'YYYY-MM-DDTHH:MM:SS' format.
    - end_datetime: str, the end date and time in 'YYYY-MM-DDTHH:MM:SS' format.
    - variables: list of str, names of the variables to extract (default: ['sds', 'sds_cs']).

    Returns:
    - xarray.Dataset, the merged dataset containing the specified variables over the date range.
    """
    date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='15min')  # Assuming files are in 15 min intervals
    datasets = []

    for target_datetime in date_range:
        target_datetime_str = target_datetime.strftime('%Y-%m-%dT%H:%M:%S')
        dataset = read_msgcpp_nc_file(base_path, target_datetime_str)
        
        if dataset:
            try:
                # Select only the specified variables
                selected_data = dataset[variables]
                datasets.append(selected_data)
            except KeyError as e:
                print(f"Variable not found: {e}")
                continue
        else:
            print(f"No file found for {target_datetime_str}")
            continue

    if datasets:
        merged_dataset = xr.concat(datasets, dim='time')
        return merged_dataset
    else:
        raise FileNotFoundError("No .nc files found for the specified date range.")


# Function to get data for a specific station and time window
def load_station_data(station_number, start_date, end_date, path_to_pyranometer_dataset):
    parquet_dataset = pd.read_parquet(path_to_pyranometer_dataset)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filter_window = (parquet_dataset['time'] >= start_date) & (parquet_dataset['time'] <= end_date)
    filter_expr = (parquet_dataset['station_number'] == station_number) & filter_window
    filter_nans = parquet_dataset['O10MGLORAD'] != -99.9
    df_station = parquet_dataset[filter_expr & filter_nans][["time", "O10MGLORAD"]].rename(columns={"O10MGLORAD": "rad10min"})
    return df_station

# Function to find all stations with data within the specified time range
def get_all_stations_with_data(metadata_df, path_to_pyranometer_dataset, start_date, end_date):
    parquet_dataset = pd.read_parquet(path_to_pyranometer_dataset)
    stations_with_data = []
    
    # Filter valid stations based on the intersection of WMO numbers in metadata_df and parquet_dataset
    valid_stations = metadata_df[metadata_df['wmo'].isin(parquet_dataset['station_number'].unique())]
    
    for _, station in valid_stations.iterrows():
        station_number = int(station['wmo'])
        
        # Check if station number has data in the parquet_dataset within the time range
        station_data = parquet_dataset[parquet_dataset['station_number'] == station_number]
        
        if not station_data.empty:
            # Filter station data within the specified time range
            filter_window = (station_data['time'] >= start_date) & (station_data['time'] <= end_date)
            station_data_filtered = station_data[filter_window]
            
            if not station_data_filtered.empty:
                if station['country'] == 'Denmark':
                    stations_with_data.append(station.to_dict())
                
    
    return pd.DataFrame(stations_with_data)



# Plotting function
def plot_sis_time_series_sarah3(station_number, start_date, end_date, latitude, longitude, path_to_pyranometer_dataset, base_nc_path, variable_name):
    end_date = pd.to_datetime(end_date)
    df_day = load_station_data(station_number, start_date, end_date, path_to_pyranometer_dataset)
    dataset = load_and_merge_nc_files(start_date, end_date, base_nc_path, variable_name)
    sis = dataset[variable_name].sel(lon=slice(6, 16), lat=slice(53, 59))

    pyr_lat = latitude
    pyr_lon = longitude
    nearest_pixel = dataset.sel(lat=pyr_lat, lon=pyr_lon, method='nearest')
    time_series = nearest_pixel[variable_name].sel(time=slice(start_date, end_date))

    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days 
    fig, axs = plt.subplots(2, num_days, figsize=(15, 8), subplot_kw={'projection': ccrs.PlateCarree()} if num_days > 1 else None, dpi=300)
    
    if num_days == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    date_range = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq='D')
    for i, date in enumerate(date_range):
        specific_time_index = pd.Timestamp(f"{date.strftime('%Y-%m-%d')}T09:00:00")
        sis_plot = sis.sel(time=specific_time_index).plot(ax=axs[0, i], transform=ccrs.PlateCarree(), cmap='gist_heat', add_colorbar=False)
        axs[0, i].add_feature(cfeature.COASTLINE)
        axs[0, i].add_feature(cfeature.LAND, facecolor='green', edgecolor='none')
        axs[0, i].add_feature(cfeature.OCEAN, facecolor='lightblue')
        axs[0, i].scatter(pyr_lon, pyr_lat, color='green', marker='o', transform=ccrs.PlateCarree())
        axs[0, i].set_title('')

    cbar = fig.colorbar(sis_plot, ax=axs[0, :], orientation='horizontal', aspect=10, shrink=0.2)
    cbar.set_label('SARAH3 SIS [Wm-2]')

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df_day['time'], df_day['rad10min'], label='Pyranometer', marker='.', color='green')
    sarah3_time_series = time_series.sel(time=slice(start_date, end_date))
    sarah3_time_series.plot(ax=ax2, marker='.', linestyle='-', color='tomato', label='SARAH3 SIS')

    for date in date_range:
        specific_time_index = pd.Timestamp(f"{date.strftime('%Y-%m-%d')}T09:00:00")
        ax2.axvline(specific_time_index, color='grey', linestyle='--', linewidth=2, alpha=0.5)

    plt.title('')
    plt.xlabel('Time')
    plt.ylabel('SIS [Wm-2]')
    plt.grid(True)
    plt.legend(loc='upper right', ncol=1)
    plt.show()



def plot_msgcpp_sds(dataset, lon_slice, lat_slice, ax):
    ds = dataset
    proj_string = '+proj=geos +a=6378.169 +b=6356.584 +h=35785.832 +lat_0=0 +lon_0=0.000000'
    proj = Proj(proj_string)
    transformer = Transformer.from_proj(proj, proj.to_latlong(), always_xy=True)

    x_coords = ds['x'].values
    y_coords = ds['y'].values
    X, Y = np.meshgrid(x_coords, y_coords)
    lon, lat = transformer.transform(X, Y)

    lon_flat = lon.flatten()
    lat_flat = lat.flatten()
    data_flat = ds.values.flatten()

    lon_mask = (lon_flat >= lon_slice[0]) & (lon_flat <= lon_slice[1])
    lat_mask = (lat_flat >= lat_slice[0]) & (lat_flat <= lat_slice[1])
    combined_mask = lon_mask & lat_mask

    lon_masked = lon_flat[combined_mask]
    lat_masked = lat_flat[combined_mask]
    data_masked = data_flat[combined_mask]

    ax.set_extent([lon_slice[0], lon_slice[1], lat_slice[0], lat_slice[1]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)

    scatter = ax.scatter(lon_masked, lat_masked, c=data_masked, cmap='gist_heat', s=10, transform=ccrs.PlateCarree())
    return scatter


def plot_sis_time_series(station_number, start_date, end_date, latitude, longitude, path_to_pyranometer_dataset, base_nc_path, variable_name, msgcpp_base_path, msgcpp_variable_name):
    # Load pyranometer data
    end_date = pd.to_datetime(end_date)
    df_day = load_station_data(station_number, start_date, end_date, path_to_pyranometer_dataset)

    # Load SARAH3 dataset
    dataset = load_and_merge_nc_files(start_date, end_date, base_nc_path, variable_name)
    sarah3_sis = dataset[variable_name].sel(lon=slice(6, 16), lat=slice(53, 59))

    # Load MSGCPP dataset
    msgcpp_dataset = load_and_merge_msgcpp_nc_files(msgcpp_base_path, start_date, end_date, variables=[msgcpp_variable_name])
    msgcpp_dataset = msgcpp_dataset.sel(time=~msgcpp_dataset.get_index('time').duplicated())
    msgcpp_sis = msgcpp_dataset[msgcpp_variable_name]

    # Determine the nearest pixel for the pyranometer location in SARAH3
    pyr_lat = latitude
    pyr_lon = longitude
    nearest_pixel_sarah3 = dataset.sel(lat=pyr_lat, lon=pyr_lon, method='nearest')
    sarah3_time_series = nearest_pixel_sarah3[variable_name].sel(time=slice(start_date, end_date))

    # Determine the nearest pixel for the pyranometer location in MSGCPP
    proj_string = '+proj=geos +a=6378.169 +b=6356.584 +h=35785.832 +lat_0=0 +lon_0=0.000000'
    proj = Proj(proj_string)
    transformer = Transformer.from_proj(proj, proj.to_latlong(), always_xy=True)

    x_coords = msgcpp_dataset['x'].values
    y_coords = msgcpp_dataset['y'].values
    X, Y = np.meshgrid(x_coords, y_coords)
    lon, lat = transformer.transform(X, Y)

    # Find the nearest grid point in MSGCPP
    nearest_index = np.argmin((lon - pyr_lon)**2 + (lat - pyr_lat)**2)
    nearest_pixel_msgcpp = msgcpp_sis.isel(x=nearest_index % len(x_coords), y=nearest_index // len(x_coords))
    msgcpp_time_series = nearest_pixel_msgcpp.sel(time=slice(start_date, end_date))

    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days 
    fig, axs = plt.subplots(3, num_days, figsize=(15, 13), subplot_kw={'projection': ccrs.PlateCarree()} if num_days > 1 else None, dpi=300)
    
    
    if num_days == 1:
        axs = np.array([[axs[0]], [axs[1]], [axs[2]]])

    date_range = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1), freq='D')
    for i, date in enumerate(date_range):
        specific_time_index = pd.Timestamp(f"{date.strftime('%Y-%m-%d')}T09:00:00")
        sarah3_plot = sarah3_sis.sel(time=specific_time_index).plot(ax=axs[0, i], transform=ccrs.PlateCarree(), cmap='gist_heat', add_colorbar=False)
        axs[0, i].add_feature(cfeature.COASTLINE)
        axs[0, i].add_feature(cfeature.LAND, facecolor='green', edgecolor='none')
        axs[0, i].add_feature(cfeature.OCEAN, facecolor='lightblue')
        axs[0, i].scatter(pyr_lon, pyr_lat, color='dodgerblue', marker='o', transform=ccrs.PlateCarree())
        axs[0, i].set_title(f'{date.strftime("%Y-%m-%d")} 09:00')

        scatter_msgcpp = plot_msgcpp_sds(msgcpp_dataset.sel(time=specific_time_index, method='nearest').sds, [6, 16], [53, 59], axs[1, i])
        #axs[1, i].set_title(f'{date.strftime("%Y-%m-%d")} 09:00')

    cbar_sarah3 = fig.colorbar(sarah3_plot, ax=axs[0, :], orientation='horizontal', aspect=10, shrink=0.2)
    cbar_sarah3.set_label('SARAH3 SIS [Wm-2]')
    
    cbar_msgcpp = fig.colorbar(scatter_msgcpp, ax=axs[1, :], orientation='horizontal', aspect=10, shrink=0.2)
    cbar_msgcpp.set_label('MSGCPP SDS [Wm-2]')

    ax2 = plt.subplot(3, 1, 3)
    ax2.plot(df_day['time'], df_day['rad10min'], label='Pyranometer', marker='.', color='dodgerblue')
    sarah3_time_series.plot(ax=ax2, marker='.', linestyle='-', color='limegreen', label='SARAH3 SIS', linewidth=0.5)
    msgcpp_time_series.plot(ax=ax2, marker='.', linestyle='-', color='orange', label='MSGCPP SDS', linewidth=0.5)

    plt.title('')
    plt.xlabel('Time')
    plt.ylabel('Surface Radiation [Wm-2]')
    plt.grid(True)
    plt.legend(loc='upper right', ncol=1)
    plt.show()
