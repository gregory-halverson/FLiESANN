# %% [markdown]
# # Running FLiES for an ECOSTRESS Scene
# 
# This is an example of running the artificial neural network emulator of the Forest Light Environmental Simulator (FLiES) corresponding to an ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) scene.

# %%
from os.path import join
from datetime import datetime, date, time
from dateutil import parser
import rasters as rt
from geos5fp import GEOS5FP
from koppengeiger import load_koppen_geiger
from solar_apparent_time import UTC_to_solar
import sun_angles
from FLiESANN import FLiESANN
from matplotlib.colors import LinearSegmentedColormap
import logging
logging.disable(logging.CRITICAL)

# %% [markdown]
# Here's an example ECOSTRESS albedo scene.

# %%
albedo_filename = "ECOv002_L2T_STARS_11SPS_20240728_0712_01_albedo.tif"
albedo_cmap = LinearSegmentedColormap.from_list(name="albedo", colors=["black", "white"])
albedo = rt.Raster.open(albedo_filename, cmap=albedo_cmap)
albedo

# %% [markdown]
# Let's get the acquisition time of the scene.

# %%
time_UTC = parser.parse(albedo_filename.split("_")[6])
longitude = albedo.geometry.centroid_latlon.x
latitude = albedo.geometry.centroid_latlon.y
time_solar = UTC_to_solar(time_UTC, longitude)
doy_solar = time_solar.timetuple().tm_yday
hour_of_day_solar = time_solar.hour + time_solar.minute / 60 + time_solar.second / 3600
print(f"{time_UTC:%Y-%m-%d %H:%M:%S} UTC")
print(f"{time_solar:%Y-%m-%d %H:%M:%S} solar apparent time at longitude {longitude}")
print(f"day of year {doy_solar} at longitude {longitude}")
print(f"hour of day {hour_of_day_solar} at longitude {longitude}")


# %%
geometry = albedo.geometry
geometry

# %%
FLiES_results = FLiESANN(
    geometry=geometry,
    time_UTC=time_UTC,
    albedo=albedo
)

# %%
Rg = FLiES_results["Rg"]
Rg.cmap = "bwr"
Rg

# %%



