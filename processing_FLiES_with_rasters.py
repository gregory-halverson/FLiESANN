from os.path import join
from datetime import datetime, date, time
from dateutil import parser
import rasters as rt
from geos5fp import GEOS5FP
from koppengeiger import load_koppen_geiger
from solar_apparent_time import UTC_to_solar
import sun_angles
from FLiES import process_FLiES
from matplotlib.colors import LinearSegmentedColormap

ST_filename = "ECOv002_L2T_LSTE_34366_004_11SPS_20240728T204025_0712_01_LST.tif"
ST_cmap = "bwr"
ST = rt.Raster.open(ST_filename, cmap=ST_cmap)

time_UTC = parser.parse(ST_filename.split("_")[6])
longitude = ST.geometry.centroid_latlon.x
latitude = ST.geometry.centroid_latlon.y
time_solar = UTC_to_solar(time_UTC, longitude)
doy_solar = time_solar.timetuple().tm_yday
hour_of_day_solar = time_solar.hour + time_solar.minute / 60 + time_solar.second / 3600
print(f"{time_UTC:%Y-%m-%d %H:%M:%S} UTC")
print(f"{time_solar:%Y-%m-%d %H:%M:%S} solar apparent time at longitude {longitude}")
print(f"day of year {doy_solar} at longitude {longitude}")
print(f"hour of day {hour_of_day_solar} at longitude {longitude}")

albedo_filename = "ECOv002_L2T_STARS_11SPS_20240728_0712_01_albedo.tif"
albedo_cmap = LinearSegmentedColormap.from_list(name="albedo", colors=["black", "white"])
albedo = rt.Raster.open(albedo_filename, cmap=albedo_cmap)

geos5fp = GEOS5FP(working_directory=join("~", "data", "GEOS5FP"))

AOT = geos5fp.AOT(time_UTC=time_UTC, geometry=albedo.geometry)

COT = geos5fp.COT(time_UTC=time_UTC, geometry=albedo.geometry)

vapor_gccm = geos5fp.vapor_gccm(time_UTC=time_UTC, geometry=albedo.geometry)

ozone_cm = geos5fp.ozone_cm(time_UTC=time_UTC, geometry=albedo.geometry)

kg = load_koppen_geiger(albedo.geometry)

elevation_filename = "ECOv002_L2T_LSTE_34366_004_11SPS_20240728T204025_0712_01_height.tif"
elevation_m = rt.Raster.open(elevation_filename)
elevation_km = elevation_m / 1000
elevation_km.cmap = "viridis"

day_angle_rad = sun_angles.day_angle_rad_from_DOY(doy_solar)
solar_dec_deg = sun_angles.solar_dec_deg_from_day_angle_rad(day_angle_rad)
SZA_deg = sun_angles.SZA_deg_from_lat_dec_hour(ST.geometry.lat, solar_dec_deg, hour_of_day_solar)

Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir, tm, puv, pvis, pnir, fduv, fdvis, fdnir = process_FLiES(
    doy=doy_solar,
    albedo=albedo,
    COT=COT,
    AOT=AOT,
    vapor_gccm=vapor_gccm,
    ozone_cm=ozone_cm,
    elevation_km=elevation_km,
    SZA=SZA_deg,
    KG_climate=kg
)
