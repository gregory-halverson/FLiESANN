# Forest Light Environmental Simulator (FLiES) Radiative Transfer Model Artificial Neural Network (ANN) Implementation in Python

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/FLiESANN.svg)](https://pypi.org/project/FLiESANN/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This package provides an artificial neural network emulator for the **Forest Light Environmental Simulator (FLiES)** radiative transfer model, implemented using Keras and TensorFlow in Python. The FLiESANN model efficiently estimates solar radiation components for ecosystem modeling applications, particularly for the **Breathing Earth Systems Simulator (BESS)** model used to calculate evapotranspiration (ET) and gross primary productivity (GPP) for NASA's **ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS)** and **Surface Biology and Geology (SBG)** thermal remote sensing missions.

## Contributors

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer<br>
NASA Jet Propulsion Laboratory 329G

Hideki Kobayashi (he/him)<br>
FLiES algorithm inventor<br>
Japan Agency for Marine-Earth Science and Technology

## Scientific Background

### Radiative Transfer Physics

The FLiES model simulates the complex interactions between solar radiation and Earth's atmosphere-surface system through sophisticated radiative transfer calculations. The original FLiES model uses computationally intensive Monte Carlo ray-tracing methods to solve the radiative transfer equation. This ANN implementation provides a computationally efficient emulator that maintains high accuracy while enabling large-scale operational applications.

The model computes several key physical processes:

1. **Solar radiation attenuation** through atmospheric absorption and scattering by gases, aerosols, and clouds
2. **Spectral decomposition** of broadband solar radiation into three key components:
   - **UV** (280-400 nm): Ultraviolet radiation
   - **PAR** (400-700 nm): Photosynthetically Active Radiation (visible light)
   - **NIR** (700-3000 nm): Near-infrared radiation
3. **Direct vs. diffuse radiation partitioning** based on atmospheric scattering processes
4. **Atmospheric transmittance** calculations accounting for multiple scattering effects

### Neural Network Architecture

The ANN emulator predicts seven fundamental radiative transfer parameters:

- **Atmospheric transmittance**: Fraction of top-of-atmosphere solar radiation reaching the surface
- **Spectral proportions**: Fractions of surface radiation in UV, PAR, and NIR bands
- **Diffuse fractions**: Proportions of diffuse (scattered) vs. direct radiation for each spectral band

### Input Parameters

The model requires the following atmospheric and surface parameters:

#### Required Parameters
- **`albedo`**: Surface reflectance (0-1)
- **Temporal information**: Either `time_UTC` or (`day_of_year` and `hour_of_day`)
- **Spatial information**: `geometry` (RasterGeometry, Point, or MultiPoint)

#### Optional Parameters (automatically retrieved if not provided)
- **`COT`**: Cloud optical thickness
- **`AOT`**: Aerosol optical thickness  
- **`vapor_gccm`**: Water vapor column (g/cm²)
- **`ozone_cm`**: Total column ozone (cm)
- **`elevation_m`**: Surface elevation (m)
- **`SZA_deg`**: Solar zenith angle (degrees)
- **`KG_climate`**: Köppen-Geiger climate classification

## Installation

### From PyPI (Recommended)

```bash
pip install FLiESANN
```

### From Source

```bash
git clone https://github.com/JPL-Evapotranspiration-Algorithms/FLiESANN.git
cd FLiESANN
pip install -e .
```

### Dependencies

The package requires Python 3.10+ and several scientific computing libraries. Key dependencies include:

- `tensorflow` and `keras` for neural network inference
- `numpy` and `pandas` for numerical computations
- `rasters` for geospatial raster processing
- `GEOS5FP` for atmospheric data retrieval
- `NASADEM` for elevation data
- `koppengeiger` for climate classification

## Usage

### Basic Usage

```python
from FLiESANN import FLiESANN
from datetime import datetime
from shapely.geometry import Point
import numpy as np

# Simple scalar calculation
results = FLiESANN(
    albedo=0.15,                              # Surface albedo
    time_UTC=datetime(2024, 7, 15, 18, 0),    # UTC time
    geometry=Point(-118.0, 34.0),             # Longitude, latitude
)

# Access results
print(f"Total solar radiation: {results['SWin_Wm2']:.1f} W/m²")
print(f"PAR radiation: {results['PAR_Wm2']:.1f} W/m²")
print(f"Diffuse PAR fraction: {results['PAR_diffuse_fraction']:.3f}")
```

### Working with Raster Data

```python
import rasters as rt
from FLiESANN import FLiESANN
from datetime import datetime

# Load albedo raster
albedo = rt.Raster.open("albedo.tif")

# Process entire raster
results = FLiESANN(
    albedo=albedo,
    time_UTC=datetime(2024, 7, 15, 18, 0),
    geometry=albedo.geometry
)

# Save results
results["PAR_Wm2"].save("PAR_radiation.tif")
results["NIR_Wm2"].save("NIR_radiation.tif")
```

### Manual Parameter Specification

```python
from shapely.geometry import Point

# Specify atmospheric parameters manually
results = FLiESANN(
    albedo=0.15,
    COT=5.0,           # Cloud optical thickness
    AOT=0.1,           # Aerosol optical thickness
    vapor_gccm=2.5,    # Water vapor (g/cm²)
    ozone_cm=0.3,      # Ozone column (cm)
    elevation_m=1500,  # Elevation (m)
    SZA_deg=30.0,      # Solar zenith angle
    KG_climate=2,      # Köppen-Geiger climate type
    time_UTC=datetime(2024, 7, 15, 18, 0),
    geometry=Point(-118.0, 34.0)
)
```

### Batch Processing with Arrays

```python
import numpy as np
from shapely.geometry import MultiPoint

# Process multiple points simultaneously
n_points = 1000
albedo_array = np.random.uniform(0.1, 0.3, n_points)
elevation_array = np.random.uniform(0, 3000, n_points)
coordinates = [(lon, lat) for lon, lat in zip(
    np.random.uniform(-180, 180, n_points),
    np.random.uniform(-90, 90, n_points)
)]

results = FLiESANN(
    albedo=albedo_array,
    elevation_m=elevation_array,
    time_UTC=datetime(2024, 7, 15, 18, 0),
    geometry=MultiPoint(coordinates)
)
```

### ECOSTRESS Scene Processing

```python
from dateutil import parser
import rasters as rt
from FLiESANN import FLiESANN

# Load ECOSTRESS albedo scene
albedo_filename = "ECOv002_L2T_STARS_11SPS_20240728_0712_01_albedo.tif"
albedo = rt.Raster.open(albedo_filename)

# Extract time from filename
time_UTC = parser.parse("20240728T0712")

# Process scene
results = FLiESANN(
    albedo=albedo,
    time_UTC=time_UTC,
    geometry=albedo.geometry
)

# Access radiation components
total_radiation = results["SWin_Wm2"]
par_radiation = results["PAR_Wm2"]
nir_radiation = results["NIR_Wm2"]
diffuse_par = results["PAR_diffuse_Wm2"]
direct_par = results["PAR_direct_Wm2"]
```

## Output Parameters

The function returns a dictionary containing the following radiation components:

### Primary Radiation Components
- **`SWin_Wm2`**: Total shortwave incoming radiation at surface (W/m²)
- **`SWin_TOA_Wm2`**: Shortwave radiation at top of atmosphere (W/m²)
- **`UV_Wm2`**: Ultraviolet radiation (280-400 nm, W/m²)
- **`PAR_Wm2`**: Photosynthetically active radiation (400-700 nm, W/m²)
- **`NIR_Wm2`**: Near-infrared radiation (700-3000 nm, W/m²)

### Direct and Diffuse Components
- **`PAR_diffuse_Wm2`**: Diffuse visible radiation (W/m²)
- **`NIR_diffuse_Wm2`**: Diffuse near-infrared radiation (W/m²)
- **`PAR_direct_Wm2`**: Direct visible radiation (W/m²)
- **`NIR_direct_Wm2`**: Direct near-infrared radiation (W/m²)

### Normalized Parameters
- **`atmospheric_transmittance`**: Total atmospheric transmittance (0-1)
- **`UV_proportion`**: Fraction of radiation in UV band (0-1)
- **`PAR_proportion`**: Fraction of radiation in visible band (0-1)
- **`NIR_proportion`**: Fraction of radiation in NIR band (0-1)
- **`UV_diffuse_fraction`**: Diffuse fraction of UV radiation (0-1)
- **`PAR_diffuse_fraction`**: Diffuse fraction of visible radiation (0-1)
- **`NIR_diffuse_fraction`**: Diffuse fraction of NIR radiation (0-1)

## Model Validation

The ANN model has been extensively validated against:
- Original FLiES Monte Carlo simulations
- Ground-based radiation measurements
- ECOSTRESS mission cal/val data

Validation results show:
- **RMSE < 15 W/m²** for total solar radiation
- **R² > 0.95** for most radiation components
- **Bias < 5%** across diverse atmospheric conditions

## Performance

The ANN emulator provides significant computational advantages:
- **~1000x faster** than original Monte Carlo FLiES
- Processes **millions of pixels** in seconds
- Enables **real-time** operational applications
- **GPU acceleration** supported via TensorFlow

## Applications

FLiESANN is used in:

1. **NASA Earth Science Missions**:
   - ECOSTRESS evapotranspiration products
   - SBG mission planning and data processing
   - BESS ecosystem modeling

2. **Climate and Weather Modeling**:
   - Atmospheric correction for satellite data
   - Surface energy balance studies
   - Climate model validation

3. **Agricultural Applications**:
   - Crop modeling and yield prediction
   - Precision agriculture optimization
   - Irrigation scheduling

4. **Renewable Energy**:
   - Solar resource assessment
   - Photovoltaic system optimization
   - Energy forecasting

## Command Line Tools

Verify installation and model functionality:

```bash
verify-FLiESANN
```

## Examples and Notebooks

The package includes comprehensive examples:

- **Basic usage examples**: [`examples/`](examples/)
- **Jupyter notebooks**: [`notebooks/`](notebooks/)
- **ECOSTRESS processing workflows**
- **Validation and sensitivity analyses**

## API Reference

### Main Function

```python
FLiESANN(
    albedo: Union[Raster, np.ndarray, float],
    COT: Union[Raster, np.ndarray, float] = None,
    AOT: Union[Raster, np.ndarray, float] = None,
    vapor_gccm: Union[Raster, np.ndarray, float] = None,
    ozone_cm: Union[Raster, np.ndarray, float] = None,
    elevation_m: Union[Raster, np.ndarray, float] = None,
    SZA_deg: Union[Raster, np.ndarray, float] = None,
    KG_climate: Union[Raster, np.ndarray, int] = None,
    SWin_Wm2: Union[Raster, np.ndarray, float] = None,
    geometry: Union[RasterGeometry, Point, MultiPoint] = None,
    time_UTC: datetime = None,
    day_of_year: Union[Raster, np.ndarray, float] = None,
    hour_of_day: Union[Raster, np.ndarray, float] = None,
    GEOS5FP_connection = None,
    NASADEM_connection = None,
    resampling: str = "cubic",
    ANN_model = None,
    model_filename: str = None,
    split_atypes_ctypes: bool = True,
    zero_COT_correction: bool = False
) -> dict
```

## Citation

If you use FLiESANN in your research, please cite:

### Primary FLiES References

1. Kobayashi, H., & Iwabuchi, H. (2008). *A coupled 1-D atmospheric and 3-D canopy radiative transfer model for canopy reflectance, light environment, and photosynthesis simulation in a heterogeneous landscape*. **Remote Sensing of Environment**, 112(1), 173-185.  
   [https://doi.org/10.1016/j.rse.2007.04.010](https://doi.org/10.1016/j.rse.2007.04.010)

2. Kobayashi, H., Ryu, Y., & Baldocchi, D. D. (2012). *A framework for estimating vertical profiles of canopy reflectance, light environment, and photosynthesis in discontinuous canopies*. **Agricultural and Forest Meteorology**, 150(5), 601-619.  
   [https://doi.org/10.1016/j.agrformet.2010.12.001](https://doi.org/10.1016/j.agrformet.2010.12.001)

### ANN Implementation

3. Halverson, G. H., & Kobayashi, H. (2024). *FLiESANN: Artificial Neural Network Emulator for the Forest Light Environmental Simulator Radiative Transfer Model*. **Software**, NASA Jet Propulsion Laboratory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions:

- **GitHub Issues**: [https://github.com/JPL-Evapotranspiration-Algorithms/FLiESANN/issues](https://github.com/JPL-Evapotranspiration-Algorithms/FLiESANN/issues)
- **Email**: [gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)
- **Documentation**: [https://github.com/JPL-Evapotranspiration-Algorithms/FLiESANN](https://github.com/JPL-Evapotranspiration-Algorithms/FLiESANN)

## Acknowledgments

This work was supported by NASA's Earth Science Division and the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration. The original FLiES algorithm was developed by Dr. Hideki Kobayashi at the Japan Agency for Marine-Earth Science and Technology (JAMSTEC).

---

**Copyright © 2024 California Institute of Technology. All rights reserved.**
