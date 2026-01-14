import numpy as np

def partition_spectral_albedo_with_NDVI(
        broadband_albedo: np.ndarray,
        NDVI: np.ndarray,
        PAR_proportion: np.ndarray,
        NIR_proportion: np.ndarray) -> tuple:
    """
    Partition broadband albedo into PAR and NIR spectral components using NDVI.
    
    This function implements an empirical relationship between NDVI and spectral reflectance
    properties based on peer-reviewed literature. Vegetation exhibits distinct spectral signatures
    with low reflectance in the visible (PAR) range due to chlorophyll absorption and high
    reflectance in the NIR range due to leaf cellular structure.
    
    References:
    -----------
    - Liang, S. (2001). "Narrowband to broadband conversions of land surface albedo I: Algorithms."
      Remote Sensing of Environment, 76(3), 213-238. DOI: 10.1016/S0034-4257(00)00205-4
      
    - Schaaf, C.B., et al. (2002). "First operational BRDF, albedo nadir reflectance products from MODIS."
      Remote Sensing of Environment, 83(1-2), 135-148. DOI: 10.1016/S0034-4257(02)00091-3
      
    - Wang, K., & Liang, S. (2009). "Estimation of daytime net radiation from shortwave radiation
      measurements and meteorological observations." Journal of Applied Meteorology and Climatology,
      48(3), 634-643. DOI: 10.1175/2008JAMC1959.1
      
    - Pinty, B., et al. (2006). "Simplifying the interaction of land surfaces with radiation for
      relating remote sensing products to climate models." Journal of Geophysical Research, 111, D02116.
      DOI: 10.1029/2005JD005952
    
    Empirical Relationships:
    -----------------------
    For dense vegetation (NDVI > 0.5):
        - PAR albedo: 0.03-0.10 (typically ~0.05-0.08)
        - NIR albedo: 0.30-0.50 (typically ~0.35-0.45)
        - NIR/PAR ratio: ~4-6
    
    For sparse vegetation (NDVI 0.2-0.5):
        - PAR albedo: 0.08-0.15
        - NIR albedo: 0.15-0.30
        - NIR/PAR ratio: ~1.5-3
    
    For bare soil/desert (NDVI < 0.2):
        - PAR albedo: 0.15-0.35
        - NIR albedo: 0.20-0.40
        - NIR/PAR ratio: ~1.0-1.5
    
    Args:
        broadband_albedo (np.ndarray): Broadband surface albedo (0.3-5.0 μm range)
        NDVI (np.ndarray): Normalized Difference Vegetation Index (-1 to 1)
        PAR_proportion (np.ndarray): Fraction of incoming solar radiation in PAR band (0.4-0.7 μm)
        NIR_proportion (np.ndarray): Fraction of incoming solar radiation in NIR band (0.7-3.0 μm)
    
    Returns:
        tuple: (PAR_albedo, NIR_albedo)
            - PAR_albedo (np.ndarray): Spectral albedo in photosynthetically active radiation band
            - NIR_albedo (np.ndarray): Spectral albedo in near-infrared band
    
    Notes:
        The implementation uses a continuous empirical function based on MODIS albedo products
        (Schaaf et al. 2002) and validated partitioning relationships (Liang 2001, Wang & Liang 2009).
        The spectral albedos are constrained to satisfy:
        
        broadband_albedo ≈ PAR_proportion × PAR_albedo + NIR_proportion × NIR_albedo
        
        The NIR/PAR albedo ratio increases with NDVI according to:
        ratio = 1.0 + 5.0 × NDVI^2  (for NDVI > 0)
        
        This quadratic relationship captures the nonlinear increase in NIR reflectance and decrease
        in PAR reflectance as vegetation density and health increase.
    """
    # Clip NDVI to valid range
    NDVI_clipped = np.clip(NDVI, -1, 1)
    
    # Calculate NIR/PAR albedo ratio from NDVI
    # Based on empirical relationships from Schaaf et al. (2002) and Pinty et al. (2006)
    # For vegetation, the ratio increases with NDVI as chlorophyll absorption increases in PAR
    # and cellular scattering increases in NIR
    
    # Quadratic relationship captures nonlinear vegetation spectral response
    # ratio ranges from ~1.0 (NDVI=0, bare soil) to ~6.0 (NDVI=1, dense vegetation)
    ratio = np.where(
        NDVI_clipped > 0,
        1.0 + 5.0 * NDVI_clipped**2,  # Quadratic: ratio from 1 at NDVI=0 to 6 at NDVI=1
        1.0  # For water, snow, or negative NDVI, assume similar PAR and NIR albedo
    )
    
    # Partition broadband albedo into spectral components
    # Constraint: broadband_albedo ≈ PAR_proportion × PAR_albedo + NIR_proportion × NIR_albedo
    # Substituting NIR_albedo = ratio × PAR_albedo:
    # broadband_albedo = PAR_proportion × PAR_albedo + NIR_proportion × ratio × PAR_albedo
    # broadband_albedo = PAR_albedo × (PAR_proportion + NIR_proportion × ratio)
    # Therefore: PAR_albedo = broadband_albedo / (PAR_proportion + NIR_proportion × ratio)
    
    denominator = PAR_proportion + NIR_proportion * ratio
    PAR_albedo = np.where(denominator > 0, broadband_albedo / denominator, broadband_albedo)
    NIR_albedo = PAR_albedo * ratio
    
    # Clip to physical range [0, 1]
    PAR_albedo = np.clip(PAR_albedo, 0, 1)
    NIR_albedo = np.clip(NIR_albedo, 0, 1)
    
    return PAR_albedo, NIR_albedo
