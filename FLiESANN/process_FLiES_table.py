import pandas as pd

from .process_FLiES_ANN import FLiESANN

def process_FLiES_table(FLiES_inputs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame of FLiES inputs and returns a DataFrame with FLiES outputs.

    Parameters:
    FLiES_inputs_df (pd.DataFrame): A DataFrame containing the following columns:
        - doy (int): Day of the year.
        - albedo (float): Surface albedo.
        - COT (float): Cloud optical thickness.
        - AOT (float): Aerosol optical thickness.
        - vapor_gccm (float): Water vapor in grams per cubic centimeter.
        - ozone_cm (float): Ozone concentration in centimeters.
        - elevation_km (float): Elevation in kilometers.
        - SZA (float): Solar zenith angle in degrees.
        - KG (str): Köppen-Geiger climate classification.

    Returns:
    pd.DataFrame: A DataFrame with the same structure as the input, but with additional columns:
        - SWin_Wm2: incoming shortwave radiation in watts per square meter
        - SWin_TOA_Wm2: top-of-atmosphere incoming shortwave radiation in watts per square meter
        - UV: Ultraviolet radiation.
        - VIS: Visible radiation.
        - NIR: Near-infrared radiation.
        - VISdiff: Diffuse visible radiation.
        - NIRdiff: Diffuse near-infrared radiation.
        - VISdir: Direct visible radiation.
        - NIRdir: Direct near-infrared radiation.
        - tm: Temperature.
        - puv: Proportion of ultraviolet radiation.
        - pvis: Proportion of visible radiation.
        - pnir: Proportion of near-infrared radiation.
        - fduv: Fraction of diffuse ultraviolet radiation.
        - fdvis: Fraction of diffuse visible radiation.
        - fdnir: Fraction of diffuse near-infrared radiation.
    """
    FLiES_results = FLiESANN(
        day_of_year=FLiES_inputs_df.doy,
        albedo=FLiES_inputs_df.albedo,
        COT=FLiES_inputs_df.COT,
        AOT=FLiES_inputs_df.AOT,
        vapor_gccm=FLiES_inputs_df.vapor_gccm,
        ozone_cm=FLiES_inputs_df.ozone_cm,
        elevation_km=FLiES_inputs_df.elevation_km,
        SZA=FLiES_inputs_df.SZA,
        KG_climate=FLiES_inputs_df.KG
    )

    SWin_TOA_Wm2 = FLiES_results["SWin_TOA_Wm2"]
    SWin_Wm2 = FLiES_results["SWin_Wm2"]
    UV = FLiES_results["UV"]
    VIS = FLiES_results["VIS"]
    NIR = FLiES_results["NIR"]
    VISdiff = FLiES_results["VISdiff"]
    NIRdiff = FLiES_results["NIRdiff"]
    VISdir = FLiES_results["VISdir"]
    NIRdir = FLiES_results["NIRdir"]
    tm = FLiES_results["tm"]
    puv = FLiES_results["puv"]
    pvis = FLiES_results["pvis"]
    pnir = FLiES_results["pnir"]
    fduv = FLiES_results["fduv"]
    fdvis = FLiES_results["fdvis"]
    fdnir = FLiES_results["fdnir"]

    FLiES_outputs_df = FLiES_inputs_df.copy()
    FLiES_outputs_df["SWin_TOA_Wm2"] = SWin_TOA_Wm2
    FLiES_outputs_df["SWin_Wm2"] = SWin_Wm2
    FLiES_outputs_df["UV"] = UV
    FLiES_outputs_df["VIS"] = VIS
    FLiES_outputs_df["NIR"] = NIR
    FLiES_outputs_df["VISdiff"] = VISdiff
    FLiES_outputs_df["NIRdiff"] = NIRdiff
    FLiES_outputs_df["VISdir"] = VISdir
    FLiES_outputs_df["NIRdir"] = NIRdir
    FLiES_outputs_df["tm"] = tm
    FLiES_outputs_df["puv"] = puv
    FLiES_outputs_df["pvis"] = pvis
    FLiES_outputs_df["pnir"] = pnir
    FLiES_outputs_df["fduv"] = fduv
    FLiES_outputs_df["fdvis"] = fdvis
    FLiES_outputs_df["fdnir"] = fdnir

    return FLiES_outputs_df
