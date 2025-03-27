import pandas as pd

from .process_FLiES_ANN import FLiESANN

def process_FLiES_table(FLiES_inputs_df: pd.DataFrame) -> pd.DataFrame:
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

    Ra = FLiES_results["Ra"]
    Rg = FLiES_results["Rg"]
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
    FLiES_outputs_df["Ra"] = Ra
    FLiES_outputs_df["Rg"] = Rg
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
