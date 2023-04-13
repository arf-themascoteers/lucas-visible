import os
import numpy as np
import pandas as pd


def create_csv():
    topsoil_file = "data/topsoil/topsoil.csv"
    gc_file = "data/topsoil/data_with_gc.csv"
    out_file = "data/topsoil/topsoilx.csv"

    topsoil_df = pd.read_csv(topsoil_file)
    gc_df = pd.read_csv(gc_file)

    topsoil_columns = ["Point_ID","Coarse","Clay","Sand","Silt","pH(CaCl2)","pH(H2O)","EC","CaCO3","P","N","K","Elevation","LC1","LU1","Soil_Stones","OC"]
    gc_columns = ["Point_ID","lon","lat"]

    topsoil_df = topsoil_df[topsoil_columns]
    gc_df = gc_df[gc_columns]

    topsoil_df = topsoil_df.merge(gc_df,on="Point_ID")
    topsoil_df = topsoil_df.set_axis(["point_id","coarse","clay","sand","silt","phc","phh","ec","caco3","p","n","k","elevation","lc1","lu1","stones","oc","lon","lat"], axis=1)
    topsoil_df = topsoil_df[["point_id","coarse","clay","sand","silt","phc","phh","ec","caco3","p","n","k","elevation","lc1","lu1","stones","lon","lat","oc"]]
    topsoil_df.to_csv(out_file, index=False)
    print("done")


if __name__ == "__main__":
    create_csv()