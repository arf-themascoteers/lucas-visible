import os

import numpy as np
import pandas as pd


def create_csv():
    spectra_src_dir = "data/spectra"
    topsoil_file = "data/topsoil/topsoil.csv"
    out_file = "data/out/full_ells.csv"

    topsoil_df = pd.read_csv(topsoil_file)
    out = open(out_file, "w")
    spec = 400
    while spec <= 2499.5:
        the_str = str(spec)
        if int(spec) == spec:
            the_str = str(int(spec))
        out.write(f"{the_str},")
        spec = spec+0.5
    out.write("phc,phh,ec,caco3,p,n,k,elevation,lc1,lu1,stones,oc")
    out.write("\n")
    done = []
    for file in os.listdir(spectra_src_dir):
        if file.endswith("csv"):
            path = os.path.join(spectra_src_dir, file)
            a_df = pd.read_csv(path)
            for i in range(len(a_df)):
                spectra_row = a_df.iloc[i]
                empties = spectra_row.isna().sum()
                if empties != 0:
                    print(file, spectra_row["PointID"])
                    continue

                point_id = spectra_row['PointID']
                if point_id in done:
                    continue
                rows = (topsoil_df.loc[topsoil_df['Point_ID'] == point_id])
                if len(rows) == 0:
                    print(point_id)
                    continue
                topsoil_row = rows.iloc[0]

                spec = 400
                while spec <= 2499.5:
                    val = spec
                    if int(val) == val:
                        val = int(val)
                    val = str(val)
                    out.write(f"{spectra_row[val]},")
                    spec = spec + 0.5

                out.write(f"{topsoil_row['pH(CaCl2)']},{topsoil_row['pH(H2O)']},{topsoil_row['EC']},{topsoil_row['CaCO3']},{topsoil_row['P']},{topsoil_row['N']},"
                          f"{topsoil_row['K']},{topsoil_row['Elevation']},{topsoil_row['LC1']},{topsoil_row['LU1']},{topsoil_row['Soil_Stones']},"
                          f"{topsoil_row['OC']}")

                out.write("\n")
                done.append(point_id)
                if len(done)%1000 == 0:
                    print(f"Done {len(done)}")


    out.close()
    print("done")


if __name__ == "__main__":
    create_csv()