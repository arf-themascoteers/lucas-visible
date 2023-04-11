import os
os.chdir("../")
import numpy as np
import pandas as pd


def check():
    topsoil_file = "data/topsoil/topsoil.csv"

    topsoil_df = pd.read_csv(topsoil_file)
    lcs = {}
    lus = {}
    done = []
    for i in range(len(topsoil_df)):
        topsoil_row = topsoil_df.iloc[i]
        # empties = topsoil_row.isna().sum()
        # if empties != 0:
        #     print(file, topsoil_row["PointID"])
        #     continue

        point_id = topsoil_row['Point_ID']
        rows = (topsoil_df.loc[topsoil_df['Point_ID'] == point_id])
        if len(rows) == 0:
            print(f"Duplicate {point_id}")

        # if np.isnan(topsoil_row['pH(CaCl2)']) or np.isnan(topsoil_row['pH(H2O)'])  or np.isnan(topsoil_row['EC']) \
        #         or np.isnan(topsoil_row['CaCO3']) or np.isnan(topsoil_row['P'])  or np.isnan(topsoil_row['N']) \
        #         or np.isnan(topsoil_row['K'])  or np.isnan(topsoil_row['OC']):
        #     print("Some property is missing")
        #     continue

        lc_code = topsoil_row["LC1"]
        lu_code = topsoil_row["LU1"]
        lc_desc = topsoil_row["LC1_Desc"]
        lu_desc = topsoil_row["LU1_Desc"]

        if lc_code in lcs:
            if lcs[lc_code] != lc_desc:
                print(f"LC {lc_code} - prev {lcs[lc_code]} - new {lc_desc}")
        else:
            lcs[lc_code] = lc_desc

        if lu_code in lus:
            if lus[lu_code] != lu_desc:
                print(f"LU {lu_code} - prev {lus[lu_code]} - new {lu_desc}")
        else:
            lus[lu_code] = lu_desc

        done.append(point_id)
        if len(done)%1000 == 0:
            print(f"Done {len(done)}")

    print(lcs)
    print(lus)
    print("done")


if __name__ == "__main__":
    check()