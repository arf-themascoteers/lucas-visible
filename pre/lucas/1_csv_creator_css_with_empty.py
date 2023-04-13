import os
import pandas as pd

NAN = -9999

def create_csv():
    spectra_src_dir = "data/spectra"
    topsoil_file = "data/topsoil/topsoilx.csv"
    out_file = "data/out/full_with_empty.csv"

    topsoil_df = pd.read_csv(topsoil_file)
    topsoil_df = topsoil_df.fillna(NAN)

    out = open(out_file, "w")
    spec = 400
    while spec <= 2499.5:
        the_str = str(spec)
        if int(spec) == spec:
            the_str = str(int(spec))
        out.write(f"{the_str},")
        spec = spec+0.5
    out.write("point_id,coarse,clay,sand,silt,phc,phh,ec,caco3,p,n,k,elevation,lc1,lu1,stones,lon,lat,oc")
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
                rows = (topsoil_df.loc[topsoil_df['point_id'] == point_id])
                if len(rows) == 0:
                    print(point_id)
                    continue
                topsoil_row = rows.iloc[0]
                # if np.isnan(topsoil_row['Clay']) or np.isnan(topsoil_row['Sand'])  or np.isnan(topsoil_row['Silt']):
                #     continue
                spec = 400
                while spec <= 2499.5:
                    val = spec
                    if int(val) == val:
                        val = int(val)
                    val = str(val)
                    out.write(f"{spectra_row[val]},")
                    spec = spec + 0.5
                #"point_id,coarse,clay,sand,silt,phc,phh,ec,caco3,p,n,k,elevation,lc1,lu1,stones,oc,lon,lat"
                out.write(f"{topsoil_row['point_id']},{topsoil_row['coarse']},{topsoil_row['clay']},{topsoil_row['sand']},"
                          f"{topsoil_row['silt']},{topsoil_row['phc']},{topsoil_row['phh']},{topsoil_row['ec']},"
                          f"{topsoil_row['caco3']},{topsoil_row['p']},{topsoil_row['n']},{topsoil_row['k']},"
                          f"{topsoil_row['elevation']},{topsoil_row['lc1']},{topsoil_row['lu1']},{topsoil_row['stones']},"
                          f"{topsoil_row['lon']},{topsoil_row['lat']},{topsoil_row['oc']}")

                out.write("\n")
                done.append(point_id)
                if len(done)%1000 == 0:
                    print(f"Done {len(done)}")


    out.close()
    print("done")


if __name__ == "__main__":
    create_csv()