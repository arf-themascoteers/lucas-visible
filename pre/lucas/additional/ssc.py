import os
os.chdir("../")
import numpy as np
import pandas as pd


def check():
    topsoil_file = "data/topsoil/topsoil.csv"

    topsoil_df = pd.read_csv(topsoil_file)
    done = 0
    non_empties = 0
    for i in range(len(topsoil_df)):
        topsoil_row = topsoil_df.iloc[i]

        clay = topsoil_row["Clay"]
        sand = topsoil_row["Sand"]
        silt = topsoil_row["Silt"]

        if not np.isnan(clay) and not np.isnan(sand) and not np.isnan(silt):
            non_empties = non_empties + 1
        done = done + 1
        if done%1000 == 0:
            print(f"Done {done}")

    print(f"{non_empties}")
    print("done")


if __name__ == "__main__":
    check()