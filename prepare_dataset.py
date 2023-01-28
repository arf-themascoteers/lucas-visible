from conversion.lucas import one_time_lucas_rgb_abs_to_refl
from conversion import  cielab, hsv, hsv_xy, xyY, XYZ
import os


def process(base):
    basedir = f"data/{base}"
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    data_rgb = f"{basedir}/rgb.csv"
    data_hsv = f"{basedir}/hsv.csv"
    data_hsv_xy = f"{basedir}/hsv_xy.csv"
    data_XYZ = f"{basedir}/XYZ.csv"
    data_xyY = f"{basedir}/xyY.csv"
    data_cielab = f"{basedir}/cielab.csv"

    hsv.process(data_rgb, data_hsv)
    hsv_xy.process(data_hsv, data_hsv_xy)
    XYZ.process(data_rgb, data_XYZ)
    xyY.process(data_XYZ, data_xyY)
    cielab.process(data_XYZ, data_cielab)

    print("Done preparing all datasets for",base)


if __name__ == "__main__":
    one_time_lucas_rgb_abs_to_refl.process()
    process("lucas")
    process("raca")
    process("oss")
    process("demmin")
    print("All dataset done")
