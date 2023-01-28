from oneoff import one_time_convert_cielab, one_time_convert_hsv, one_time_convert_hsv_xy, one_time_convert_xyY, one_time_convert_XYZ
from oneoff.lucas import one_time_lucas_rgb_abs_to_refl
from oneoff.raca import one_time_raca_rgb_normalize
from oneoff.oss import one_time_oss_rgb_normalize
import os


def process(base):
    basedir = f"data/{base}"
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    rgb = f"{basedir}/rgb.csv"
    hsv = f"{basedir}/hsv.csv"
    hsv_xy = f"{basedir}/hsv_xy.csv"
    XYZ_original = f"{basedir}/XYZ_original.csv"
    XYZ = f"{basedir}/XYZ.csv"
    xyY_original = f"{basedir}/xyY_original.csv"
    xyY = f"{basedir}/xyY.csv"
    cielab_original = f"{basedir}/cielab_original.csv"
    cielab = f"{basedir}/cielab.csv"

    one_time_convert_hsv.process(rgb, hsv)
    one_time_convert_hsv_xy.process(hsv, hsv_xy)
    one_time_convert_XYZ.process(rgb, XYZ_original, XYZ)
    one_time_convert_xyY.process(XYZ_original, xyY_original, xyY)
    one_time_convert_cielab.process(XYZ_original, cielab_original, cielab)

    print("Done preparing all datasets for",base)


if __name__ == "__main__":
    one_time_lucas_rgb_abs_to_refl.process()
    process("lucas")
    # one_time_raca_rgb_normalize.process()
    # process("raca")
    # one_time_oss_rgb_normalize.process()
    # process("oss")
