from conversion.lucas import one_time_lucas_rgb_abs_to_refl
from conversion import  cielab, hsv, hsv_xy, xyY, XYZ, hue, saturation, value, l,a,b, red, green, blue
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

    data_red = f"{basedir}/red.csv"
    data_green = f"{basedir}/green.csv"
    data_blue = f"{basedir}/blue.csv"

    data_hue = f"{basedir}/hue.csv"
    data_saturation = f"{basedir}/saturation.csv"
    data_value = f"{basedir}/value.csv"

    data_l = f"{basedir}/l.csv"
    data_a = f"{basedir}/a.csv"
    data_b = f"{basedir}/b.csv"



    # hsv.process(data_rgb, data_hsv)
    # hsv_xy.process(data_hsv, data_hsv_xy)
    # XYZ.process(data_rgb, data_XYZ)
    # xyY.process(data_XYZ, data_xyY)
    # cielab.process(data_XYZ, data_cielab)

    # hue.process(data_hsv, data_hue)
    # saturation.process(data_hsv, data_saturation)
    # value.process(data_hsv, data_value)
    #
    # l.process(data_cielab, data_l)
    # a.process(data_cielab, data_a)
    # b.process(data_cielab, data_b)

    red.process(data_rgb, data_red)
    green.process(data_rgb, data_green)
    blue.process(data_rgb, data_blue)

    print("Done preparing all datasets for",base)


if __name__ == "__main__":
    # one_time_lucas_rgb_abs_to_refl.process()
    process("lucas")
    process("raca")
    process("ossl")
    print("All dataset done")
