import one_time_convert_hsv
import one_time_convert_hsv_xy
import one_time_convert_XYZ
import one_time_convert_xyY
import one_time_convert_cielab


def process(base):
    rgb = f"data_{base}_rgb.csv"
    hsv = f"data_{base}_hsv.csv"
    hsv_xy = f"data_{base}_hsv_xy.csv"
    XYZ_original = f"data_{base}_XYZ_original.csv"
    XYZ = f"data_{base}_XYZ.csv"
    xyY_original = f"data_{base}_xyY_original.csv"
    xyY = f"data_{base}_xyY.csv"
    cielab_original = f"data_{base}_cielab_original.csv"
    cielab = f"data_{base}_cielab.csv"

    one_time_convert_hsv.process(rgb, hsv)
    one_time_convert_hsv_xy.process(hsv, hsv_xy)
    one_time_convert_XYZ.process(rgb, XYZ_original, XYZ)
    one_time_convert_xyY.process(XYZ_original, xyY_original, xyY)
    one_time_convert_cielab.process(XYZ_original, cielab_original, cielab)


if __name__ == "__main__":
    process("mangrove")