import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        datasets=["lucas"],
        algorithms=["nn"],
        colour_space_models=["hsv", "hsv_xy"],
        prefix="lucas_hsv_nn",
        verbose=True
    )
    ev.process()
    print("Done all")