import os
os.chdir("../")
from evaluator import Evaluator
import ds_manager


if __name__ == "__main__":
    ev = Evaluator(
        datasets=["lucas"],
        algorithms=["nn"],
        colour_space_models=[{"cspace":"hsv","si":["soci"],"si_only":True},"hsv"],
        prefix="lucas_hsv_nn",
        verbose=True
    )
    ev.process()
    print("Done all")