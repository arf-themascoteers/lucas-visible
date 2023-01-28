import os
os.chdir("../")
from eval_all import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        datasets=["lucas"],
        algorithms=["nn"],
        colour_space_models=["hsv"],
        prefix="lucas_hsv_nn",
        verbose=True
    )
    ev.process()
    print("Done all")