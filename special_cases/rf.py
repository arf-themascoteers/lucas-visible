import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        datasets=["lucas"],
        algorithms=["rf"],
        colour_space_models=["hsv"],
        prefix="rf",
        verbose=True
    )
    ev.process()
    print("Done all")