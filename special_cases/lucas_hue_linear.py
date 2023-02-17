import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        datasets=["lucas"],
        algorithms=["linear"],
        colour_space_models=["hue"],
        prefix="hue",
        verbose=True
    )
    ev.process()
    print("Done all")