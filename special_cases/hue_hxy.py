import os
os.chdir("../")
from eval_all import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        datasets=["lucas"],
        algorithms=["nn"],
        colour_space_models=["hue", "hxy"],
        prefix="lucas_hue_hxy",
        verbose=True
    )
    ev.process()
    print("Done all")