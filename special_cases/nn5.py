import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        datasets=["lucas"],
        algorithms=[
                    {"atype":"nn","mid":[300,20],"batch_size":600,"name":"300x20"}
                    ],
        colour_space_models=["hsv"],
        prefix="30x20-b600",
        verbose=True
    )
    ev.process()
    print("Done all")