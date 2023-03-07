import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        datasets=["ossl"],
        algorithms=[
                    {"atype":"nn","mid":[10,5],"batch_size":600,"name":"ossl","num_epochs":100}
                    ],
        colour_space_models=["hsv"],
        prefix="ossl",
        verbose=True,
        folds=2,
        repeat=1
    )
    ev.process()
    print("Done all")