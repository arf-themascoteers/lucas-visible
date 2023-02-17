import os
os.chdir("../")
from evaluator import Evaluator


if __name__ == "__main__":
    ev = Evaluator(
        datasets=["lucas"],
        algorithms=["nn",
                    {"atype":"nn","lr":0.01},
                    {"atype":"nn","batch_size":600},
                    {"atype":"nn","mid":[50,40]},
                    {"atype":"nn","mid":[300,3]}
                    ],
        colour_space_models=["hsv"],
        prefix="nn5",
        verbose=True
    )
    ev.process()
    print("Done all")