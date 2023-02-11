import os
os.chdir("../")
from evaluator import Evaluator
import ds_manager


if __name__ == "__main__":
    sis = ds_manager.DSManager.si_list()
    colour_space_models = ["hsv"]
    colour_space_models.append({"cspace":"hsv","si":sis,"name":"hsv-si"})
    colour_space_models.append({"cspace":"hsv","si":sis,"si_only":True,"name":"si"})

    ev = Evaluator(
        datasets=["lucas"],
        algorithms=["nn"],
        colour_space_models=colour_space_models,
        prefix="nn-hsv-si",
        verbose=True,
        folds=10,
        repeat=1
    )
    ev.process()
    print("Done all")