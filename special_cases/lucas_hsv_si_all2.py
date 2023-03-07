import os
os.chdir("../")
from evaluator import Evaluator
import ds_manager


if __name__ == "__main__":
    sis = ds_manager.DSManager.si_list()
    colour_space_models = []
    # for a_sis in sis:
    #     colour_space_models.append({"cspace":"hsv","si":[a_sis],"si_only":True,"name":f"{a_sis}"})
    colour_space_models.append("hsv")
    colour_space_models.append({"cspace":"hsv","si":sis,"name":"hsv-si"})
    colour_space_models.append({"cspace":"hsv","si":sis,"si_only":True,"name":"si"})

    ev = Evaluator(
        datasets=["lucas"],
        algorithms=["rf", "nn"],
        colour_space_models=colour_space_models,
        prefix="sis2",
        verbose=False,
        folds=10,
        repeat=5
    )
    ev.process()
    print("Done all")