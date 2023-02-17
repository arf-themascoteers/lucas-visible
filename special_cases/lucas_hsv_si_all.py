import os
os.chdir("../")
from evaluator import Evaluator
import ds_manager


if __name__ == "__main__":
    sis = ds_manager.DSManager.si_list()
    colour_space_models = ["hsv"]
    colour_space_models.append({"cspace":"hsv","si":sis})
    colour_space_models.append({"cspace":"hsv","si":sis,"si_only":True})
    for x in sis:
        colour_space_models.append({"cspace":"hsv", "si":[x], "si_only":True})
    ev = Evaluator(
        datasets=["lucas"],
        algorithms=["lr","rf","nn"],
        colour_space_models=colour_space_models,
        prefix="lucas_hsv_si_all",
        verbose=True
    )
    ev.process()
    print("Done all")