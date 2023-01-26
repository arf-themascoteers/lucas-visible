import numpy as np
import pandas as pd
import ds_manager
import evaluate
import os
from datetime import datetime


class Evaluator:
    def __init__(self):
        self.datasets = ["lucas", "raca"]
        self.algorithms = ["linear", "plsr", "rf", "svr", "nn"]
        self.colour_spaces = ["rgb", "hsv", "hsv_xy", "XYZ", "xyY", "cielab"]
        self.summary = np.zeros((len(self.colour_spaces) * len(self.datasets), len(self.algorithms)))
        self.summary_file = f"summary.csv"
        self.detail_file = f"details.csv"
        self.log_file = f"log.txt"
        self.sync_summary_file()
        #self.create_details_file()
        self.create_log_file()

    def sync_summary_file(self):
        if not os.path.exists(self.summary_file):
            self.write_summary()
        df = pd.read_csv(self.summary_file)
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        self.summary = df.to_numpy()


    def create_details_file(self):
        f = open(self.detail_file)
        f.close()

    def create_log_file(self):
        log_file = open(self.log_file, "a")
        log_file.write("\n")
        log_file.write(str(datetime.now()))
        log_file.write("\n==============================\n")
        log_file.close()

    def write_summary(self):
        index = self.colour_spaces * len(self.datasets)
        df = pd.DataFrame(data=self.summary, columns=self.algorithms, index=index)
        df.to_csv(self.summary_file)

    def get_summary_row(self, index_dataset, index_colour_space):
        return (index_dataset*len(self.colour_spaces)) + index_colour_space

    def write_scores(self, dataset, algorithm, colour_space, r2s):
        log_file = open(self.log_file, "a")
        log_file.write(f"\n{dataset} - {algorithm} - {colour_space}\n")
        log_file.write(str(r2s))
        log_file.write("\n")
        log_file.close()

    def set_score(self, index_dataset, index_algorithm, index_colour_space, score):
        row = self.get_summary_row(index_dataset, index_colour_space)
        self.summary[row][index_algorithm] = score

    def get_score(self, index_dataset, index_algorithm, index_colour_space):
        row = self.get_summary_row(index_dataset, index_colour_space)
        return self.summary[row][index_algorithm]

    def process_dataset_algorithm_colour_space(self, index_dataset, index_algorithm, index_colour_space):
        dataset = self.datasets[index_dataset]
        algorithm = self.algorithms[index_algorithm]
        colour_space = self.colour_spaces[index_colour_space]
        print("Start", f"{dataset} - {algorithm} - {colour_space}")

        if self.get_score(index_dataset, index_algorithm, index_colour_space) != 0:
            print(f"{dataset} - {algorithm} - {colour_space} Was done already")
        else:
            ds = ds_manager.DSManager(dataset, colour_space)
            r2s = evaluate.r2(ds, algorithm)
            r2_mean = np.round(np.mean(r2s), 3)
            r2s = np.round(r2s, 3)
            self.write_scores(dataset, algorithm, colour_space, r2s)
            self.set_score(index_dataset, index_algorithm, index_colour_space, r2_mean)
            self.write_summary()

    def process_dataset_algorithm(self, index_dataset, index_algorithm):
        for index_colour_space, colour_space in enumerate(self.colour_spaces):
            self.process_dataset_algorithm_colour_space(index_dataset, index_algorithm, index_colour_space)

    def process_dataset(self, index_dataset):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_dataset_algorithm(index_dataset, index_algorithm)

    def process(self):
        for index_dataset, dataset in enumerate(self.datasets):
            self.process_dataset(index_dataset)


if __name__ == "__main__":
    ev = Evaluator()
    ev.process()
    print("Done all")