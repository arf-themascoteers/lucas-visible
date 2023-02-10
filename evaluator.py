import numpy as np
import pandas as pd
import ds_manager
import os
from datetime import datetime
import torch
from train import train
from test import test
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class Evaluator:
    def __init__(self, datasets=None, algorithms=None, colour_space_models=None,prefix="", verbose=False):
        self.datasets = datasets
        if self.datasets is None:
            self.datasets = ["lucas", "raca", "ossl"]
        self.algorithms = algorithms
        if self.algorithms is None:
            self.algorithms = ["linear", "plsr", "rf", "svr", "nn"]
        self.colour_spaces = colour_space_models
        if self.colour_spaces is None:
            self.colour_spaces = ["rgb", "hsv", "hsv_xy", "XYZ", "xyY", "cielab"]
        self.colour_space_names = []
        for i in self.colour_spaces:
            if isinstance(i, str):
                self.colour_space_names.append(i)
            if type(i) is dict:
                self.colour_space_names.append(i["cspace"])
        self.verbose = verbose
        self.summary = np.zeros((len(self.colour_spaces) * len(self.datasets), len(self.algorithms)))
        self.summary_file = f"results/{prefix}_summary.csv"
        self.details_file = f"results/{prefix}_details.csv"
        self.log_file = f"results/{prefix}_log.txt"

        self.summary_index = self.create_summary_index()

        self.folds = [self.get_folds(i) for i in self.datasets]
        self.detail_row_start = self.get_details_row_start()
        self.details = np.zeros((sum(self.folds)*10, len(self.algorithms) * len(self.colour_spaces)))
        self.details_index = self.get_details_index()
        self.details_columns = self.get_details_columns()

        self.sync_summary_file()
        self.sync_details_file()
        self.create_log_file()

        self.TEST = False
        self.TEST_SCORE = 0

    def get_details_row_start(self):
        row_start = [0]
        if len(self.datasets) == 1:
            return row_start

        for i in range(len(self.folds)-1):
            row_start.append(row_start[i] + self.folds[i])
        return row_start

    def get_details_columns(self):
        details_columns = []
        for algorithm in self.algorithms:
            for colour_space in self.colour_spaces:
                details_columns.append(f"{algorithm}-{colour_space}")
        return details_columns

    def get_details_index(self):
        details_index = []
        for index_dataset, dataset in enumerate(self.datasets):
            for i in range(10):
                for fold in range(self.folds[index_dataset]):
                    num = i * 10 + fold
                    details_index.append(f"{dataset}-{num}-{fold}")
        return details_index

    def get_details_row(self, index_dataset, itr_no):
        start = self.detail_row_start[index_dataset]
        return start+itr_no

    def get_details_column(self, index_algorithm, index_colour_space):
        return len(self.colour_spaces) * index_algorithm + index_colour_space


    def set_details(self, index_algorithm, index_colour_space, index_dataset, itr_no, score):
        details_row = self.get_details_row(index_dataset, itr_no)
        details_column = self.get_details_column(index_algorithm, index_colour_space)
        self.details[details_row][details_column] = round(score,3)

    def get_details(self, index_algorithm, index_colour_space, index_dataset, itr_no):
        details_row = self.get_details_row(index_dataset, itr_no)
        details_column = self.get_details_column(index_algorithm, index_colour_space)
        return self.details[details_row][details_column]


    def get_folds(self, dataset):
        ds = ds_manager.DSManager(dataset, "rgb")
        return ds.get_count_itr()

    def sync_summary_file(self):
        if not os.path.exists(self.summary_file):
            self.write_summary()
        df = pd.read_csv(self.summary_file)
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        self.summary = df.to_numpy()

    def sync_details_file(self):
        if not os.path.exists(self.details_file):
            self.write_details()
        df = pd.read_csv(self.details_file)
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        self.details = df.to_numpy()

    def create_log_file(self):
        log_file = open(self.log_file, "a")
        log_file.write("\n")
        log_file.write(str(datetime.now()))
        log_file.write("\n==============================\n")
        log_file.close()

    def write_summary(self):
        df = pd.DataFrame(data=self.summary, columns=self.algorithms, index=self.summary_index)
        df.to_csv(self.summary_file)

    def write_details(self):
        df = pd.DataFrame(data=self.details, columns=self.details_columns, index=self.details_index)
        df.to_csv(self.details_file)


    def get_summary_row(self, index_dataset, index_colour_space):
        return (index_dataset*len(self.colour_spaces)) + index_colour_space

    def log_scores(self, dataset, algorithm, colour_space, r2s):
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

    def process_algorithm_colour_space_dataset(self, index_algorithm, index_colour_space, index_dataset):
        algorithm = self.algorithms[index_algorithm]
        colour_space = self.colour_spaces[index_colour_space]
        dataset = self.datasets[index_dataset]
        print("Start", f"{dataset} - {algorithm} - {colour_space}")

        if self.get_score(index_dataset, index_algorithm, index_colour_space) != 0:
            print(f"{dataset} - {algorithm} - {colour_space} Was done already")
        else:
            scores = self.calculate_scores_folds(index_algorithm, index_colour_space, index_dataset)
            score_mean = np.round(np.mean(scores), 3)
            scores = np.round(scores, 3)
            self.log_scores(dataset, algorithm, colour_space, scores)
            self.set_score(index_dataset, index_algorithm, index_colour_space, score_mean)
            self.write_summary()

    def process_algorithm_colour_space(self, index_algorithm, index_colour_space):
        for index_dataset, dataset in enumerate(self.datasets):
            self.process_algorithm_colour_space_dataset(index_algorithm, index_colour_space, index_dataset)

    def process_algorithm(self, index_algorithm):
        for index_colour_space, colour_space in enumerate(self.colour_spaces):
            self.process_algorithm_colour_space(index_algorithm, index_colour_space)

    def process(self):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_algorithm(index_algorithm)

    def create_summary_index(self):
        index = []
        for dataset in self.datasets:
            for colour_space in self.colour_spaces:
                index.append(f"{dataset} - {colour_space}")
        return index

    def calculate_scores_folds(self, index_algorithm, index_colour_space, index_dataset):
        algorithm = self.algorithms[index_algorithm]
        colour_space = self.colour_spaces[index_colour_space]
        dataset = self.datasets[index_dataset]
        scores = []
        for i in range(10):
            if type(colour_space) is dict:
                ds = ds_manager.DSManager(dataset, **colour_space, random_state=i)
            else:
                ds = ds_manager.DSManager(dataset, colour_space, random_state=i)

            for itr_no, (train_ds, test_ds) in enumerate(ds.get_10_folds()):
                score = self.calculate_score(train_ds, test_ds, algorithm)
                if self.verbose:
                    print(score)
                scores.append(score)
                it_now = i*10 + itr_no
                self.set_details(index_algorithm, index_colour_space, index_dataset, it_now, score)
                self.write_details()
        return scores

    def calculate_score(self, train_ds, test_ds, algorithm):
        if self.TEST:
            self.TEST_SCORE = self.TEST_SCORE + 1
            return self.TEST_SCORE

        model_instance = None
        if algorithm == "nn":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = train(device, train_ds)
            return test(device, test_ds, model_instance)
        else:
            train_x = train_ds.get_x()
            train_y = train_ds.get_y()
            test_x = test_ds.get_x()
            test_y = test_ds.get_y()

            if algorithm == "linear":
                model_instance = LinearRegression()
            elif algorithm == "plsr":
                model_instance = PLSRegression(n_components=2)
            elif algorithm == "rf":
                model_instance = RandomForestRegressor(max_depth=5, n_estimators=100)
            elif algorithm == "svr":
                model_instance = SVR()


            model_instance = model_instance.fit(train_x, train_y)
            return model_instance.score(test_x, test_y)


if __name__ == "__main__":
    ev = Evaluator()
    ev.process()
    print("Done all")