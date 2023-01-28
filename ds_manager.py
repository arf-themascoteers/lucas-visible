from lucas_dataset import LucasDataset
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold


class DSManager:
    def __init__(self, dt, cspace, si=None, si_only=False, normalize = True):
        if si is None:
            si = []
        csv_file_location = f"data/{dt}/{cspace}.csv"
        df = pd.read_csv(csv_file_location)
        npdf = df.to_numpy()
        npdf = self.process_si(dt, npdf, si, si_only)
        if normalize:
            npdf = self._normalize(npdf)
        train, test = model_selection.train_test_split(npdf, test_size=0.2, random_state=1)
        self.full_data = np.concatenate((train, test), axis=0)
        self.full_ds = LucasDataset(npdf)
        self.train_ds = LucasDataset(train)
        self.test_ds = LucasDataset(test)

    def get_test_ds(self):
        return self.test_ds

    def get_train_ds(self):
        return self.train_ds

    def get_10_folds(self):
        kf = KFold(n_splits=10)
        for i, (train_index, test_index) in enumerate(kf.split(self.full_data)):
            train_data = self.full_data[train_index]
            test_data = self.full_data[test_index]
            yield LucasDataset(train_data), LucasDataset(test_data)

    def get_count_itr(self):
        i = 0
        for train_ds, test_ds in self.get_10_folds():
            i = i+1
        return i

    def _normalize(self, data):
        for i in range(data.shape[1]):
            # if i != data.shape[1]-1:
            #     continue
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(data[:,i].reshape(-1, 1))
            data[:,i] = np.squeeze(x_scaled)
        return data

    def process_si(self, dt, npdf, si, si_only):
        if len(si) == 0:
            return npdf

        csv_file_location = f"data/{dt}/rgb.csv"
        df = pd.read_csv(csv_file_location)
        rgb = df.to_numpy()[:,0:3]

        si_values = np.zeros((rgb.shape[0], len(si)))

        for index, a_si in enumerate(si):
            a_si_values = self.determine_si(rgb, a_si)
            si_values[:, index] = a_si_values

        base_features = npdf[:,0:-1]
        if si_only:
            base_features = si_values
        else:
            base_features = np.concatenate((base_features, si_values), axis=1)
        soc = npdf[:, -1].reshape(-1,1)
        return np.concatenate((base_features, soc), axis=1)

    def determine_si(self, rgb, a_si):
        RED = 0
        GREEN = 1
        BLUE = 2

        if a_si == "soci":
            return (rgb[:,BLUE]) / (rgb[:,RED] * rgb[:,GREEN])

        if a_si == "ibs":
            return 1/(rgb[:,BLUE] ** 2)

        return None