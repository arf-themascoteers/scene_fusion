import pandas as pd
from sklearn.model_selection import KFold
import torch
from sklearn import model_selection
from s2_bands import S2Bands


class FoldDSManager:
    def __init__(self, csv, folds=10):
        torch.manual_seed(0)
        df = pd.read_csv(csv)
        self.x = S2Bands.get_all_bands()
        self.y = "som"
        self.folds = folds
        columns = self.x + [self.y]
        print("Input")
        print(self.x)
        df = df[columns]
        df = df.sample(frac=1)
        self.full_data = df.to_numpy()

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.full_data)):
            train_data = self.full_data[train_index]
            train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
            test_data = self.full_data[test_index]
            train_x = train_data[:, :-1]
            train_y = train_data[:, -1]
            test_x = test_data[:, :-1]
            test_y = test_data[:, -1]
            validation_x = validation_data[:, :-1]
            validation_y = validation_data[:, -1]

            yield train_x, train_y, test_x, test_y, validation_x, validation_y

    def get_folds(self):
        return self.folds

