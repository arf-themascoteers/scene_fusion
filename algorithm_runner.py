import torch
from ann import ANN
from ann_es import ANNEarlyStop
from siamese import Siamese
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


class AlgorithmRunner:
    @staticmethod
    def calculate_score(train_x, train_y,
                        test_x, test_y,
                        validation_x,
                        validation_y,
                        algorithm
                        ):
        y_hats = None
        print(f"Train: {len(train_y)}, Test: {len(test_y)}, Validation: {len(validation_y)}")
        if algorithm == "ann":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANN(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_es":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNEarlyStop(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "siamese":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = Siamese(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()

        else:
            model_instance = None
            if algorithm == "mlr":
                model_instance = LinearRegression()
            elif algorithm == "plsr":
                size = train_x.shape[1]//2
                if size == 0:
                    size = 1
                model_instance = PLSRegression(n_components=size)
            elif algorithm == "rf":
                model_instance = RandomForestRegressor(max_depth=4, n_estimators=100)
            elif algorithm == "svr":
                model_instance = SVR()

            model_instance = model_instance.fit(train_x, train_y)
            y_hats = model_instance.predict(test_x)

        r2 = r2_score(test_y, y_hats)
        rmse = mean_squared_error(test_y, y_hats, squared=False)
        return max(r2,0), rmse