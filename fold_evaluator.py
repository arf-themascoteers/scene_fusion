from s2_extractor import S2Extractor
from fold_reporter import FoldReporter
from algorithm_runner import AlgorithmRunner
from fold_ds_manager import FoldDSManager


class FoldEvaluator:
    def __init__(self, prefix="", verbose=False, repeat=1, folds=10, algorithms=None):
        self.repeat = repeat
        self.folds = folds
        self.verbose = verbose
        self.algorithms = algorithms

        if self.algorithms is None:
            self.algorithms = ["mlr", "rf", "svr", "ann", "ann_es"]

        self.config_list = ["Run"]
        self.csvs = []
        self.scenes = []

        for config in self.config_list:
            s2 = S2Extractor(["S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040"])
            ml = s2.process()
            self.csvs.append(ml)

        self.reporter = FoldReporter(prefix, self.config_list, self.algorithms, self.repeat, self.folds)

    def process(self):
        for repeat_number in range(self.repeat):
            self.process_repeat(repeat_number)

    def process_repeat(self, repeat_number):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_algorithm(repeat_number, index_algorithm)

    def process_algorithm(self, repeat_number, index_algorithm):
        for index_config in range(len(self.config_list)):
            config = self.config_list[index_config]
            print("Start", f"{repeat_number}:{self.algorithms[index_algorithm]} - {config}")
            self.process_config(repeat_number, index_algorithm, index_config)

    def process_config(self, repeat_number, index_algorithm, index_config):
        algorithm = self.algorithms[index_algorithm]
        ds = FoldDSManager(self.csvs[index_config], folds=self.folds)
        for fold_number, (train_x, train_y, test_x, test_y, validation_x, validation_y) in enumerate(ds.get_k_folds()):
            print("CSV: ", self.csvs[index_config])
            r2, rmse = self.reporter.get_details(index_algorithm, repeat_number, fold_number, index_config)
            if r2 != 0:
                print(f"{repeat_number}-{fold_number} done already")
                continue
            else:
                r2, rmse = AlgorithmRunner.calculate_score(train_x, train_y,
                                                           test_x, test_y,
                                                           validation_x, validation_y,
                                                           algorithm
                                                           )
            if self.verbose:
                print(f"{r2} - {rmse}")
                print(f"R2 - RMSE")
            self.reporter.set_details(index_algorithm, repeat_number, fold_number, index_config, r2, rmse)
            self.reporter.write_details()
            self.reporter.update_summary()

