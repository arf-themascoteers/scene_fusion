import os.path
from csv_processor import CSVProcessor
import pandas as pd


class CSVIntegrator:
    def __init__(self, processed_path, dir_hash_path, scene_list):
        self.processed_path = processed_path
        self.dir_hash_path = dir_hash_path
        self.scene_list = scene_list

        self.complete = os.path.join(self.dir_hash_path, "complete.csv")
        self.ag = os.path.join(self.dir_hash_path, "ag.csv")
        self.grid = os.path.join(self.dir_hash_path, "grid.csv")
        self.ml = os.path.join(self.dir_hash_path, "ml.csv")

    def integrate(self):
        all_complete = None
        all_ag = None
        for index, scene in enumerate(self.scene_list):
            scene_home = os.path.join(self.processed_path, scene)
            scene_csvs_home = os.path.join(scene_home, "csvs")

            scene_complete = os.path.join(scene_csvs_home, "complete.csv")
            complete = pd.read_csv(scene_complete)
            complete.insert(0, "scene", pd.Series([index] * len(complete)))
            if all_complete is None:
                all_complete = complete
            else:
                all_complete = pd.concat([all_complete, complete])

            scene_ag = os.path.join(scene_csvs_home, "ag.csv")
            ag = pd.read_csv(scene_ag)
            ag.insert(0, "scene", pd.Series([index] * len(ag)))
            if all_ag is None:
                all_ag = ag
            else:
                all_ag = pd.concat([all_ag, ag])

        all_complete.to_csv(self.complete, index=False)
        all_ag.to_csv(self.ag, index=False)
        CSVProcessor.make_ml_ready(self.ag, self.ml)
        CSVProcessor.gridify(self.ml, self.grid, scene_fusion=True)
        return self.complete, self.ag, self.ml, self.grid

