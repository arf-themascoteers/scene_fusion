import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class CSVProcessor:
    @staticmethod
    def aggregate(complete, ag):
        df = pd.read_csv(complete)
        df.drop(columns=CSVProcessor.get_geo_columns(), axis=1, inplace=True)
        spatial_columns = CSVProcessor.get_spatial_columns(df)
        columns_to_agg = df.columns.drop(spatial_columns)

        agg_dict = {}
        agg_dict["counter"] = ("som", 'count')
        agg_dict["som_std"] = ("som", 'std')
        for col in columns_to_agg:
            agg_dict[col] = (col, "mean")

        df_group_object = df.groupby(spatial_columns)
        df_mean = df_group_object.agg(**agg_dict).reset_index()
        df_mean.insert(0, "cell", df_mean.index)
        df_mean = df_mean[df_mean["counter"] >= 1]
        df_mean = df_mean.round(4)
        df_mean.to_csv(ag, index=False)

    @staticmethod
    def make_ml_ready(ag, ml):
        df = pd.read_csv(ag)
        df = CSVProcessor.make_ml_ready_df(df)
        df = df.round(4)
        df.to_csv(ml, index=False)

    @staticmethod
    def make_ml_ready_df(df):
        for col in ["when"]:
            if col in df.columns:
                df.drop(inplace=True, columns=[col], axis=1)
        for col in df.columns:
            if col not in ["scene","row","column","counter","som_std","cell"]:
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(df[[col]])
        return df

    @staticmethod
    def get_spatial_columns(df):
        spatial_columns = ["row", "column"]
        if "scene" in df.columns:
            spatial_columns = ["scene"] + spatial_columns
        return spatial_columns

    @staticmethod
    def get_geo_columns():
        return ["lon", "lat", "when"]

    @classmethod
    def gridify(cls, ag, grid, scene_fusion = False):
        df = pd.read_csv(ag)
        columns = list(df.columns)
        n_cols = ["nB01", "nB02", "nB03", "nB04", "nB05", "nB06", "nB07", "nB08", "nB8A", "nB09", "nB11", "nB12"]
        columns = columns + n_cols

        dest = pd.DataFrame(columns=columns)

        row_offset = [-1,0,1]
        col_offset = [-1,0,1]

        for index, row in df.iterrows():
            the_row = row["row"]
            the_column = row["column"]
            the_scene = None
            if scene_fusion:
                the_scene = row["scene"]

            neighbours = None

            for ro in row_offset:
                for co in col_offset:
                    if ro == 0 and co == 0:
                        continue
                    target_row = the_row + ro
                    target_col = the_column + co
                    if scene_fusion:
                        filter = df[(df["row"] == target_row) & (df["column"] == target_col) & (df["scene"] == the_scene)]
                    else:
                        filter = df[(df["row"] == target_row) & (df["column"] == target_col)]
                    if len(filter) == 0:
                        continue

                    if neighbours is None:
                        neighbours = filter
                    else:
                        neighbours = pd.concat((neighbours,filter), axis=0)

            if neighbours is None:
                continue

            new_row = {}
            for column in df.columns:
                new_row[column] = row[column]

            for ncol in n_cols:
                band = ncol[1:]
                new_row[ncol] = neighbours[band].mean()

            df_dictionary = pd.DataFrame([new_row])
            dest = pd.concat([dest, df_dictionary], ignore_index=True)

        dest = dest.round(4)
        dest.to_csv(grid, index=False)