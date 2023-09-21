import pandas as pd
import numpy as np
from datetime import datetime
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window
import os
from rasterio.warp import transform
import re
from csv_processor import CSVProcessor


class SceneToCSVs:
    def __init__(self, scene_list, processed_path, source_csv_path):
        self.scene_list = scene_list
        self.processed_path = processed_path
        self.source_csv_path =  source_csv_path

    def create_csvs(self):
        for index, scene in enumerate(self.scene_list):
            dest_clipped_scene_folder_path = os.path.join(self.processed_path, scene)
            clip_path = os.path.join(dest_clipped_scene_folder_path, "clipped")
            csvs_root = os.path.join(dest_clipped_scene_folder_path, "csvs")
            if os.path.exists(csvs_root):
                print(f"csvs dir exist for scene {scene}. Skipping.")
                continue
            else:
                os.mkdir(csvs_root)
            table, columns = self.create_table(clip_path)
            df = pd.DataFrame(data=table, columns=columns)
            df.sort_values(CSVProcessor.get_spatial_columns(df), inplace=True)
            complete_path = os.path.join(csvs_root, "complete.csv")
            ag_path = os.path.join(csvs_root, "ag.csv")
            ml_path = os.path.join(csvs_root, "ml.csv")
            grid_path = os.path.join(csvs_root, "grid.csv")
            df.to_csv(complete_path, index=False)
            CSVProcessor.aggregate(complete_path, ag_path)
            CSVProcessor.make_ml_ready(ag_path, ml_path)
            CSVProcessor.gridify(ml_path, grid_path)
            print(f"Done scene {index+1}: {scene}")

    def create_table(self, clip_path):
        epsg = self.get_epsg()
        bands = self.get_band_list(clip_path)
        df = pd.read_csv(self.source_csv_path)
        df["when"] = SceneToCSVs.get_epoch(df["when"])
        spatial_columns = CSVProcessor.get_spatial_columns(df)
        all_columns = list(df.columns) + spatial_columns + bands
        table = np.zeros((len(df), len(all_columns)))
        data = df.to_numpy()
        table[:,0:data.shape[1]] = data[:,0:data.shape[1]]
        spatial_info_column_start = len(df.columns)
        band_index_start = spatial_info_column_start + len(spatial_columns)
        self.populate_scene_info(table, clip_path, spatial_info_column_start)
        for column_offset, (band, src) in enumerate(self.iterate_bands(clip_path)):
            column_index = band_index_start + column_offset
            for i in range(len(table)):
                if i!=0 and i%1000 == 0:
                    print(f"Done band processing {i+1} ({table.shape[0]}) of band {column_offset+1} ({len(bands)})")
                lon = table[i, 0]
                lat = table[i, 1]
                row, column = self.get_row_col_by_lon_lat(epsg, src, lon, lat)
                window = Window(column, row, 1, 1)
                pixel_value = src.read(1, window=window)
                pixel_value = pixel_value[0,0]
                table[i,column_index] = pixel_value

        return table, all_columns

    def get_epsg(self):
        return CRS.from_epsg(4326)

    @staticmethod
    def get_epoch(str_dates):
        return [int((datetime.strptime(str_date, '%d-%b-%y')).timestamp()) for str_date in str_dates]

    def get_band_list(self, dest_clipped_scene_folder_path):
        return [band for band, src in self.iterate_bands(dest_clipped_scene_folder_path)]

    def iterate_bands(self, dest_clipped_scene_folder_path):
        bands = []
        for band in os.listdir(dest_clipped_scene_folder_path):
            if not band.endswith(".jp2"):
                continue
            parts = band.split(".")
            band_part = parts[0]
            if band_part[0] != 'B':
                continue
            bands.append(band_part)
        bands = sorted(bands, key=lambda x: int(re.findall(r'\d+', x)[0]))

        for band in bands:
            file_name = f"{band}.jp2"
            band_path = os.path.join(dest_clipped_scene_folder_path, file_name)
            with rasterio.open(band_path) as src:
                yield band,src

    def populate_scene_info(self, table, dest_clipped_scene_folder_path, start_index):
        epsg = self.get_epsg()
        ROW_INDEX = start_index
        COLUMN_INDEX = start_index + 1
        src = self.get_src_by_band(dest_clipped_scene_folder_path, "B05")

        for i in range(table.shape[0]):
            lon = table[i, 0]
            lat = table[i, 1]
            row, column = self.get_row_col_by_lon_lat(epsg, src, lon, lat)
            table[i, ROW_INDEX] = row
            table[i, COLUMN_INDEX] = column
            if i != 0 and i % 1000 == 0:
                print(f"Done populating spatial {i + 1} of {table.shape[0]}")

        return table

    @staticmethod
    def get_src_by_band(dest_clipped_scene_folder_path, band):
        file_name = f"{band}.jp2"
        band_path = os.path.join(dest_clipped_scene_folder_path, file_name)
        with rasterio.open(band_path) as src:
            return src

    @staticmethod
    def get_row_col_by_lon_lat(epsg, src, lon, lat):
        pixel_x, pixel_y = transform(epsg, src.crs, [lon], [lat])
        row, column = src.index(pixel_x, pixel_y)
        row = row[0]
        column = column[0]
        row = min(row, src.height - 1)
        column = min(column, src.width - 1)
        return row, column

