import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
import math
import pandas as pd


class Clipper:
    def __init__(self, source, dest, source_csv_path, padding=20):
        self.source = source
        self.dest = dest
        self.source_csv_path = source_csv_path
        self.PADDING = padding

    def get_bounding_box(self, source_csv_path):
        PADDING = 0
        vpd = pd.read_csv(source_csv_path)
        min_x = vpd["lon"].min()
        max_x = vpd["lon"].max()
        min_y = vpd["lat"].max()
        max_y = vpd["lat"].min()
        min_x = min_x - PADDING
        max_x = max_x + PADDING
        min_y = min_y + PADDING
        max_y = max_y - PADDING
        return min_x, min_y, max_x, max_y

    def clip(self):
        min_x, min_y, max_x, max_y = self.get_bounding_box(self.source_csv_path)
        epsg_4326 = CRS.from_epsg(4326)
        with rasterio.open(self.source) as src:
            min_x, min_y, max_x, max_y = transform_bounds(epsg_4326, src.crs, min_x, max_y, max_x, min_y)
            (column, row) = (~src.transform) * (min_x, max_y)
            row = math.floor(row)
            column = math.floor(column)

            row = max(0,row-self.PADDING)
            column = max(0,column-self.PADDING)

            (column_max, row_max) = (~src.transform) * (max_x, min_y)
            row_max = math.ceil(row_max)
            column_max = math.ceil(column_max)

            row_max = min(src.height-1, row_max+self.PADDING)
            column_max = min(src.width-1, column_max+self.PADDING)

            height = row_max - row
            width = column_max - column
            #window = src.window(min_x, min_y, max_x, max_y)
            window = Window(column, row, width, height)

            data = src.read(window=window)
            profile = src.profile
            profile.update({
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, src.transform),
                'nodata': None
            })
            with rasterio.open(self.dest, 'w', **profile) as dst:
                dst.write(data)


if __name__ == "__main__":
    source = r"D:\Data\Tim\Created\Vectis\Higher\SKYWATCH_PHR_PS_20220204T0037_RGBN_Tile_0_0_e031.tif"
    dest = r"D:\out\abc.tif"
    source_csv_path = "data/shorter.csv"
    clipper = Clipper(source, dest, source_csv_path)
    clipper.clip()