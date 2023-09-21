import os
import rasterio
from rasterio.warp import transform
from rasterio.crs import CRS


def get_src_by_band(dest_clipped_scene_folder_path, band):
    file_name = f"{band}.jp2"
    band_path = os.path.join(dest_clipped_scene_folder_path, file_name)
    with rasterio.open(band_path) as src:
        return src

def get_row_col_by_lon_lat(epsg, src, lon, lat):
    pixel_x, pixel_y = transform(epsg, src.crs, [lon], [lat])
    row, column = src.index(pixel_x, pixel_y)
    row = row[0]
    column = column[0]
    row = min(row, src.height - 1)
    column = min(column, src.width - 1)
    return row, column


if __name__ == "__main__":
    epsg = CRS.from_epsg(4326)
    lons = [142.1348106, 142.1348095, 142.13481, 142.134807, 142.1348079, 142.1348146, 142.1348186, 142.1348275, 142.1348376, 142.1348524]
    lats = [-36.74234796, -36.74236078, -36.7423742, -36.74238929, -36.7424042, -36.74241704, -36.74243226, -36.74244811, -36.74246359, -36.74247529]

    f1 = "data/processed/S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040/clipped"
    x1 = get_src_by_band(f1,"B01")

    f2 = "data/processed/S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511/clipped"
    x2 = get_src_by_band(f2,"B01")

    lon = lons[0]
    lat = lats[0]

    row1, col1 = get_row_col_by_lon_lat(epsg, x1, lon, lat)
    print(row1, col1)

    row2, col2 = get_row_col_by_lon_lat(epsg, x2, lon, lat)
    print(row2, col2)

    print(x1.height, x1.width)
    print(x2.height, x2.width)