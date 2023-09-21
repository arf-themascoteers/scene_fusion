from clipper import Clipper

source = r"D:\Data\Tim\Created\Vectis\Sentinel-2\S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040\S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040.SAFE\GRANULE\L2A_T54HWE_A034619_20220207T003105\IMG_DATA\R20m\T54HWE_20220207T002711_AOT_20m.jp2"
dest = r"data/flood_clipped.tif"
source_csv_path = "data/shorter.csv"
clipper = Clipper(source, dest, source_csv_path)
clipper.clip()