import os
from base_path import BASE_PATH
from clipper import Clipper
from s2_bands import S2Bands


class SceneProcessor:
    def __init__(self, scene_list, processed_path, source_csv_path):
        self.scene_list = scene_list
        self.processed_path = processed_path
        self.source_csv_path = source_csv_path

    def create_clips(self):
        for index, scene in enumerate(self.scene_list):
            dest_clipped_scene_folder_path = os.path.join(self.processed_path, scene)
            if os.path.exists(dest_clipped_scene_folder_path):
                print(f"Processed scene dir exists {index + 1}: {scene}. Skipping")
                continue
            os.mkdir(dest_clipped_scene_folder_path)
            clip_path = os.path.join(dest_clipped_scene_folder_path, "clipped")
            os.mkdir(clip_path)
            base = self.get_scene_source(scene)
            self.clip_bands(base, clip_path)
            print(f"Done clipping scene {index+1}: {scene}")

    def clip_file(self, base, src, dest):
        parts = src.split("_")
        band = parts[2]
        source_band_path = os.path.join(base, src)
        dest_band_path = os.path.join(dest, f"{band}.jp2")
        clipper = Clipper(source_band_path, dest_band_path, self.source_csv_path)
        clipper.clip()
        return band

    def clip_band(self, resolution_path, clip_path, target_band):
        for file_name in os.listdir(resolution_path):
            if not file_name.endswith(".jp2"):
                continue
            parts = file_name.split("_")
            band = parts[2]
            if band == target_band:
                self.clip_file(resolution_path, file_name, clip_path)

    def clip_bands(self, base, clip_path):
        bands = []

        resolution_path = os.path.join(base, "R20m")
        for band in S2Bands.get_R20m_bands():
            self.clip_band(resolution_path, clip_path, band)
            bands.append(band)

        resolution_path = os.path.join(base, "R10m")
        for band in S2Bands.get_R10m_bands():
            self.clip_band(resolution_path, clip_path, band)
            bands.append(band)

        resolution_path = os.path.join(base, "R60m")
        for band in S2Bands.get_R60m_bands():
            self.clip_band(resolution_path, clip_path, band)
            bands.append(band)

        return bands

    def get_scene_source(self, scene):
        scene_path = os.path.join(BASE_PATH, scene)
        return SceneProcessor.get_img_data_path(scene_path)

    @staticmethod
    def get_img_data_path(scene_path):
        safe = os.listdir(scene_path)[0]
        safe_path = os.path.join(scene_path, safe)
        granule_path = os.path.join(safe_path,"GRANULE")
        sub = os.listdir(granule_path)[0]
        sub_path = os.path.join(granule_path, sub)
        img_path = os.path.join(os.path.join(sub_path,"IMG_DATA"))
        return img_path

    @staticmethod
    def get_all_scenes():
        scene_list = os.listdir(BASE_PATH)
        scene_list = [scene for scene in scene_list if scene.startswith("S2")
                           and os.path.isdir(os.path.join(BASE_PATH, scene))]
        return scene_list
