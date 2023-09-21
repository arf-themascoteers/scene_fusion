import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


class Reconstructor:
    @staticmethod
    def recon(csv, height=None, width=None, save=True, pad=False):
        df = None
        if isinstance(csv, str):
            df = pd.read_csv(csv)
        else:
            df = csv
        if "scene" in df.columns:
            df = df[df["scene"] == 1]
        if height is None or width is None:
            max_row = df["row"].max()
            max_col = df["column"].max()
            height = int(max_row)
            width = int(max_col)
        x = np.zeros((height+1,width+1),dtype=np.float64)
        for i in df.index:
            row = df.loc[i,"row"]
            col = df.loc[i,"column"]
            pix = df.loc[i,"B03"]
            row = int(row)
            col = int(col)
            if pad:
                x[row-5:row+5,col-5:col+5] = pix
            else:
                x[row,col] = pix
        plt.imshow(x)
        file_name = os.path.basename(csv)
        if save:
            plt.savefig(f"plots/{file_name}.png")
        else:
            plt.show()
        plt.clf()
        return height, width

    @staticmethod
    def recon_folder(folder):
        path = os.path.join(folder,"ag.csv")
        Reconstructor.recon(path)


if __name__ == "__main__":
    Reconstructor.recon("data/processed/S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040/csvs/ag.csv", save=False, pad=False)
    Reconstructor.recon("data/processed/S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511/csvs/ag.csv", save=False, pad=False)