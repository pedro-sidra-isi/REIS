# %% setup
from pathlib import Path
import patoolib
import numpy as np
import gdown
import pandas as pd
from pypcd import pypcd
from reis.generic_dash import PointCloudDashboard
import argparse
import diskcache

cache = diskcache.Cache("./cache")

S3DIS_CLASSES_MAP = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter",
    13: "background",
    14: "None",
}
S3DIS_CLASSES = list(S3DIS_CLASSES_MAP.values())


def get_args():
    parser = argparse.ArgumentParser(
        "Example dashboard for REIS on the s3dis dataset with pre-computed inferences"
    )
    parser.add_argument("--path", help="Run on point clouds in this path")
    return parser.parse_args()


def load_infer_pcd(path):
    cloud = pypcd.PointCloud.from_path(path)
    df = pd.DataFrame(cloud.pc_data)

    # Subsample because too many points
    df = df.iloc[::2]

    rgb = pypcd.decode_rgb_from_pcl(df["rgb"].to_numpy())

    df["r"] = rgb[:, 0].astype(float)
    df["g"] = rgb[:, 1].astype(float)
    df["b"] = rgb[:, 2].astype(float)

    # Solve NaN and other numerical issues
    df["instance_pred"] = df["instance_pred"].fillna(-1)
    df["instance_gt"] = df["instance_gt"].fillna(-1)
    df["instance_pred"].loc[df["instance_pred"] > 4e6] = -1
    df["instance_gt"].loc[df["instance_gt"] > 4e6] = -1
    df["instance_gt"] = df["instance_gt"].fillna(-1)

    # "None"
    df["instance_pred"].loc[df["instance_pred"] == -1] = 14
    df["instance_gt"].loc[df["instance_gt"] == -1] = 14

    df = df.drop(columns=["instance_labels"])

    # Need to return these columns:
    # x,y,z
    # r,g,b
    # semantic_pred
    # semantic_pred_confs
    # semantic_gt
    # instance_pred
    # instance_gt
    return df


if __name__ == "__main__":
    load_scene_func = load_infer_pcd
    args = get_args()

    scenes_folder = args.path

    if scenes_folder is None:
        url = "https://drive.google.com/uc?id=1X8KHGU6e_za4GtB0EfPQj8V5W5secjIZ"
        rar_path = gdown.cached_download(url)
        data_path = Path(rar_path).parent / "s3dis_bg"

        patoolib.extract_archive(rar_path, outdir=data_path, interactive=False)

        scenes_folder = data_path

    scenes_format = "pcd"

    kwargs = dict(
        load_scene_func=load_scene_func,
        scenes_folder=scenes_folder,
        scenes_format=scenes_format,
        classes=S3DIS_CLASSES,
    )

    kwargs.update(sem2ins_classes=["ceiling", "floor", "None"])

    app = PointCloudDashboard(**kwargs, instance=True)

    app.run()
