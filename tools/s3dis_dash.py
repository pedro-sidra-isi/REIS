# %% setup
from pathlib import Path
import patoolib
import tempfile
import numpy as np
import gdown
import pandas as pd
import pathlib
from pypcd import pypcd
from reis.generic_dash import InferPointCloudDashboard
import argparse
import diskcache

cache = diskcache.Cache("./cache")

S3DIS_CLASSES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "chair",
    "table",
    "bookcase",
    "sofa",
    "board",
    "clutter",
    "background",
)


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

    # Get rid of small instances (noise), since i used the wrong config for some files
    unique_pred, pred_count = np.unique(df["instance_pred"], return_counts=True)
    df["instance_pred"].loc[
        df["instance_pred"].isin(unique_pred[pred_count < 100])
    ] = -1

    label_to_class = dict(list(enumerate(S3DIS_CLASSES)))

    # Create string columns for classifications
    df["class_gt"] = df["semantic_gt"].astype(int).map(label_to_class)
    df["class_pred"] = df["semantic_pred"].astype(int).map(label_to_class)

    # Instance is encoded as 1000*label + id
    df["class_instance_pred"] = (
        (df["instance_pred"].astype(int) // 1000).map(label_to_class).astype(str)
    )
    df["class_instance_pred"].loc[df["instance_pred"] == -1] = "None"

    df["class_errors"] = df["semantic_pred"] != df["semantic_gt"]
    df["class_instance_errors"] = df["class_instance_pred"] != df["class_gt"]

    ids = (
        df.groupby("instance_pred")
        .first()  # get a single row for each predicted instance
        .reset_index()  # Keep the "instance_pred" column
        .groupby("class_instance_pred")  # Get a group for each predicted class
        .apply(
            lambda df: df.reset_index()
        )  # Return a stacked df with the predicted class as a column
        .loc[
            :, "instance_pred"
        ]  # We want to map each "instance_pred" to a sequential id class-by-class
        .reset_index(1)
        .rename(
            columns={"level_1": "object_id"}
        )  # Instances will be mapped sequentially inside each class
    )

    # Helper columns to visualize object ids
    instance_pred_to_object_id = {
        d["instance_pred"]: d["object_id"] for d in ids.to_dict("records")
    }
    df["object_id"] = df["instance_pred"].map(instance_pred_to_object_id)
    df["object_pred"] = (
        df["class_instance_pred"].astype(str) + "_" + df["object_id"].astype(str)
    )

    df = df.drop(columns=["instance_labels", "semantic_gt", "semantic_pred"])
    df = df.reindex(sorted(df.columns), axis=1).reset_index(drop=True)
    return df


if __name__ == "__main__":
    load_scene_func = load_infer_pcd
    args = get_args()

    scenes_folder = args.path

    if scenes_folder is None:
        url = "https://drive.google.com/uc?id=1mO2DiE5oRJEm3BeFO-Q7X_nGqRbqtR5L"
        rar_path = gdown.cached_download(url)
        data_path = Path(rar_path).parent

        patoolib.extract_archive(rar_path, outdir=data_path, interactive=False)

        scenes_folder = str(data_path / "s3dis_bg")

    scenes_format = "pcd"

    kwargs = dict(
        load_scene_func=load_scene_func,
        scenes_folder=scenes_folder,
        scenes_format=scenes_format,
        classes=S3DIS_CLASSES,
    )

    kwargs.update(sem2ins_classes=["ceiling", "floor", "None"])

    app = InferPointCloudDashboard(**kwargs, instance=True)

    app.run()
