# %% setup
from tqdm import tqdm
from pypcd import pypcd
import colorsys
import numpy as np
import functools
import time
from sklearn.cluster import KMeans
from dash import Dash, Input, Output, html, dcc, State
import dash_bootstrap_components as dbc
from pathlib import Path
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go

import pandas as pd


def color_kmeans(rgb, n_clusters=10, subsample=25):
    km = KMeans(n_clusters=n_clusters, n_init=1).fit(rgb[::subsample])

    colors = km.cluster_centers_
    clusters = km.predict(rgb)
    rc, gc, bc = colors[clusters].round(0).astype(int).T

    return rc, gc, bc


def draw_cube(x_min, x_max):
    xmin, ymin, zmin = x_min
    xmax, ymax, zmax = x_max

    # Xyz of cube points
    x = [
        xmin,
        xmax,
        xmax,
        xmin,
        xmin,
        xmax,
        xmax,
        xmin,
    ]
    y = [
        ymin,
        ymin,
        ymax,
        ymax,
        ymin,
        ymin,
        ymax,
        ymax,
    ]
    z = [
        zmin,
        zmin,
        zmin,
        zmin,
        zmax,
        zmax,
        zmax,
        zmax,
    ]

    # indexes of edges i_1-i_2
    i_1 = [0, 1, 2, 3, 0, 4, 5, 6, 7, 1, 6, 3]
    i_2 = [1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 2, 7]

    # Points
    Xe = []
    Ye = []
    Ze = []
    for i1, i2 in zip(i_1, i_2):
        Xe.extend([x[i1], x[i2], None])
        Ye.extend([y[i1], y[i2], None])
        Ze.extend([z[i1], z[i2], None])

    # Define the trace for triangle sides
    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode="lines",
        showlegend=False,
        hoverinfo="skip",
        line=dict(color="#000000", width=3),
    )
    return lines


def plot_discrete_shaded(
    df,
    discrete_col,
    colors,
    shade_col="gray",
    point_contour=dict(),
    text_col=None,
    hover_data=None,
):
    """Use one color for shading and another for discrete coloring, shade the discrete colors accordingly"""

    fig = go.Figure(layout=go.Layout())

    if shade_col == "gray" and shade_col not in df.columns:
        df["gray"] = df[["r", "g", "b"]].astype(float).to_numpy().mean(axis=-1)

    if text_col is None:
        text_col = discrete_col

    data = []
    for c in df[discrete_col].unique():
        if c not in colors:
            continue
        class_places = df[discrete_col] == c
        color = colors[c].lower()

        int_color = int(color.replace("#", ""), 16)

        rgb = (
            np.array(
                [
                    (int_color >> 16) & 0xFF,
                    (int_color >> 8) & 0xFF,
                    (int_color) & 0xFF,
                ]
            )
            / 255
        )

        h, s, v = colorsys.rgb_to_hsv(*rgb)

        rgb_dimmed = (np.array(colorsys.hsv_to_rgb(h, s, 0.3)) * 255).astype(int)
        rgb_bright = (np.array(colorsys.hsv_to_rgb(h, s, 1)) * 255).astype(int)

        color = f"rgb({ rgb_dimmed[0] },{ rgb_dimmed[1] },{ rgb_dimmed[2] })"
        other_color = f"rgb({ rgb_bright[0] },{ rgb_bright[1] },{ rgb_bright[2] })"
        # other_color = "#{:06X}".format( int(rgb_bright) )

        hover_cols = [k for k, use_k in hover_data.items() if use_k]
        hover_formats = []
        for col in hover_cols:
            if isinstance(hover_data[col], str):
                hover_formats.append(hover_data[col])
            else:
                hover_formats.append("")
        hover_templates = [
            f"{col}: %{{customdata[{i}]{fmt}}}"
            for i, (col, fmt) in enumerate(zip(hover_cols, hover_formats))
        ]
        plot = go.Scatter3d(
            x=df.x.loc[class_places],
            y=df.y.loc[class_places],
            z=df.z.loc[class_places],
            mode="markers",
            text=df.loc[class_places, text_col],
            customdata=df.loc[class_places, hover_cols],
            hovertemplate="<br>".join(hover_templates) + " <extra></extra>",
            name=c,
            marker=dict(
                size=3,
                line=point_contour,
                color=df[shade_col].loc[class_places],
                colorscale=[color, other_color],
                opacity=1,
            ),
        )
        data.append(plot)

    fig.add_traces(data)
    return fig


def plot_semantic_errors(
    df,
    column="class_errors",
    pred_col="class_pred",
    point_contour=dict(),
    plot_wrongs=False,
):
    gray = df[["r", "g", "b"]].to_numpy().mean(axis=-1)
    df["gray"] = gray

    wrong = df[column]
    right = ~df[column]

    rights = go.Scatter3d(
        x=df.x.loc[right],
        y=df.y.loc[right],
        z=df.z.loc[right],
        mode="markers",
        text=df.loc[right, "class_gt"],
        hovertemplate="<b>Correct: %{text}<b><extra></extra>",
        name="Correct Prediction",
        marker=dict(
            size=4,
            line=point_contour,
            color=df.loc[right, "gray"],
            colorscale=["#005555", "#22ff22"],
            opacity=1,
        ),
    )

    correction_txt = df.loc[wrong, pred_col]
    preds = go.Scatter3d(
        x=df.x.loc[wrong],
        y=df.y.loc[wrong],
        z=df.z.loc[wrong],
        text=correction_txt,
        customdata=df.loc[wrong, "class_gt"].to_numpy(),
        hovertemplate="Predicted  : <b>%{text}</b><br>True Label: <b>%{customdata}</b><extra></extra>",
        name="Incorrect Prediction",
        mode="markers",
        marker_symbol="circle",
        marker=dict(
            size=4,
            line=point_contour,
            color=df.loc[wrong, "gray"],
            colorscale=["#991212", "#ff4444"],
            opacity=1,
        ),
    )

    if plot_wrongs:
        return (rights, preds)
    else:
        return (rights,)


def plot_rgb_clustered(df):
    rc, gc, bc = color_kmeans(df[["r", "g", "b"]].to_numpy())
    color_strs = ["rgb({},{},{})".format(r, g, b) for r, g, b in zip(rc, gc, bc)]
    trace = go.Scatter3d(
        x=df.x,
        y=df.y,
        z=df.z,
        mode="markers",
        marker=dict(
            size=3,
            color=color_strs,
            opacity=1,
        ),
    )
    return trace


def plot_discrete_scatter(
    df,
    color,
    color_discrete_map="category20",
    point_contour=dict(),
    shade_col="gray",
    hover_data=None,
):
    df_ = df.copy()
    df_[color] = df_[color].astype(str)

    hover_data[color] = True

    if shade_col == "None":
        fig = px.scatter_3d(
            df_,
            x="x",
            y="y",
            z="z",
            color=color,
            color_discrete_map=color_discrete_map,
            hover_data=hover_data,
        )
    else:
        fig = plot_discrete_shaded(
            df_,
            discrete_col=color,
            colors=color_discrete_map,
            point_contour=point_contour,
            shade_col=shade_col,
            hover_data=hover_data,
        )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=100))
    fig.update_traces(
        marker=dict(
            size=3,
            line=point_contour,
        )
    )
    return fig


def plot_continuous_scatter(df, color, point_contour=dict(), hover_data={}):
    df[color] = df[color].astype(float)
    hover_data[color] = True
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color=color,
        color_continuous_scale="viridis",
        hover_data=hover_data,
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=100))
    fig.update_traces(
        marker=dict(
            size=3,
            line=point_contour,
        )
    )
    return fig


def get_PCA_transform(pcloud, label_col="label", floor_label=7, all=False):
    # Centralize clouds
    pcloud["x"] = pcloud["x"] - pcloud["x"].mean()
    pcloud["y"] = pcloud["y"] - pcloud["y"].mean()
    pcloud["z"] = pcloud["z"] - pcloud["z"].mean()

    # Isolate the "Floor" of each scene, if it has a floor
    if not all:
        floor_pclouds = pcloud.query(f"{label_col}=={floor_label}")
        if len(floor_pclouds) == 0:
            floor_pclouds = pcloud
    else:
        floor_pclouds = pcloud

    # Calculate the principal axis of the floor on the X-Y plane
    # (Want to align all scenes to a 'common' orientation)
    u, s, vt = np.linalg.svd(floor_pclouds[["x", "y"]].to_numpy(), full_matrices=False)

    # Rotate the scene so the principal axis aligns with the XY axis
    rotation = vt
    transform = np.eye(4)
    transform[:2, :2] = rotation
    return transform


def apply_transform(pcloud, transform):
    pcloud = pcloud.copy()
    # Add a new column for the homogeneous transform
    transform_pcloud = np.ones((pcloud.shape[0], 4))
    transform_pcloud[:, :-1] = pcloud[["x", "y", "z"]].to_numpy()

    # Matrix multiply to rotate the scene
    pcloud_out = transform @ transform_pcloud.T
    pcloud_out = pcloud_out.T[:, :-1]
    pcloud[["x", "y", "z"]] = pcloud_out

    return pcloud


def align_pca(pcloud, label_col="label", floor_label=7, all=False):
    transform = get_PCA_transform(pcloud, label_col, floor_label, all)

    pcloud = apply_transform(pcloud, transform)

    return pcloud


def pcdToDataFrame(pcd):
    cloud = pypcd.PointCloud.from_path(pcd)

    rgb = pd.DataFrame(
        pypcd.decode_rgb_from_pcl(cloud.pc_data["rgb"]), columns=["r", "g", "b"]
    )

    df = pd.DataFrame(cloud.pc_data)
    df["r"] = rgb["r"]
    df["g"] = rgb["g"]
    df["b"] = rgb["b"]
    return df


def filter_labels_from_multiple_files(
    label, files, label_col="label", instance_col="instance"
):
    """Get instances of `label` from multiple point clouds and display them as a grid"""

    labels = []
    last_id = 0
    for file in tqdm(files):
        # Read pcd
        df = pcdToDataFrame(str(file))

        # Get points from `label`
        filt = df.query(f"{label_col}==@label").reset_index(drop=True).copy()
        if len(filt) > 0:
            # Unique ID for each instance of each file
            filt["instance_id"] = pd.factorize(filt[instance_col])[0] + last_id
            last_id = filt["instance_id"].max() + 1

            # Convert scene name into scene ID
            filt["scene"] = Path(file).stem
            labels.append(filt)

    instances = pd.concat(labels).reset_index(drop=True)
    # round up the square
    square = np.ceil(np.sqrt(instances["instance_id"].max() + 1))
    instances["grid_i"] = instances["instance_id"] // square
    instances["grid_j"] = instances["instance_id"] % square

    def center_instance_coords(df):
        return df[["x", "y", "z"]] - df[["x", "y", "z"]].min()

    xyz_cent = instances.groupby("instance_id").apply(center_instance_coords)

    instances[["x", "y", "z"]] = xyz_cent[["x", "y", "z"]].to_numpy()

    instance_groups = instances.groupby("instance_id")
    # Size of each instance
    ranges = instance_groups.max()[["x", "y"]] - instance_groups.min()[["x", "y"]]

    # Maximum size of an instance
    cell_x, cell_y = ranges.max()
    # cell size on x and cell size on y
    cell_x = cell_x * 1.5
    cell_y = cell_y * 1.5

    # Shift instances to their cell position
    instances["x"] = instances["x"] + instances["grid_j"] * cell_x
    instances["y"] = instances["y"] + instances["grid_i"] * cell_y

    return instances


def get_cm_samples_from_files(info_df, folder, file_format, instance_col="instance"):
    """Get instances of `label` from multiple point clouds and display them as a grid"""

    labels = []
    last_id = 0
    i = 0
    instances_info = []
    for file, data in tqdm(info_df.groupby("scene")):
        pcloud = pcdToDataFrame(
            f"{folder}/{file}{ '.' if file_format[0]!='.' else '' }{file_format}"
        )

        for instance, instance_data in data.groupby(instance_col):
            filt = pcloud.query(f"{instance_col}=={instance}")
            if len(filt) > 0:
                i += 1
                # Unique ID for each instance of each file
                instance_id = pd.factorize(filt[instance_col])[0] + last_id

                filt["instance_id"] = instance_id
                last_id = filt["instance_id"].max() + 1

                # Convert scene name into scene ID
                filt["Scene"] = file
                labels.append(filt)

                new_data = instance_data.copy()
                new_data["Instance Id"] = instance_id[0]
                new_data["Scene"] = file
                instances_info.append(new_data)
        if i > 100:
            print("Warning: more than 100 instances... capping at 100")
            break

    instances = pd.concat(labels).reset_index(drop=True)
    # round up the square
    square = np.ceil(np.sqrt(instances["instance_id"].max() + 1))
    instances["grid_i"] = instances["instance_id"] // square
    instances["grid_j"] = instances["instance_id"] % square

    def center_instance_coords(df):
        return df[["x", "y", "z"]] - df[["x", "y", "z"]].min()

    xyz_cent = instances.groupby("instance_id").apply(center_instance_coords)

    instances[["x", "y", "z"]] = xyz_cent[["x", "y", "z"]].to_numpy()

    instance_groups = instances.groupby("instance_id")
    # Size of each instance
    ranges = instance_groups.max()[["x", "y"]] - instance_groups.min()[["x", "y"]]

    # Maximum size of an instance
    cell_x, cell_y = ranges.max()
    # cell size on x and cell size on y
    cell_x = np.clip(cell_x * 1.5, 2, 5)
    cell_y = np.clip(cell_y * 1.5, 2, 5)

    # Shift instances to their cell position
    instances["x"] = instances["x"] + instances["grid_j"] * cell_x
    instances["y"] = instances["y"] + instances["grid_i"] * cell_y

    instances_info = pd.concat(instances_info)
    instances_info = instances_info.rename(
        columns={"Instance Id": "Row", "iou": "IoU", "confidence": "Confidence"}
    )
    instances = instances.rename(columns={"instance_id": "Row"})
    return instances, instances_info[["Row", "Scene", "IoU", "Confidence"]]
