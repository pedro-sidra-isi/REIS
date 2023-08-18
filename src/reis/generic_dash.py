# %% setup
import os
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import colorsys
import numpy as np
import json
import functools
from pathlib import Path
from typing import Type, Optional
from dash.exceptions import PreventUpdate
from dash import ctx, dash_table
from reis.confusion import (
    process_confusion_data,
    plot_confusion_matrix,
    instance_confusion_matrix,
    FP_NAME,
    FN_NAME,
    TP_NAME,
)
import time
from sklearn.cluster import KMeans
from dash import Dash, Input, Output, html, dcc, State
import dash_bootstrap_components as dbc
from pathlib import Path
import pandas as pd

## Diskcache
from dash.long_callback import DiskcacheLongCallbackManager

import diskcache

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)


import plotly.express as px
import plotly.graph_objs as go
import reis.pcloud_utils as ut

import pandas as pd

DASH = Dash
# change to jupyter
# from jupyter_dash import JupyterDash
# DASH=JupyterDash


def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))


def hsv_colorscale(size):
    colors = []
    for i in range(size):
        c = hsv2rgb(i / (size + 1), 0.3 + (i % 2) * 0.7, 1)
        colors.append(f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}")
    return colors


# Can only instantiate once
class Singleton(type):
    """Models a class that can only have one object"""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class fake_color_map(object):
    def __getitem__(self, items):
        return hex(items)

    def copy(self):
        return fake_color_map()


the_instance = None


class PointCloudDashboard(metaclass=Singleton):
    app = DASH(__name__, external_stylesheets=[dbc.themes.FLATLY])

    @staticmethod
    def get() -> "PointCloudDashboard":
        global the_instance
        return the_instance

    def __init__(
        self,
        load_scene_func,
        scenes_folder,
        classes,
        scenes_format="",
        colors=None,
        point_contour=False,
        instance=True,
    ):
        global the_instance
        the_instance = self

        if colors is None:
            colors = [
                *hsv_colorscale(len(classes)),
            ]
        if len(classes) != len(colors):
            print(
                "WARNING: colors with different length from classes, {colors=}, {classes=}"
            )

        self.load_scene = load_scene_func
        self.colors_list = colors
        self.colors_map = dict(list(zip(classes, colors)))
        self.colors_map["nan"] = "#000000"
        self.colors_map["None"] = "#000000"
        self.classes = classes
        self.scenes_folder = scenes_folder
        self.scenes_format = scenes_format
        self.title = Path(scenes_folder).stem
        self.point_contour = dict(color="#000000", width=1) if point_contour else dict()
        self.use_instance = instance

        scenes_filetype = scenes_format.split(".")[-1]
        if scenes_filetype and scenes_filetype[0] != ".":
            scenes_filetype = f".{scenes_filetype}"

        self.scenes_filetype = scenes_filetype

        files = sorted(list(Path(scenes_folder).glob(f"*{scenes_filetype}")))
        scenes = list(str(p) for p in files)
        scene_names = list(p.name for p in files)

        self.scenes_df = pd.DataFrame(dict(filename=scenes, name=scene_names))

        self.scenes_df.rename(
            columns={"name": "label", "filename": "value"}, inplace=True
        )

        self.scene_options = self.scenes_df[["label", "value"]].to_dict("records")

        self.class_options = [
            {
                "label": html.Div(
                    [
                        html.Div(str(_class), style={"margin-left": "5px"}),
                        html.Div(
                            "",
                            style={
                                "background-color": c,
                                "border-radius": "100%",
                                "padding": "10px",
                                "margin": "2px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justify-content": "space-between",
                        "width": "100%",
                    },
                ),
                "value": _class,
            }
            for _class, c in self.colors_map.items()
        ]

        self.setup_fixed_columns()
        self.setup_interface()

    def setup_fixed_columns(self):
        self.color_cols = ["rgb", "instance", "class_gt"]
        self.shade_cols = ["gray", "None"]

    def setup_interface(self):
        self.app.layout = self.create_layout()
        self.setup_figure_callback()
        self.setup_control_callbacks()

    def create_controls(self):
        inputgroup_style = {"margin-bottom": "0.5em", "width": "100%"}
        dropdown_style = {}
        inputText_style = {"width": "90px"}

        return [
            html.H4("Controls"),
            dbc.InputGroup(
                style=inputgroup_style,
                children=[
                    dbc.InputGroupText(
                        "Scene",
                        style=inputText_style,
                    ),
                    dbc.Select(
                        self.scene_options,
                        self.scene_options[0]["value"],
                        id="scene_dropdown",
                        persistence=True,
                    ),
                    dbc.Button("Next", id="btn_next_scene"),
                ],
            ),
            dbc.InputGroup(
                style=inputgroup_style,
                children=[
                    dbc.InputGroupText(
                        "Color by",
                        style=inputText_style,
                    ),
                    dbc.Select(
                        self.color_cols,
                        "class_gt",
                        id="color_dropdown",
                        persistence=True,
                    ),
                    dbc.Button("Next", id="btn_next_color"),
                ],
            ),
            dbc.InputGroup(
                style=inputgroup_style,
                children=[
                    dbc.InputGroupText(
                        "Shade by",
                        style=inputText_style,
                    ),
                    dbc.Select(
                        self.shade_cols,
                        self.shade_cols[0],
                        id="shade_dropdown",
                        persistence=True,
                    ),
                ],
            ),
            dbc.Label("Hide Classes"),
            dbc.Row(
                style={"align-items": "center"},
                children=[
                    dbc.Col(
                        children=dcc.Dropdown(
                            self.class_options,
                            id="filter_list",
                            persistence=True,
                            value=[],
                            multi=True,
                            style=dropdown_style,
                        ),
                        md=9,
                    ),
                    dbc.Col(
                        children=dbc.Button(
                            "Apply",
                            id="btn_apply_filter",
                            style={"width": "100%", "height": "100%"},
                        ),
                    ),
                ],
            ),
            dbc.Label("Hover Info"),
            dbc.Row(
                style={"align-items": "center"},
                children=[
                    dbc.Col(
                        children=dcc.Dropdown(
                            self.color_cols,
                            id="hover_dropdown",
                            persistence=True,
                            value=[],
                            multi=True,
                            style=dropdown_style,
                        ),
                        md=9,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Apply",
                            id="btn_apply_hover",
                            style={"width": "100%", "height": "100%"},
                        ),
                    ),
                ],
            ),
        ]

    def create_layout(self):
        plot = dcc.Graph(id="scatter_fig", style={"width": "80vw", "height": "80vh"})

        return dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [
                        dbc.Col(self.create_controls(), md=3),
                        dbc.Col(
                            [
                                html.H2(f"Point Cloud Visualization"),
                                html.H3(f"{self.title}"),
                                # html.Hr(),
                                dcc.Loading(plot),
                            ],
                            md=9,
                        ),
                    ],
                    align="top",
                ),
                dcc.Store(id="local_store", storage_type="memory"),
            ],
        )

    def run(self):
        PointCloudDashboard.app.run(
            debug=True,
            host=os.environ.get("HOST", default="localhost"),
            port=os.environ.get("PORT", default=8050),
        )

    @functools.lru_cache(maxsize=3)
    def get_scene(self, scene, key="df"):
        result = self.load_scene(scene)

        if not isinstance(result, dict):
            return result

        if key in result:
            return result[key]
        else:
            return result

    def scene_scatterplot(self, df, color, shade_col, hover_data=None):
        if color == "rgb":
            trace = ut.plot_rgb_clustered(df)
            fig = go.Figure(
                data=[trace], layout=go.Layout(scene=dict(aspectmode="data"))
            )
        else:
            is_discrete = len(df[color].unique()) < 30

            if is_discrete:
                fig = ut.plot_discrete_scatter(
                    df,
                    color,
                    color_discrete_map=self.colors_map,  # type: ignore
                    point_contour=self.point_contour,
                    shade_col=shade_col,
                    hover_data=hover_data,
                )
            else:
                fig = ut.plot_continuous_scatter(
                    df, color, point_contour=self.point_contour, hover_data=hover_data
                )

        return fig

    def setup_figure_callback(self):
        @PointCloudDashboard.app.long_callback(
            Output(component_id="scatter_fig", component_property="figure"),
            Input(component_id="color_dropdown", component_property="value"),
            State(component_id="filter_list", component_property="value"),
            Input(component_id="scene_dropdown", component_property="value"),
            Input(component_id="shade_dropdown", component_property="value"),
            State(component_id="hover_dropdown", component_property="value"),
            Input(component_id="btn_apply_hover", component_property="n_clicks"),
            Input(component_id="btn_apply_filter", component_property="n_clicks"),
            prevent_initial_call=True,
            manager=long_callback_manager,
        )
        def figure_callback(color, class_filter, scene, shade_col, hover_list, _, __):
            start = time.time()

            fig = PointCloudDashboard.draw_point_cloud_scene(
                scene, color, class_filter, shade_col, hover_list
            )
            print(f"Fig build took {time.time()-start} seconds")
            fig.update_layout(
                title=dict(text=f"{Path(scene).stem}, {color}"), uirevision=scene
            )

            return fig

    def draw_point_cloud_scene(scene, color, class_filter, shade_col, hover_list):
        self = PointCloudDashboard.get()
        df = self.get_scene(scene)  # type:ignore
        fig = PointCloudDashboard.draw_point_cloud(
            df, color, class_filter, shade_col, hover_list
        )
        return fig

    def draw_point_cloud(df, color, class_filter, shade_col, hover_list):
        self = PointCloudDashboard.get()
        df = df.copy()

        hover_data = {col: False for col in df.columns}
        for col in hover_list:
            if "float" in str(df[col].dtype):
                hover_data[col] = ":.2f"
            else:
                hover_data[col] = True

        # Filter points
        df = df.loc[~df["class_gt"].isin(class_filter)]  # type:ignore
        # Plot
        fig = self.scene_scatterplot(
            df, color=color, shade_col=shade_col, hover_data=hover_data
        )  # type:ignore

        fig.update_layout(
            scene=dict(aspectmode="data"),
            autosize=True,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=16),
                itemsizing="constant",
            ),
        )

        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        return fig

    def setup_control_callbacks(self):
        @PointCloudDashboard.app.callback(
            Output(component_id="scene_dropdown", component_property="value"),
            State(component_id="scene_dropdown", component_property="options"),
            State(component_id="scene_dropdown", component_property="value"),
            Input(component_id="btn_next_scene", component_property="n_clicks"),
            prevent_initial_call=False,
        )
        def next_scene(options, value, btn):
            self = PointCloudDashboard.get()
            if btn:
                list_options = [d["value"] for d in options]  # type:ignore
                i_cur = list_options.index(value)
                o = options[i_cur + 1]["value"]  # type:ignore
                return o
            else:
                return value

        @PointCloudDashboard.app.callback(
            Output(component_id="hover_dropdown", component_property="options"),
            Input(component_id="scene_dropdown", component_property="value"),
        )  # type:ignore
        def update_hover_options(scene):
            self = PointCloudDashboard.get()
            df = self.get_scene(scene)  # type:ignore
            return df.columns  # type:ignore

        @PointCloudDashboard.app.callback(
            Output(component_id="color_dropdown", component_property="value"),
            State(component_id="color_dropdown", component_property="options"),
            State(component_id="color_dropdown", component_property="value"),
            Input(component_id="btn_next_color", component_property="n_clicks"),
            prevent_initial_call=True,
        )
        def next_color(options, value, btn):
            i_cur = options.index(value)  # type:ignore
            return options[min(i_cur + 1, len(options) - 1)]  # type:ignore

        @PointCloudDashboard.app.callback(
            Output(component_id="color_dropdown", component_property="options"),
            Input(component_id="scene_dropdown", component_property="value"),
        )
        def update_scalar_fields(scene):
            self = PointCloudDashboard.get()
            df = self.get_scene(scene)  # type:ignore

            unintersting_cols = {"x", "y", "z", "r", "g", "b"}
            cols = [c for c in df.columns if c not in unintersting_cols]  # type:ignore

            return cols


class InferPointCloudDashboard(PointCloudDashboard):
    def __init__(self, sem2ins_classes=[], *args, **kwargs):
        self.sem2ins_classes = sem2ins_classes
        self.instance = kwargs.get("instance")

        super().__init__(*args, **kwargs)

        if self.instance:
            self.load_info(
                scenes_folder=kwargs.get("scenes_folder"), classes=kwargs.get("classes")
            )
        else:
            self.infer_info = pd.DataFrame()
            self.confusion_info = pd.DataFrame()
            self.mtx = np.zeros((10, 10))
            self.infer_info = pd.DataFrame()

    def setup_fixed_columns(self):
        super().setup_fixed_columns()
        self.shade_cols = ["gray", "semantic_pred_confs", "None"]

    def setup_interface(self):
        PointCloudDashboard.app.layout = self.create_layout()
        self.setup_control_callbacks()
        self.setup_figure_callback()
        self.setup_confusion_callbacks()

    def create_layout(self):
        controls = super().create_controls()

        control_tab = dbc.Tab(
            label="Controls", children=controls, style={"justify-content": "center"}
        )

        side_col_members = [control_tab]
        if self.instance:
            confusion_tab = dbc.Tab(
                label="Confusion Matrix",
                children=dcc.Graph(
                    id="confusion_mtx",  # config={"displayModeBar": False}
                    # style={"width": "70vw", "height": "90vh"},
                ),
            )
            side_col_members.append(confusion_tab)

        side_col_members.append(html.H4("Instances", id="instances_label"))
        side_col_members.append(
            dash_table.DataTable(
                [{}],
                id="confusion_table",
            )
        )

        side_col = dbc.Col(dbc.Card(children=side_col_members, body=True), md=3)

        main_view = dcc.Graph(
            id="scatter_fig",
            style={"width": "70vw", "height": "90vh"},
        )

        figure_col = dbc.Col(
            [
                html.H1(f"Point Cloud Inference Visualization"),
                html.H4(f"{self.title}"),
                html.Hr(),
                dbc.Row(
                    [
                        dcc.Loading([main_view]),
                        # dcc.Loading([confusion_tab]),
                    ]
                ),
            ],
            md=9,
        )

        return dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    [side_col, figure_col],
                    align="top",
                ),
                dcc.Store(id="local_store", storage_type="memory"),
            ],
        )

    def scene_scatterplot(self, df, color, shade_col, hover_data=None):
        self = PointCloudDashboard.get()
        fig = None

        if color == "class_errors" or color == "class_instance_errors":
            pred_col = (
                "class_pred" if color == "class_errors" else "class_instance_pred"
            )
            traces = ut.plot_semantic_errors(
                df,
                pred_col=pred_col,
                column=color,
                point_contour=self.point_contour,
                plot_wrongs=True,
            )

            fig = go.Figure(
                layout=go.Layout(
                    scene=dict(aspectmode="data"),
                )
            )
            fig.add_traces(traces)

        if color == "object_pred":
            df_ = df  # .query(
            #     "class_instance_pred not in @self.sem2ins_classes"
            # )
            transform = ut.get_PCA_transform(df_, label_col="`class_gt`", all=True)
            df_ = ut.apply_transform(df_, transform)

            fig = ut.plot_discrete_shaded(
                df_,
                discrete_col="class_instance_pred",
                colors=self.colors_map,  # type:ignore
                point_contour=self.point_contour,  # type:ignore
                shade_col=shade_col if shade_col != "None" else "gray",
                hover_data=hover_data,
            )

            df_ = df_.query("class_instance_pred not in @self.sem2ins_classes")
            min = df_.groupby("object_pred").min()[["x", "y", "z"]].to_numpy()
            max = df_.groupby("object_pred").max()[["x", "y", "z"]].to_numpy()

            traces = []
            for i_min, i_max in zip(min, max):
                cube = ut.draw_cube(i_min, i_max)
                fig.add_trace(cube)

        if fig is None:
            return super().scene_scatterplot(
                df, color, shade_col=shade_col, hover_data=hover_data
            )
        else:
            return fig

    def get_confusion_info(self, f, classes):
        df = self.get_scene(f)  # type:ignore
        df["instance_pred"] = df["instance_pred"].fillna(-1)
        df["instance_gt"] = df["instance_gt"].fillna(-1)
        _, mapping = instance_confusion_matrix(
            df["instance_pred"],
            df["instance_gt"],
            iou_threshold=0.25,
            n_semantic_classes=len(classes),
            skip_labels=self.sem2ins_classes,
            prediction_confidences=df["semantic_pred_confs"],
        )
        info = pd.DataFrame(mapping)
        info["scene"] = f.stem
        return info

    def load_info(self, scenes_folder, classes):
        files = sorted(list(Path(scenes_folder).glob(f"*{self.scenes_filetype}")))
        mappings = []
        print("[INFO] building confusion matrix....")

        with Pool() as p:
            get_confusion_info_func = partial(self.get_confusion_info, classes=classes)
            mapper = p.imap(get_confusion_info_func, files)
            mappings = list(tqdm(mapper, total=len(files)))

        self.infer_info = pd.concat(mappings)

        self.confusion_info, self.mtx, self.infer_info = process_confusion_data(
            self.infer_info,
            class_map={i: c for i, c in enumerate(classes)},  # type:ignore
            sem2ins_classes=self.sem2ins_classes,
        )

    def display_click_data(self, gt, pred, color, class_filter, shade_col, hover_list):
        self = PointCloudDashboard.get()
        subset = self.infer_info.query(f"`true`==@gt and pred==@pred")

        point_clouds, instance_info = ut.get_cm_samples_from_files(
            subset,
            folder=self.scenes_folder,
            file_format=self.scenes_format,
            instance_col="instance_gt" if gt != FP_NAME else "instance_pred",
        )

        hover_list.append("Row")
        hover_list.append("Scene")
        label_to_class = {label: cls for label, cls in enumerate(self.classes)}
        point_clouds["class_gt"] = (
            point_clouds["semantic_gt"].astype(int).map(label_to_class)
        )
        point_clouds["class_pred"] = (
            point_clouds["semantic_pred"].astype(int).map(label_to_class)
        )
        point_clouds["class_instance_pred"] = (
            (point_clouds["instance_pred"].astype(int) // 1000)
            .map(label_to_class)
            .astype(str)
        )
        fig = PointCloudDashboard.draw_point_cloud(
            point_clouds, color, class_filter, shade_col, hover_list
        )

        fig.update_layout(
            scene=dict(aspectmode="data"),
            title=dict(text=f"GT={gt} ; Pred={pred}", xanchor="center", x=0.5),
            autosize=True,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

        return fig, instance_info.round(2).to_dict("records")

    def setup_confusion_callbacks(self):
        if not self.instance:
            return

        @PointCloudDashboard.app.long_callback(
            Output(component_id="confusion_mtx", component_property="figure"),
            Input(component_id="scene_dropdown", component_property="value"),
            manager=long_callback_manager,
        )
        def update_cmat(scene):
            self = PointCloudDashboard.get()
            fig = plot_confusion_matrix(
                self.mtx, class_names=self.classes
            )  # type:ignore
            return fig

    def setup_figure_callback(self):
        callback_signature = [
            [
                Output(component_id="scatter_fig", component_property="figure"),
                Output(component_id="confusion_table", component_property="data"),
                Output(component_id="instances_label", component_property="children"),
            ],
            [
                Input(component_id="color_dropdown", component_property="value"),
                State(component_id="filter_list", component_property="value"),
                Input(component_id="scene_dropdown", component_property="value"),
                Input(component_id="shade_dropdown", component_property="value"),
                State(component_id="hover_dropdown", component_property="value"),
                Input(component_id="btn_apply_hover", component_property="n_clicks"),
                Input(component_id="btn_apply_filter", component_property="n_clicks"),
            ],
        ]

        if self.instance:
            callback_signature.append(Input("confusion_mtx", "clickData"))

        @PointCloudDashboard.app.long_callback(
            *callback_signature,
            prevent_initial_call=True,
            manager=long_callback_manager,
        )
        def update_figure(
            color, class_filter, scene, shade_col, hover_list, confusion_click, *args
        ):
            self = PointCloudDashboard.get()
            start = time.time()

            table_data = [{}]
            instances_label = "Instances"
            if "confusion_mtx" in ctx.triggered[0]["prop_id"]:
                confusion_click = ctx.triggered[0]["value"]

                if confusion_click:
                    gt = confusion_click.get("points")[0].get("y")
                    pred = confusion_click.get("points")[0].get("x")
                else:
                    raise PreventUpdate

                fig, table_data = self.display_click_data(
                    gt, pred, color, class_filter, shade_col, hover_list
                )
                instances_label = f"GT: {gt}, Predicted: {pred}"
                fig.update_layout(
                    title=dict(text=instances_label, xanchor="center", x=0.5),
                    uirevision=0,
                )
            else:
                fig = PointCloudDashboard.draw_point_cloud_scene(
                    scene, color, class_filter, shade_col, hover_list
                )
                fig.update_layout(
                    title=dict(
                        text=f"{Path(scene).stem}, {color}", xanchor="center", x=0.5
                    ),
                    uirevision=scene,
                )

            print(f"Fig build took {time.time()-start} seconds")
            return fig, table_data, instances_label

    def setup_control_callbacks(self):
        @PointCloudDashboard.app.callback(
            Output(component_id="scene_dropdown", component_property="value"),
            State(component_id="scene_dropdown", component_property="options"),
            State(component_id="scene_dropdown", component_property="value"),
            Input(component_id="btn_next_scene", component_property="n_clicks"),
            Input(component_id="confusion_table", component_property="active_cell"),
            State(component_id="confusion_table", component_property="data"),
            prevent_initial_call=False,
        )
        def next_scene(options, value, btn, active_cell, table_data):
            self = PointCloudDashboard.get()
            if active_cell:
                scene_label = (
                    table_data[active_cell["row"]]["Scene"] + self.scenes_filetype
                )
                values = [d["value"] for d in options]  # type:ignore
                labels = [d["label"] for d in options]  # type:ignore
                return values[labels.index(scene_label)]
            if btn:
                list_options = [d["value"] for d in options]  # type:ignore
                i_cur = list_options.index(value)
                o = options[i_cur + 1]["value"]  # type:ignore
                return o
            else:
                return value

        # Same as super(), but Dash is dumb
        @PointCloudDashboard.app.callback(
            Output(component_id="hover_dropdown", component_property="options"),
            Input(component_id="scene_dropdown", component_property="value"),
        )  # type:ignore
        def update_hover_options(scene):
            self = PointCloudDashboard.get()
            df = self.get_scene(scene)  # type:ignore
            return df.columns  # type:ignore

        @PointCloudDashboard.app.callback(
            Output(component_id="color_dropdown", component_property="value"),
            State(component_id="color_dropdown", component_property="options"),
            State(component_id="color_dropdown", component_property="value"),
            Input(component_id="btn_next_color", component_property="n_clicks"),
            prevent_initial_call=True,
        )
        def next_color(options, value, btn):
            i_cur = options.index(value)  # type:ignore
            return options[min(i_cur + 1, len(options) - 1)]  # type:ignore

        @PointCloudDashboard.app.callback(
            Output(component_id="color_dropdown", component_property="options"),
            Input(component_id="scene_dropdown", component_property="value"),
        )
        def update_scalar_fields(scene):
            self = PointCloudDashboard.get()
            df = self.get_scene(scene)  # type:ignore

            unintersting_cols = {"x", "y", "z", "r", "g", "b"}
            cols = [c for c in df.columns if c not in unintersting_cols]  # type:ignore

            return cols
