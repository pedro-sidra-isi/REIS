"""Utilities for plotting confusion matrix visualizations"""
from itertools import product
from typing import Dict, List
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.figure_factory as ff

FN_NAME = "No Prediction"
FP_NAME = "No Match"
TP_NAME = "Correct"


def process_confusion_data(
    inference_info: pd.DataFrame, class_map: Dict, sem2ins_classes: List
):
    """Preprocess `confusion_df` to get confusion classes with names and the confusion matrix

    :param confusion_df: dataframe with the semantic_gt and semantic_pred columns representing a confusion matrix
    :param class_map: dict that maps from class_id to class_name

    returns: (new_df, confusion_matrix), where new_df is the processed df and confusion_matrix is
    an (n_classes+1 x n_classes+1) confusion_matrix (includes false positives and false negatives)
    """
    n_classes = len(class_map)

    # Workaround the NaN entries
    class_map[-1] = "None"

    df = inference_info
    df[["semantic_gt", "semantic_pred"]] = (
        df[["semantic_gt", "semantic_pred"]].fillna(-1).astype(int)
    )

    confusions_df = (
        df.groupby("semantic_gt")["semantic_pred"]  # Get each unique (GT, Pred) pair
        .value_counts()  # Count how much times it appears
        .rename("count")  # type: ignore
        .reset_index()  # Keep semantic_Gt column
    )

    # Build classes columns
    inference_info["pred"] = inference_info["semantic_pred"].map(class_map)
    inference_info["true"] = inference_info["semantic_gt"].map(class_map)
    inference_info["pred"] = inference_info["pred"].str.replace("None", FN_NAME)
    inference_info["true"] = inference_info["true"].str.replace("None", FP_NAME)

    confusions_df["class_pred"] = confusions_df["semantic_pred"].map(class_map)
    confusions_df["class_gt"] = confusions_df["semantic_gt"].map(class_map)
    # FP and FN columns
    confusions_df["class_pred"] = confusions_df["class_pred"].str.replace(
        "None", FN_NAME
    )
    confusions_df["class_gt"] = confusions_df["class_gt"].str.replace("None", FP_NAME)

    # sem2ins doesn't make sense in the instance matrix
    confusions_df = confusions_df.query("class_pred not in @sem2ins_classes")

    # sem2ins do not have FN, remove them
    sem2ins_FN = confusions_df.query(
        "class_gt in @sem2ins_classes and class_pred==@FN_NAME"
    ).index
    confusions_df.loc[sem2ins_FN] = 0

    def norm_count(df):
        df["percent"] = df["count"] / df["count"].sum()
        return df

    confusions_df = (
        confusions_df.groupby("class_gt", group_keys=True)  # Normalize by row
        .apply(norm_count)
        .reset_index(drop=True)
    )

    # Build matrix
    confusion_mat = np.zeros((n_classes + 1, n_classes + 1))

    # %%
    confusions_ = confusions_df.to_numpy()

    # First two columns are indexes
    idxs = confusions_[:, :2].astype(int)
    # Last column is confusion count
    count = confusions_[:, 2].astype(int)

    confusion_mat[idxs[:, 0], idxs[:, 1]] = count

    return confusions_df, confusion_mat, inference_info


def plot_confusion_matrix(confusion_mat, class_names):
    """Use plotly to display `confusion_mat` and return the plotly.fig"""

    # Clear useless classes
    row_eq_0 = np.all(confusion_mat == 0, axis=0)
    col_eq_0 = np.all(confusion_mat == 0, axis=-1)

    # Must have at least one non-zero entry
    useful_idxs = ~(row_eq_0 & col_eq_0)
    mtx_range = np.arange(useful_idxs.shape[0])
    i_s, j_s = np.meshgrid(mtx_range[useful_idxs], mtx_range[useful_idxs])
    # Select rows and cols
    confusion_mat = confusion_mat[j_s, i_s]
    class_names = [
        class_names[i] for i, useful in enumerate(useful_idxs[:-1]) if useful
    ]

    # Create a 'contrast' matrix for color-coding
    contrast_mat = confusion_mat / np.sum(confusion_mat, axis=1)[:, np.newaxis]

    # Make off-diagonal entries negative (this way the diverging colorscale shows another color)
    contrast_mat = contrast_mat * -1
    contrast_mat[np.eye(len(contrast_mat)) == 1] = (
        -1 * contrast_mat[np.eye(len(contrast_mat)) == 1]
    )
    contrast_mat[np.isnan(contrast_mat)] = 0
    contrast_mat[contrast_mat == 0] = np.nan

    # Texts are the original matrix (not contrast_mat)
    texts = confusion_mat.astype(int).astype(str)
    # Don't show zeroes
    texts[texts == "0"] = ""

    # Need to invert the Y axis for plotly
    fig = ff.create_annotated_heatmap(
        contrast_mat[::-1],
        annotation_text=texts[::-1],
        font_colors=["black"],
        # Custom diverging colorscale
        colorscale=["rgb(202, 86, 44)", "rgb(255,255,255)", "rgb(0, 128, 128)"],
        x=[*class_names, FN_NAME],
        y=[*class_names, FP_NAME][::-1],
    )

    fig.update_xaxes(
        automargin=True,
        tickangle=45,
        ticklabelposition="outside bottom",
        showgrid=False,
    )
    fig.update_yaxes(showgrid=False)
    fig.update_layout(
        xaxis_title="Prediction",
        yaxis_title="Ground Truth",
        font_size=12,
        autosize=True,
        margin=dict(l=0, r=0, t=100, b=0),
        plot_bgcolor="white",
    )
    return fig


def mask_iou(mask_a, mask_b):
    """Returns IoU between two binary masks"""
    intersection = np.sum(np.logical_and(mask_a, mask_b))

    # Intersection over Union, |A \cup B| = |A| + |B| - |A \cap B|
    return intersection / (mask_a.sum() + mask_b.sum() - intersection)


def instance_confusion_matrix(
    predicted_instances,
    groundtruth_instances,
    iou_threshold=0.0,
    n_semantic_classes=None,
    skip_labels=[],
    prediction_confidences=None,
):
    """Calculate Instance Detection Matrix, keeping the list of instances belonging to each cell

    :param predicted_instances: instance IDs with semantic labels encoded. `predicted_instances//1000` should give the semantic label
    :param groundtruth_instances: instance IDs with semantic labels encoded. `groundtruth_instances//1000` should give the semantic label

    :return: (Matrix, Mappings), where matrix is a n_semantic_classesxn_semantic_classes confusion matrix, and
             `Mappings` are the instances corresponding to each cell, represented in a list of dicts
    """
    confusion_cell_mappings = []

    # Confidence is optional
    if prediction_confidences is None:
        prediction_confidences = np.ones_like(predicted_instances)

    # Class encoded in the thousands
    predicted_classes = predicted_instances // 1000
    groundtruth_classes = groundtruth_instances // 1000

    # Find number of semantic labels
    max_label = max(np.max(predicted_classes), np.max(groundtruth_classes))
    if n_semantic_classes is None:
        n_semantic_classes = max_label
    elif max_label > n_semantic_classes:
        print(
            f"WARNING: found label {max_label} > n_semantic_classes={n_semantic_classes}. Changing n_semantic_classes to {max_label}"
        )
        n_semantic_classes = max_label

    # Last row and last column are FP and FN
    confusion_matrix = np.zeros((n_semantic_classes + 1, n_semantic_classes + 1))
    false_positives_row = false_negatives_col = -1

    # Each unique prediction ID
    predicted_ids, predicted_id_indexes = np.unique(
        predicted_instances, return_index=True
    )
    # The confidence (of the first point) of each instance
    confidences = prediction_confidences[predicted_id_indexes]
    predicted_confidences = dict(zip(predicted_ids, confidences))

    # Each unique groundtruth instance ID
    groundtruth_ids = np.unique(groundtruth_instances)

    # All pairs of prediction and ground truth
    combinations = product(predicted_ids, groundtruth_ids)

    matches = []
    for prediction, groundtruth in combinations:
        # IoU between each instance mask
        iou = mask_iou(
            predicted_instances == prediction, groundtruth_instances == groundtruth
        )
        # Keep track of matches and their IoU
        if iou > iou_threshold:
            matches.append([groundtruth, prediction, iou])

    matches = np.array(matches)

    # Adapted from
    # https://github.com/svpino/tf_object_detection_cm/blob/91a7373aae338e1d36d9332b8bd6e7cf87dfc3eb/confusion_matrix_tf2.py#L143
    if matches.shape[0] > 0:
        # Sort list of matches by descending IOU so we can remove duplicate detections
        # while keeping the highest IOU entry.
        # TODO: use a pd.DataFrame for this... much more legible  to use sort_values(by='iou')
        matches = matches[matches[:, 2].argsort()[::-1][: len(matches)]]

        # Remove duplicate detections from the list.
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

        # Sort the list again by descending IOU. Removing duplicates doesn't preserve
        # our previous sort.
        matches = matches[matches[:, 2].argsort()[::-1][: len(matches)]]

        # Remember: Matches are [groundtruth, prediction, iou]
        GT, PRED, IOU = 0, 1, 2

        # Remove duplicate ground truths from the list.
        matches = matches[np.unique(matches[:, GT], return_index=True)[1]]

        # Create a groundtruth:prediction dict
        matched_groundtruths = dict(zip(matches[:, GT], matches[:, PRED]))
        # Create a prediction:groudtruth dict
        matched_predictions = dict(zip(matches[:, PRED], matches[:, GT]))

        # Convenience to get IoUs
        groundtruth_ious = dict(zip(matches[:, GT], matches[:, IOU]))
    else:
        matched_groundtruths = {}
        matched_predictions = {}
        groundtruth_ious = {}

    for groundtruth_id in groundtruth_ids:
        # Label is encoded in the ID
        label = groundtruth_id // 1000

        # Skip non-instance labels
        if label in skip_labels:
            continue

        if groundtruth_id in matched_groundtruths:
            # This groundtruth matched a prediction,
            # so get its label and update confusion matrix
            predicted_id = matched_groundtruths[groundtruth_id]
            predicted_label = predicted_id // 1000
            confusion_matrix[int(label), int(predicted_label)] += 1

            confusion_cell_mappings.append(
                dict(
                    instance_gt=groundtruth_id,
                    instance_pred=predicted_id,
                    semantic_gt=label,
                    semantic_pred=predicted_label,
                    iou=groundtruth_ious[groundtruth_id],
                    confidence=predicted_confidences[predicted_id],
                )
            )
        else:
            # Did not match, this is a false negative
            confusion_matrix[int(label), int(false_negatives_col)] += 1
            confusion_cell_mappings.append(
                dict(
                    instance_gt=groundtruth_id,
                    instance_pred=-1,
                    semantic_gt=label,
                    semantic_pred=false_negatives_col,
                    iou=np.nan,
                    confidence=np.nan,
                )
            )

    for predicted_id in predicted_ids:
        # match is symmetrical, so all the matched_predictions are already accounted for;
        # Only unmatched predictions (false positives) are left, so account for them here
        label = predicted_id // 1000

        if label in skip_labels:
            continue

        if predicted_id not in matched_predictions:
            confusion_matrix[int(false_positives_row), int(label)] += 1

            confusion_cell_mappings.append(
                dict(
                    instance_gt=-1,
                    instance_pred=predicted_id,
                    semantic_gt=false_positives_row,
                    semantic_pred=label,
                    iou=np.nan,
                    confidence=predicted_confidences[predicted_id],
                )
            )

    return confusion_matrix, confusion_cell_mappings
