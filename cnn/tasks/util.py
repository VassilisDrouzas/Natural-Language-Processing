import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np

from typing import Callable
import os


def save_plot(filename: str, dir_name: str = "output") -> None:
    """
    Saves a plot to the output directory.

    :param filename: The name of the file for the Figure.
    :type filename: str
    :param dir_name: The directory where the plot will be saved. Default is "output".
    :type dir_name: str
    """
    path = os.path.join(dir_name, filename)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.savefig(path, bbox_inches="tight")
    print(f"Figure saved to " + path)


def pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the area under the precision-recall curve.

    :param y_true: True labels.
    :type y_true: np.ndarray
    :param y_pred: Predicted labels.
    :type y_pred: np.ndarray
    :return: Area under the precision-recall curve.
    :rtype: float
    """
    prec, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    auc = metrics.auc(recall, prec)
    return 0.0 if np.isnan(auc) else auc


def pr_auc_macro(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray
) -> float:
    """
    Calculate macro-averaged area under the precision-recall curve.

    :param y_true: True labels.
    :type y_true: np.ndarray
    :param y_pred: Predicted labels.
    :type y_pred: np.ndarray
    :param labels: Unique labels.
    :type labels: np.ndarray
    :return: Macro-averaged area under the precision-recall curve.
    :rtype: float
    """
    macro_scores = []

    for i, label in enumerate(np.unique(labels)):
        label_mask = labels == label
        y_true_tag = y_true[label_mask]
        y_pred_tag = y_pred[label_mask]
        y_true_vec = np.ones_like(y_true_tag).astype(int)
        y_pred_vec = np.where(y_pred_tag == label, 1, 0).astype(int)
        macro_scores.append(pr_auc(y_true_vec, y_pred_vec[:, i]))
    return np.mean(macro_scores)


def stats_for_tag(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tag: str,
    proba: np.ndarray,
    tag_index: int,
) -> pd.DataFrame:
    """
    Calculate precision, recall, and F1-score for a specific tag.

    :param y_true: True labels.
    :type y_true: np.ndarray
    :param y_pred: Predicted labels.
    :type y_pred: np.ndarray
    :param tag: The tag for which to calculate the statistics.
    :type tag: str
    :param proba: Probabilities predicted by the model.
    :type proba: np.ndarray
    :param tag_index: Index of the tag.
    :type tag_index: int
    :return: DataFrame containing precision, recall, and F1-score for the specified tag.
    :rtype: pd.DataFrame
    """
    # Select tag with boolean mask
    indexes = np.where(y_true == tag, True, False)

    y_true_tag = y_true[indexes]
    y_pred_tag = y_pred[indexes]

    # Replace correct tag with 1 and incorrect with 0
    y_true_vec = np.ones_like(y_true_tag).astype(int)
    y_pred_vec = np.where(y_pred_tag == tag, 1, 0).astype(int)

    # Get stats
    accuracy = metrics.accuracy_score(y_true_vec, y_pred_vec)
    precision = metrics.precision_score(y_true_vec, y_pred_vec, zero_division=0)
    recall = metrics.recall_score(y_true_vec, y_pred_vec, zero_division=0)
    f1 = metrics.f1_score(y_true_vec, y_pred_vec, zero_division=0)

    if proba is None:
        auc = "-"
    else:
        proba_tag = proba[indexes]
        auc = pr_auc(y_true_vec, proba_tag[:, tag_index])

    return pd.DataFrame(
        {
            "accuracy": [accuracy],
            "tag": [tag],
            "precision": [precision],
            "recall": [recall],
            "f1": [f1],
            "auc": [auc],
        }
    )


def stats_macro(
    y_true: np.ndarray, y_pred: np.ndarray, tags: np.ndarray, proba: np.ndarray
) -> pd.DataFrame:
    """
    Calculate macro-averaged precision, recall, and F1-score.

    :param y_true: True labels.
    :type y_true: np.ndarray
    :param y_pred: Predicted labels.
    :type y_pred: np.ndarray
    :param tags: Unique labels.
    :type tags: np.ndarray
    :param proba: Probabilities predicted by the model.
    :type proba: np.ndarray
    :return: DataFrame containing macro-averaged precision, recall, and F1-score.
    :rtype: pd.DataFrame
    """
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(
        y_true, y_pred, labels=tags, zero_division=0, average="macro"
    )
    recall = metrics.recall_score(
        y_true, y_pred, labels=tags, zero_division=0, average="macro"
    )
    f1 = metrics.f1_score(
        y_true, y_pred, labels=tags, zero_division=0, average="macro"
    )

    if proba is None:
        auc = "-"
    else:
        auc = pr_auc_macro(y_true, proba, tags)

    return pd.DataFrame(
        {
            "tag": ["MACRO"],
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1": [f1],
            "auc": [auc],
        }
    )


def stats_all_tags(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, proba: np.ndarray
) -> pd.DataFrame:
    """
    Calculate precision, recall, and F1-score for all unique tags.

    :param y_true: True labels.
    :type y_true: np.ndarray
    :param y_pred: Predicted labels.
    :type y_pred: np.ndarray
    :param labels: Unique labels.
    :type labels: np.ndarray
    :param proba: Probabilities predicted by the model.
    :type proba: np.ndarray
    :return: DataFrame containing precision, recall, and F1-score for each unique tag.
    :rtype: pd.DataFrame
    """
    dfs = []
    for i, tag in enumerate(np.unique(y_true)):
        dfs.append(stats_for_tag(y_true, y_pred, tag, proba, i))

    dfs.append(stats_macro(y_true,

 y_pred, labels, proba))

    return pd.concat(dfs)


def stats_by_label(
    y_true_train: np.ndarray,
    y_true_valid: np.ndarray,
    y_true_test: np.ndarray,
    y_train_pred: np.ndarray,
    y_valid_pred: np.ndarray,
    y_test_pred: np.ndarray,
    y_train_proba: np.ndarray,
    y_valid_proba: np.ndarray,
    y_test_proba: np.ndarray,
) -> pd.DataFrame:
    """
    Generate statistics for training, validation, and test splits.

    :param y_true_train: True labels for training data.
    :type y_true_train: np.ndarray
    :param y_true_valid: True labels for validation data.
    :type y_true_valid: np.ndarray
    :param y_true_test: True labels for test data.
    :type y_true_test: np.ndarray
    :param y_train_pred: Predicted labels for training data.
    :type y_train_pred: np.ndarray
    :param y_valid_pred: Predicted labels for validation data.
    :type y_valid_pred: np.ndarray
    :param y_test_pred: Predicted labels for test data.
    :type y_test_pred: np.ndarray
    :param y_train_proba: Probabilities predicted by the model for training data.
    :type y_train_proba: np.ndarray
    :param y_valid_proba: Probabilities predicted by the model for validation data.
    :type y_valid_proba: np.ndarray
    :param y_test_proba: Probabilities predicted by the model for test data.
    :type y_test_proba: np.ndarray
    :return: DataFrame containing statistics for each split.
    :rtype: pd.DataFrame
    """
    train_df = stats_all_tags(
        y_true_train, y_train_pred, y_true_train, y_train_proba
    )
    valid_df = stats_all_tags(
        y_true_valid, y_valid_pred, y_true_valid, y_valid_proba
    )
    test_df = stats_all_tags(y_true_test, y_test_pred, y_true_test, y_test_proba)

    train_df["split"] = "training"
    valid_df["split"] = "validation"
    test_df["split"] = "test"

    return pd.concat([train_df, valid_df, test_df])


def get_statistics(
    model,
    transform_output_func,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    calculate_proba: bool,
    time_distributed_func: Callable = lambda x: x,
):
    """
    Calculate statistics for training, validation, and test splits.

    :param model: The model to evaluate.
    :type model: Any
    :param transform_output_func: A function to transform model outputs.
    :type transform_output_func: Callable
    :param x_train: Training input data.
    :type x_train: np.ndarray
    :param x_valid: Validation input data.
    :type x_valid: np.ndarray
    :param x_test: Test input data.
    :type x_test: np.ndarray
    :param y_train: True labels for training data.
    :type y_train: np.ndarray
    :param y_valid: True labels for validation data.
    :type y_valid: np.ndarray
    :param y_test: True labels for test data.
    :type y_test: np.ndarray
    :param calculate_proba: Flag to indicate whether to calculate probabilities.
    :type calculate_proba: bool
    :param time_distributed_func: A function to apply to the model outputs. Default is identity.
    :type time_distributed_func: Callable, optional
    :return: DataFrame containing statistics for each split.
    :rtype: pd.DataFrame
    """
    y_true_train = transform_output_func(y_train)
    y_true_valid = transform_output_func(y_valid)
    y_true_test = transform_output_func(y_test)

    raw_pred_train = model.predict(x_train)
    raw_pred_valid = model.predict(x_valid)
    raw_pred_test = model.predict(x_test)

    y_train_proba = time_distributed_func(raw_pred_train)
    y_valid_proba = time_distributed_func(raw_pred_valid)
    y_test_proba = time_distributed_func(raw_pred_test)

    y_pred_train = transform_output_func(raw_pred_train)
    y_pred_valid = transform_output_func(raw_pred_valid)
    y_pred_test = transform_output_func(raw_pred_test)

    if not calculate_proba:
        y_train_proba = y_valid_proba = y_test_proba = None

    return stats_by_label(
        y_true_train,
        y_true_valid,
        y_true_test,
        y_pred_train,
        y_pred_valid,
        y_pred_test,
        y_train_proba,
        y_valid_proba,
        y_test_proba,
    )
