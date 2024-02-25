import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
import pandas as pd
import numpy as np

import os


def save_plot(filename: str, dir_name: str = "output") -> None:
    """
    Saves a plot to the output directory.

    :param filename: The name of the file for the Figure.
    :type filename: str
    """
    path = os.path.join(dir_name, filename)
    
    if not os.path.exists(dir_name): 
        os.makedirs(dir_name) 
    
    plt.savefig(path, bbox_inches="tight")
    print(f"Figured saved to " + path)


def stats_for_tag(y_true, y_pred, tag: str) -> pd.DataFrame:
    """
    Calculate precision, recall, and F1-score for a specific tag.
    
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param tag: The tag for which to calculate the statistics.
    :return: DataFrame containing precision, recall, and F1-score for the specified tag.
    """
    # Select tag with boolean mask
    indexes = np.where(y_true == tag, True, False)

    y_true_tag = y_true[indexes]
    y_pred_tag = y_pred[indexes]

    # Replace correct tag with 1 and incorrect with 0
    y_true_vec = np.ones_like(y_true_tag).astype(int)
    y_pred_vec = np.where(y_pred_tag == tag, 1, 0)

    # Get stats
    precision = metrics.precision_score(y_true_vec, y_pred_vec, zero_division=0)
    recall = metrics.recall_score(y_true_vec, y_pred_vec, zero_division=0)
    f1 = metrics.f1_score(y_true_vec, y_pred_vec, zero_division=0)

    return pd.DataFrame({"tag": [tag], "precision": [precision], "recall": [recall], "f1": [f1]})


def stats_all_tags(y_true, y_pred) -> pd.DataFrame:
    """
    Calculate precision, recall, and F1-score for all unique tags.
    
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: DataFrame containing precision, recall, and F1-score for each unique tag.
    """
    dfs = []
    for tag in np.unique(y_true):
        dfs.append(stats_for_tag(y_true, y_pred, tag))
    dfs.append(stats_macro(y_true, y_pred))

    return pd.concat(dfs)


def stats_macro(y_true, y_pred):
    """
    Calculate macro-averaged precision, recall, and F1-score.
    
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: DataFrame containing macro-averaged precision, recall, and F1-score.
    """
    precision = metrics.precision_score(y_true, y_pred, zero_division=0, average="macro")
    recall = metrics.recall_score(y_true, y_pred, zero_division=0, average="macro")
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0, average="macro")

    return pd.DataFrame({"tag": ["MACRO"], "precision": [precision], "recall": [recall], "f1": [f1]})


def stats_all_splits(model, transform_output_func, x_train, x_valid, x_test, y_train, y_valid, y_test):
    """
    Calculate statistics for training, validation, and test splits.
    
    :param model: The model to evaluate.
    :param transform_output_func: A function to transform model outputs.
    :param x_train: Training input data.
    :param x_valid: Validation input data.
    :param x_test: Test input data.
    :param y_train: True labels for training data.
    :param y_valid: True labels for validation data.
    :param y_test: True labels for test data.
    :return: DataFrame containing statistics for each split.
    """
    y_true_train = transform_output_func(y_train)
    y_true_valid = transform_output_func(y_valid)
    y_true_test = transform_output_func(y_test)

    y_pred_train = transform_output_func(model.predict(x_train))
    y_pred_valid = transform_output_func(model.predict(x_valid))
    y_pred_test = transform_output_func(model.predict(x_test))

    train_df = stats_all_tags(y_true_train, y_pred_train)
    valid_df = stats_all_tags(y_true_valid, y_pred_valid)
    test_df = stats_all_tags(y_true_test, y_pred_test)

    train_df["split"] = "training"
    valid_df["split"] = "validation"
    test_df["split"] = "test"

    return pd.concat([train_df, valid_df, test_df])
    


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        val_targ = tf.cast(val_targ,dtype=tf.float32)
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
          val_targ = np.argmax(val_targ, -1)      
        

        _val_f1 = metrics.f1_score(val_targ, val_predict,average="weighted", zero_division=0)
        _val_recall = metrics.recall_score(val_targ, val_predict,average="weighted", zero_division=0)
        _val_precision = metrics.precision_score(val_targ, val_predict,average="weighted", zero_division=0)

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return