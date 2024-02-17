from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


def _record_label(label_dict: dict, word: str, label: str) -> None:
    """
    Helper function to record the label count for a given word.
    :param label_dict: Dictionary to store label counts for each word.
    :param word: The word for which the label count is recorded.
    :param label: The label associated with the word.
    """
    if word in label_dict and label in label_dict[word]:
        label_dict[word][label] += 1
    else:
        label_dict[word] = {label: 1}


class BaselineLabelClassifier(ClassifierMixin, BaseEstimator):
    """
    A sklearn majority classifier which classifies a word with the majority label associated with it.
    Unknown words are assigned the globally most likely label.
    """

    def __init__(self):
        self.word_pos_dict = {}
        self.most_popular_pos_tag = None
        self.is_fitted_ = False

    def fit(self, X: list[str], y: list[str]):
        """
        Fit the BaselineLabelClassifier on the training data.
        :param X: List of words.
        :param y: List of corresponding labels.
        :return: The fitted classifier object.
        """
        all_labels_dict = {}
        inter_dict = {}

        for word, label in zip(X, y):
            _record_label(self.word_pos_dict, word, label)

            # find global maximum label
            if label in all_labels_dict:
                all_labels_dict[label] += 1
            else:
                all_labels_dict[label] = 1

        self.most_popular_pos_tag = max(
            all_labels_dict, key=all_labels_dict.get
        )

        for word in inter_dict.keys():
            self.word_pos_dict = max(word, key=word.get)

        self.is_fitted_ = True
        return self

    def predict(self, words: list[str]):
        """
        Predict the labels for a list of words using the fitted model.
        :param words: List of words to predict labels for.
        :return: List of predicted labels.
        """
        check_is_fitted(self)
        response = []

        for word in words:
            if word in self.word_pos_dict:
                response.append(next(iter(self.word_pos_dict[word])))
            else:
                response.append(self.most_popular_pos_tag)
        return response
    


class SelfAttention(keras.layers.Layer):
    def __init__(
        self, mlp_layers=0, units=[], dropout_rate=0, return_attention=False, **kwargs
    ):
        """
        Self-attention layer for a Keras model.
        
        :param mlp_layers: Number of MLP layers in the attention mechanism.
        :param units: A list containing the number of units in each MLP layer.
        :param dropout_rate: Dropout rate applied to the MLP layers.
        :param return_attention: Whether to return the attention weights along with the output.
        :param kwargs: Additional keyword arguments.
        """
        super(SelfAttention, self).__init__(**kwargs)
        self.mlp_layers = mlp_layers
        self.mlp_units = units
        self.return_attention = return_attention
        self.dropout_rate = dropout_rate
        self.attention_mlp = self.build_mlp()

    def build_mlp(self):
        """
        Build the MLP layers for the attention mechanism.
        :return: The MLP model.
        """
        mlp = Sequential()
        for i in range(self.mlp_layers):
            mlp.add(Dense(self.mlp_units[i], activation="relu"))
            mlp.add(Dropout(self.dropout_rate))
        mlp.add(Dense(1))
        return mlp

    def call(self, x, mask=None):
        """
        Call function for the self-attention layer.
        :param x: Input tensor.
        :param mask: Mask tensor for sequence padding.
        :return: Output tensor with attention weights if return_attention is True.
        """
        a = self.attention_mlp(x)

        if mask is not None:
            mask = tf.keras.backend.cast(mask, tf.keras.backend.floatx())
            a -= 100000.0 * (1.0 - mask)

        a = tf.keras.backend.expand_dims(tf.keras.backend.softmax(a, axis=-1))
        weighted_input = x * a
        result = tf.keras.backend.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result


