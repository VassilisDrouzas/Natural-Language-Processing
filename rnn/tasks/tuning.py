import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Bidirectional, GRU, LayerNormalization, Dropout, Dense
from keras.optimizers import Adam

from tasks.models import SelfAttention


def tune_self_attention_gru(
    hp: kt.HyperParameters,
    input_layers: list[tf.keras.layers.Layer],
    output_size: int,
    bidirect_num_min: int,
    bidirect_num_max: int,
    self_attention_num_min: int,
    self_attention_num_max: int,
    self_attention_neurons_min: int,
    self_attention_neurons_max: int,
    self_attention_neurons_step: int,
    metrics: list,
    loss: str,
    gru_size: int = 300,
) -> keras.Model:
    """
    Build and compile a Keras model with bidirectional GRU layers and self-attention mechanism.

    :param hp: Hyperparameters for Keras Tuner.
    :param input_layers: List of input layers for the model.
    :param output_size: Output size of the model.
    :param bidirect_num_min: Minimum number of bidirectional GRU layers.
    :param bidirect_num_max: Maximum number of bidirectional GRU layers.
    :param self_attention_num_min: Minimum number of self-attention layers.
    :param self_attention_num_max: Maximum number of self-attention layers.
    :param self_attention_neurons_min: Minimum number of neurons in self-attention layers.
    :param self_attention_neurons_max: Maximum number of neurons in self-attention layers.
    :param self_attention_neurons_step: Step size for the number of neurons in self-attention layers.
    :param metrics: List of evaluation metrics.
    :param loss: Loss function for model training.
    :param gru_size: Size of the GRU layers.
    :return: Compiled Keras model.
    """
    model = tf.keras.Sequential()

    # Bidirectional GRUs
    for layer in input_layers:
        model.add(layer)

    use_layer_norm = hp.Boolean("use-layer-norm")
    variational_dropout = hp.Choice("variational-dropout", [0.0, 0.33])

    # Don't mix both
    if use_layer_norm:
        variational_dropout = 0

    for _ in range(
        hp.Int(
            name="bidirectional-layers",
            min_value=bidirect_num_min,
            max_value=bidirect_num_max,
        )
        - 1
    ):
        model.add(
            Bidirectional(
                GRU(
                    gru_size,
                    return_sequences=True,
                    recurrent_dropout=variational_dropout,
                )
            )
        )

        if use_layer_norm:
            model.add(LayerNormalization())
        else:
            model.add(Dropout(0.33))

    model.add(
        Bidirectional(
            GRU(
                gru_size,
                return_sequences=False,
                recurrent_dropout=variational_dropout,
            )
        )
    )

    # Self-attention

    mlp_layers = hp.Int(
        "self-attention-layers",
        min_value=self_attention_num_min,
        max_value=self_attention_num_max,
    )
    mlp_units = [
        hp.Int(
            "self-attention-neurons",
            min_value=self_attention_neurons_min,
            max_value=self_attention_neurons_max,
            step=self_attention_neurons_step,
        )
        for i in range(mlp_layers)
    ]
    model.add(
        SelfAttention(mlp_layers=mlp_layers, units=mlp_units, dropout_rate=0.33)
    )

    # Output layer
    if output_size == 1:
        model.add(Dense(output_size, activation="sigmoid"))
    else:
        model.add(Dense(output_size, activation="softmax"))

    # Model compilation
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-3, 1e-4])
    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=hp_learning_rate),
        metrics=metrics,
    )

    return model
