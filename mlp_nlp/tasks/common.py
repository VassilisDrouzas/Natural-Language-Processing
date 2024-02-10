import os
import matplotlib.pyplot as plt
from tensorflow import keras
import keras.backend as K


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


def reset_weights_keras(model: keras.Model) -> keras.Model:
    """
    Reset the weights of any keras model.

    :param model: The model to be reset.
    :return: the same model with random weights
    """
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'): 
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)  