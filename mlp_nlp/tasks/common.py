import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


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