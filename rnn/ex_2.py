#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:28:06 2024

@author: dimits
"""

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

INPUT_DIR = "input/UD_English-EWT"
OUTPUT_DIR = "output"
INTERMEDIATE_DIR = "intermediate"