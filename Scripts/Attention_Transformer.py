#import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import joblib
import warnings

warnings.filterwarnings("ignore")

class Transformer:
    def __init__(self):
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential()

        return model
    
    def attention(self):

        return model
    