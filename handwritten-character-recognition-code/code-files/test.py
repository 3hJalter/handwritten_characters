import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
#from keras.optimizers import Adam
#from tensorflow.keras.optimizers import Adam
from keras.optimizer_v2 import adam
#from keras.optimizers import SGD
from keras.optimizers import gradient_descent_v2 
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
#from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.utils import shuffle

data = pd.read_csv(r"D:\Code\Code\Workspace\Python\archive\A_Z Handwritten Data\A_Z Handwritten Data.csv").astype('float32')
print(data.head(10))