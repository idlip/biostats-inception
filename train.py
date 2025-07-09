# Get the data
# To use in google colab via cloud
import kagglehub
path = kagglehub.dataset_download("rijulshr/pneumoniamnist")
print("Path to dataset files:", path)
path = f"{path}/pneumoniamnist.npz"

# or manual load file locally
path = "~/projects/biostats/pneumoniamnist.npz"

# import modules and libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

