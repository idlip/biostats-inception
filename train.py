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

# Set parameters
IMG_SIZE = 299  # InceptionV3 input size
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# 1. LOAD DATA FROM NPZ FILE
def load_data_from_npz(npz_path):
    """
    Load data from NPZ file containing numpy arrays
    
    Args:
        npz_path: Path to the NPZ file
    
    Returns:
        Dictionary containing all arrays
    """
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    
    # Extract all arrays
    train_images = data['train_images.npy']
    train_labels = data['train_labels.npy']
    val_images = data['val_images.npy']
    val_labels = data['val_labels.npy']
    test_images = data['test_images.npy']
    test_labels = data['test_labels.npy']
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Image shape: {train_images[0].shape}")
    
    return {
        'train': (train_images, train_labels),
        'val': (val_images, val_labels),
        'test': (test_images, test_labels)
    }

# 2. PREPROCESS IMAGES
def preprocess_images(images, target_size=(299, 299)):
    """
    Preprocess images for InceptionV3
    - Resize to target size if needed
    - Normalize pixel values
    """
    processed_images = []
    
    for img in images:
        # Check if resizing is needed
        if img.shape[:2] != target_size:
            img = tf.image.resize(img, target_size)
        
        # Ensure the image has 3 channels (RGB)
        if len(img.shape) == 2:  # Grayscale
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:  # Single channel
            img = np.concatenate([img] * 3, axis=-1)
        
        processed_images.append(img)
    
    # Convert to numpy array and normalize
    processed_images = np.array(processed_images, dtype=np.float32)
    
    # Normalize to [0, 1] if not already normalized
    if processed_images.max() > 1:
        processed_images = processed_images / 255.0
    
    return processed_images

