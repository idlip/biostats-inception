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

# 3. CREATE DATA AUGMENTATION FUNCTION
def augment_image(image):
    """
    Apply data augmentation to a single image
    """
    # Random rotation
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Random flip
    image = tf.image.random_flip_left_right(image)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Ensure values are in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image

# 4. CREATE TF.DATA DATASETS
def create_datasets(data_dict, batch_size=32, augment_train=True):
    """
    Create tf.data.Dataset objects for training, validation, and testing
    """
    # Preprocess all images
    train_images = preprocess_images(data_dict['train'][0])
    train_labels = data_dict['train'][1]
    val_images = preprocess_images(data_dict['val'][0])
    val_labels = data_dict['val'][1]
    test_images = preprocess_images(data_dict['test'][0])
    test_labels = data_dict['test'][1]
    
    # Create training dataset with augmentation
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    if augment_train:
        train_dataset = train_dataset.map(
            lambda x, y: (augment_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset (no augmentation)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create test dataset (no augmentation)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset, (train_images, train_labels)

# 5. CALCULATE CLASS WEIGHTS
def calculate_class_weights_from_labels(labels):
    """
    Calculate class weights from label array
    """
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=labels
    )
    
    class_weight_dict = dict(zip(unique_classes, class_weights))
    
    # Print class distribution
    class_counts = np.bincount(labels.astype(int))
    print(f"\nClass distribution:")
    print(f"Normal (0): {class_counts[0]} samples")
    print(f"Pneumonia (1): {class_counts[1]} samples")
    print(f"Class weights: {class_weight_dict}")
    
    return class_weight_dict

# 6. BUILD AND COMPILE MODEL (Same as before)
def build_inception_model(num_classes=1):
    """
    Build InceptionV3 model with custom top layers for binary classification
    """
    # Load pre-trained InceptionV3 without top layers
    base_model = InceptionV3(
        input_shape=(299, 299, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom layers
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model, base_model

# 7. TRAINING FUNCTION
def train_model(model, train_dataset, val_dataset, class_weights, steps_per_epoch=None):
    """
    Compile and train the model
    """
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_pneumonia_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# 8. EVALUATION FUNCTION
def evaluate_model_npz(model, test_dataset, test_labels):
    """
    Evaluate model performance
    """
    # Get predictions
    predictions = model.predict(test_dataset)
    y_pred = (predictions > 0.5).astype(int).reshape(-1)
    y_true = test_labels
    
    print("\n=== MODEL PERFORMANCE ===\n")
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=['Normal', 'Pneumonia']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC-AUC Score
    auc_score = roc_auc_score(y_true, predictions)
    print(f"\nROC-AUC Score: {auc_score:.4f}")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print(f"\nSensitivity (Recall for Pneumonia): {sensitivity:.4f}")
    print(f"Specificity (Recall for Normal): {specificity:.4f}")
    
    return {
        'confusion_matrix': cm,
        'auc_score': auc_score,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
