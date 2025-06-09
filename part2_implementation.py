#!/usr/bin/env python
# Part 2: CNN Classification Implementation

# ===== Setup and Installation =====
# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, mixed_precision
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configure matplotlib for better visualization
# Use a valid matplotlib style
plt.style.use('seaborn-v0_8')  # Updated version of seaborn style compatible with matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results/part_2', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Setup TensorFlow performance optimizations
print("Setting up TensorFlow performance optimizations...")

# Enable XLA JIT compilation for faster execution
tf.config.optimizer.set_jit(True)

# Enable mixed precision training for compatible GPUs
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Check for available GPUs and set up distribution strategy
use_strategy = False
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Number of devices: {len(physical_devices)}")
if len(physical_devices) > 1:
    # Set up MirroredStrategy for multi-GPU training
    strategy = tf.distribute.MirroredStrategy()
    use_strategy = True
    print(f"Number of devices for distributed training: {strategy.num_replicas_in_sync}")

print("Setup complete!")

# ===== 1. Data Loading and Preprocessing =====
# Load EMNIST dataset using tensorflow_datasets
print("Loading EMNIST dataset...")
ds_train, ds_test = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True)

# Convert to numpy arrays
x_train, y_train = [], []
for image, label in tfds.as_numpy(ds_train):
    x_train.append(image)
    y_train.append(label)

x_test, y_test = [], []
for image, label in tfds.as_numpy(ds_test):
    x_test.append(image)
    y_test.append(label)

# Convert to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Print dataset information
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Plot sample images
plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')  # EMNIST from tfds doesn't need transpose
    plt.title(f'Label: {chr(y_train[i] + 96)}')  # ASCII: 97='a', 98='b', etc.
    plt.axis('off')
plt.savefig('results/part_2/sample_images.png')

# Preprocess data
# Check number of classes in the dataset
num_classes = len(np.unique(y_train))
print(f"Detected {num_classes} unique classes")

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape for CNN input (samples, height, width, channels)
# No need to flatten for CNN - keep the 2D structure
if len(x_train.shape) < 4:
    x_train = x_train[..., np.newaxis]  # Add channel dimension if not present
    x_test = x_test[..., np.newaxis]

print(f"Preprocessed training data shape: {x_train.shape}")
print(f"Preprocessed test data shape: {x_test.shape}")

# EMNIST letters labels start at 1, need to shift to 0-based for one-hot encoding
# Subtract 1 to make 0-indexed
y_train = y_train - 1
y_test = y_test - 1

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Split training data into train and validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

print(f"Preprocessed training data shape: {x_train.shape}")
print(f"Preprocessed validation data shape: {x_val.shape}")
print(f"Preprocessed test data shape: {x_test.shape}")

# ===== 2. CNN Model Implementation =====
# Create CNN using TensorFlow/Keras
def create_cnn_keras(input_shape, num_classes):
    """
    Create a CNN using TensorFlow/Keras.
    
    Requirements:
    - Uses at least 2 convolutional layers
    - Includes pooling and batch normalization
    - Uses categorical crossentropy loss
    
    Goals:
    - Achieve > 85% accuracy on test set
    - Minimize overfitting using batch normalization and dropout
    - Train efficiently with appropriate batch size and learning rate
    
    Args:
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and compile model
input_shape = x_train.shape[1:]  # Get actual input shape from data
if use_strategy:
    with strategy.scope():
        model = create_cnn_keras(input_shape=input_shape, num_classes=num_classes)
else:
    model = create_cnn_keras(input_shape=input_shape, num_classes=num_classes)

model.summary()

# ===== 3. Training and Evaluation =====
# Define callbacks for training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'models/cnn_keras.keras',
        save_best_only=True
    )
]

# Train model
print("\nTraining CNN model...")
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=64,  # Larger batch size for CNN
    callbacks=callbacks
)

# ===== 4. Visualization and Analysis =====
print("\nGenerating visualizations...")
# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Plot loss
ax2.plot(history.history['loss'], label='Training')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.savefig('results/part_2/training_curves.png')

# ===== 5. Model Evaluation =====
print("\nEvaluating model on test set...")
# Evaluate model on test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Get predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate metrics
precision = precision_score(y_test_classes, y_pred_classes, average='macro')
recall = recall_score(y_test_classes, y_pred_classes, average='macro')
f1 = f1_score(y_test_classes, y_pred_classes, average='macro')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Calculate and plot confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('results/part_2/confusion_matrix.png')

# ===== 6. Save Model and Metrics =====
# Save model metrics to file
metrics = {
    'accuracy': float(test_acc),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'confusion_matrix': cm.tolist()
}

# Save metrics to file
with open('results/part_2/cnn_keras_metrics.txt', 'w') as f:
    f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
    f.write(f"Precision: {metrics['precision']:.4f}\n")
    f.write(f"Recall: {metrics['recall']:.4f}\n")
    f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
    f.write(f"Confusion Matrix: See confusion_matrix.png")

print("\nPart 2 implementation complete!")
print(f"Model saved at: models/cnn_keras.keras")
print(f"Metrics saved at: results/part_2/cnn_keras_metrics.txt")
