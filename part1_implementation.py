# Part 1: Neural Networks Basics
# Implementation based on part1_neural_networks_basics.md

# ===== Setup and Installation =====
# Import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision

# Enable performance optimizations
print("Setting up TensorFlow performance optimizations...")

# Enable XLA compilation for faster execution
tf.config.optimizer.set_jit(True)

# Set up mixed precision policy for faster training on compatible hardware
mixed_precision.set_global_policy('mixed_float16')

# Set up distribution strategy for multi-GPU training if available
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
    use_strategy = True
except:
    print("No distributed training devices found. Running with default settings.")
    use_strategy = False

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
os.makedirs('results/part_1', exist_ok=True)
os.makedirs('logs', exist_ok=True)

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
    # EMNIST from tfds has a different format - no need for transpose
    plt.imshow(x_train[i], cmap='gray')
    # Labels in EMNIST/letters are 1-indexed (1=a, 2=b, etc.)
    plt.title(f'Label: {chr(y_train[i] + 96)}')  # ASCII: 97='a', 98='b', etc.
    plt.axis('off')
plt.savefig('results/part_1/sample_images.png')  # Save the figure

# Preprocess data
# Check number of classes in the dataset
num_classes = len(np.unique(y_train))
print(f"Detected {num_classes} unique classes")

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape for dense layers (from [height, width, channels] to [flattened])
x_train = x_train.reshape(-1, np.prod(x_train.shape[1:]))
x_test = x_test.reshape(-1, np.prod(x_test.shape[1:]))

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

# ===== 2. Model Implementation =====
# Create simple neural network
def create_simple_nn(input_shape, num_classes):
    """
    Create a simple neural network for EMNIST classification.
    
    Requirements:
    - Must use at least 2 dense layers
    - Must include dropout layers
    - Must use categorical crossentropy loss
    
    Goals:
    - Achieve > 80% accuracy on test set
    - Minimize overfitting using dropout
    - Train efficiently with appropriate batch size
    
    Args:
        input_shape: Shape of input data (should be (784,) for flattened 28x28 images)
        num_classes: Number of output classes (26 for letters)
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # First dense layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Second dense layer
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and compile model
input_shape = x_train.shape[1]  # Get actual input shape from data
if use_strategy:
    with strategy.scope():
        model = create_simple_nn(input_shape=(input_shape,), num_classes=num_classes)
else:
    model = create_simple_nn(input_shape=(input_shape,), num_classes=num_classes)

model.summary()

# ===== 3. Training and Evaluation =====
# Define callbacks
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
        'models/emnist_classifier.keras',
        save_best_only=True
    )
]

# Train model
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)

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
plt.savefig('results/part_1/training_history.png')  # Save the figure

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Get predictions
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Calculate metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
cm = confusion_matrix(true_labels, predicted_labels)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('results/part_1/confusion_matrix.png')  # Save the figure

# ===== 4. Save Metrics =====
# Save metrics to file as required by the tests
metrics = {
    'model': 'emnist_classifier',
    'accuracy': float(test_accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'confusion_matrix': cm.tolist()
}

# Ensure directory exists
os.makedirs('results/part_1', exist_ok=True)

# Save to file in the required format
with open('results/part_1/emnist_classifier_metrics.txt', 'w') as f:
    f.write(f"model: {metrics['model']}\n")
    f.write(f"accuracy: {metrics['accuracy']}\n")
    f.write(f"precision: {metrics['precision']}\n")
    f.write(f"recall: {metrics['recall']}\n")
    f.write(f"f1_score: {metrics['f1_score']}\n")
    f.write(f"confusion_matrix: {metrics['confusion_matrix']}\n")
    f.write("----\n")

print("\nPart 1 implementation complete!")
print(f"Model saved at: models/emnist_classifier.keras")
print(f"Metrics saved at: results/part_1/emnist_classifier_metrics.txt")
