
# Program to train the model.
#
# Usage :-
# python train_model.py <source_folder_name> 
#
# Example :
# python train_model.py p1_chars_dataset
#
# The program trains the model, and the model will be saved as 'model_output.h5'.

# Code inspired by assistance from ChatGPT
# OpenAI GPT-3.5 model trained by OpenAI
# https://openai.com

# Load dataset
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import argparse

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            # Read the image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize the image if needed
                # img = cv2.resize(img, (desired_width, desired_height))
                images.append(img)
                # Extract label from filename or use any other method to get labels
                label = filename.split("-")[0]  # Example: label is before the first underscore in filename
                labels.append(label)
    return np.array(images), np.array(labels)

def train_model(folder): 
    # Load images and labels
    images, labels = load_images_from_folder(folder)

    # Check the shape of the loaded images
    print("Shape of images array:", images.shape)

    # Check the unique labels
    print("Unique labels:", np.unique(labels))

    # Assuming labels are strings, convert them to numerical labels
    label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}
    y_train = np.array([label_to_index[label] for label in labels])

    # Assuming images are already preprocessed and resized
    X_train = images

    # Optionally, normalize pixel values to [0, 1]
    X_train = X_train / 255.0

    # Split dataset into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Check the shapes of X_train and y_train
    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)

    # Convert labels to one-hot encoded format
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=10)

    # Define the CNN architecture
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train_one_hot, epochs=15, validation_data=(X_val, y_val_one_hot))

    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_val, y_val_one_hot)
    print('Validation accuracy:', val_acc)

    # Save model
    model.save("model_output.h5")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with the given character images dataset")
    parser.add_argument("source_folder", help="Path to the folder containing input images")
    # parser.add_argument("epochs", help="Specify number of epochs")
    args = parser.parse_args()

    train_model(args.source_folder)

    print("Model training complete and saved.")