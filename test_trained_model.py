#HareKrsna

# Program to load a tensorflow model and test it.

# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import argparse

# # Load an image for prediction
# image = cv2.imread('Bhanu_Numerals_1.jpg', cv2.IMREAD_GRAYSCALE)

# # Preprocess the image (if necessary)
# # Resize, normalize, etc.

# # Load the model
# loaded_model = load_model('model_output.h5')

# # Make predictions using the loaded model
# # Assuming 'loaded_model' is the loaded model
# # Ensure the input shape matches the model's input shape
# input_image = np.expand_dims(image, axis=0)  # Add batch dimension if necessary
# prediction = loaded_model.predict(input_image)

# # Process the prediction results as needed
# # For example, get the predicted class label
# predicted_class = np.argmax(prediction)

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

def preprocess_image(image):
    # Preprocess the image here if necessary (e.g., resize, normalize)
    # Example: Resize the image to match the input shape of the model
    resized_image = cv2.resize(image, (32, 32))  # Assuming the model expects 32x32 images
    # Normalize pixel values to [0, 1]
    normalized_image = resized_image / 255.0
    return normalized_image

def predict_image(model, image):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image)
    # Add batch dimension to match the model's input shape
    input_image = np.expand_dims(preprocessed_image, axis=0)
    # Make predictions
    prediction = model.predict(input_image)
    return prediction

def print_predictions(prediction):
    # Print the predicted probabilities for each class
    print("Predicted Probabilities:")
    for i, prob in enumerate(prediction[0]):
        print(f"Class {i}: {prob:.4f}")

    # Print the predicted class label
    predicted_class = np.argmax(prediction)
    print(f"Predicted Class: {predicted_class}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict image using a trained model')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    # Load the model
    loaded_model = load_model('model_output.h5')

    # Load and preprocess the input image
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    
    # Make predictions
    prediction = predict_image(loaded_model, image)
    
    # Print predictions
    print_predictions(prediction)

if __name__ == "__main__":
    main()
