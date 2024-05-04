#HareKrsna

# Generate Character dataset from a given image.
# The program automatically generates a list of characters from a given image
# The input Image should be a picture of document with handwritten characters.

# It is observed that instead of extracting characters and then applying preprocessing,
# Preprocessing can be performed to the full document image and then extract characters.

# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# The above link has good example demonstrations which can be very useful for preprocessing.

import cv2
import os
from PIL import Image
import numpy as np
import argparse

def extract_characters(input_image_path, output_folder):
    # Read the input image
    image = cv2.imread(input_image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each contour
    for i, contour in enumerate(contours):
        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the character from the original image using the bounding box
        character_image = image[y:y+h, x:x+w]

        # Save the character image to the output folder
        output_path = os.path.join(output_folder, f"character_{i}.png")
        cv2.imwrite(output_path, character_image)

        print(f"Character {i} saved as {output_path}")

def delete_small_images(folder_path, min_file_size, min_width_height):
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if it's a file
        if os.path.isfile(file_path):
            # Check file size
            if os.path.getsize(file_path) < min_file_size:
                # Delete the file if smaller than min_file_size
                os.remove(file_path)
                print(f"Deleted {filename} due to small file size")

            else:
                # Check image dimensions
                try:
                    img = Image.open(file_path)
                    width, height = img.size
                    if width < min_width_height or height < min_width_height:
                        # Delete the file if width or height is less than min_width_height
                        os.remove(file_path)
                        print(f"Deleted {filename} due to small dimensions")
                except Exception as e:
                    # Print error if unable to open the image
                    print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare images for character extraction")
    parser.add_argument("input_image", help="Path to the folder containing input images")
    parser.add_argument("dest_folder", help="Path to the folder where the output images will be saved")
    args = parser.parse_args()
    
    # Step 1: Extract character images from ipnut document image.
    # Input image path
    input_image_path = args.input_image    
    output_folder = args.dest_folder                          # Output folder for character images

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Extract characters from the input image
    extract_characters(input_image_path, output_folder)

    # Step 2: Small images are deleted
    # Specify minimum file size in bytes
    min_file_size = 200  # Example: 1 KB

    # Specify minimum width or height of the image
    min_width_height = 10  # Example: 100 pixels

    # Call the function to delete small images
    delete_small_images(output_folder, min_file_size, min_width_height)